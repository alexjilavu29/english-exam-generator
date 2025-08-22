import os
import json
import re
import time
from typing import List, Dict, Optional, Tuple
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    import google.generativeai as genai
except ImportError:
    genai = None

DEFAULT_PROMPT = """
You are given an English exam question that contains a gap to be filled with the word "{correct_answer}".

Original question text (with the gap already shown as dots): {question_body}

Answer options (index aligned with the original): {answers_json}

The correct answer is "{correct_answer}".
{context_section}
**TASK**  
Write **exactly {num_variations}** alternative formulations of this question.  
Hard rules:  
1. Keep the gap as seven dots: "......."  
2. Make sure "{correct_answer}" is still the only correct answer.  
3. Each formulation must differ significantly from the original and from each other while maintaining the same difficulty.

**OUTPUT FORMAT (MANDATORY)**  
Respond **only** with a valid JSON array of {num_variations} strings, for example:  

["First reformulation with .......", "Second reformulation with .......", "Third reformulation with ......."]

No additional JSON keys, Markdown, or comments.
"""

CONTEXT_PROMPT_SECTION = """

**CONTEXT**
Here are some previously reformatted questions for reference. Use these ONLY to understand the reformatting style and quality expected. 
DO NOT copy or mix any text from these examples into your reformulations of the current question:

{context_examples}

IMPORTANT: Create reformulations ONLY for the current question provided above. Do not use any text from the context examples.
"""

class AIFormatter:
    def __init__(self, openai_api_key=None, gemini_api_key=None, model="gpt-4o", prompt=None, 
                 num_variations=3, use_context=False, context_limit=10):
        """Initialize the AI formatter with support for multiple providers."""
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.model = model
        self.prompt = prompt or DEFAULT_PROMPT
        self.num_variations = num_variations
        self.use_context = use_context
        self.context_limit = context_limit
        
        # Initialize clients
        self.openai_client = None
        if openai_api_key and OpenAI:
            self.openai_client = OpenAI(api_key=openai_api_key)
        
        if gemini_api_key and genai:
            genai.configure(api_key=gemini_api_key)
    
    def is_gemini_model(self, model: str) -> bool:
        """Check if the model is a Gemini model."""
        return model.startswith('gemini-') or model.startswith('gemini/')
    
    def is_openai_model(self, model: str) -> bool:
        """Check if the model is an OpenAI model."""
        openai_models = ['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-4', 'gpt-3.5', 'o1', 'o3', 'o4-mini']
        return any(model.startswith(m) for m in openai_models)
    
    def set_openai_api_key(self, api_key: str):
        """Set or update the OpenAI API key."""
        self.openai_api_key = api_key
        if api_key and OpenAI:
            self.openai_client = OpenAI(api_key=api_key)
    
    def set_gemini_api_key(self, api_key: str):
        """Set or update the Gemini API key."""
        self.gemini_api_key = api_key
        if api_key and genai:
            genai.configure(api_key=api_key)
    
    def set_model(self, model: str):
        """Set or update the AI model to use."""
        self.model = model
    
    def set_prompt(self, prompt: str):
        """Set or update the AI prompt template."""
        self.prompt = prompt or DEFAULT_PROMPT
    
    def set_num_variations(self, num: int):
        """Set the number of variations to generate."""
        self.num_variations = max(1, min(num, 10))  # Limit between 1 and 10
    
    def set_use_context(self, use: bool):
        """Enable or disable context usage."""
        self.use_context = use
    
    def set_context_limit(self, limit: int):
        """Set the maximum number of context examples to use."""
        self.context_limit = max(1, limit)  # Minimum of 1, no upper limit
    
    def get_context_examples(self, questions: List[Dict], current_question: Dict) -> str:
        """Get formatted context examples from previously reformatted questions."""
        context_questions = []
        
        # Find questions with "Reformatted with AI" tag
        for q in questions:
            if q.get('tags') and 'Reformatted with AI' in q.get('tags', []):
                # Skip the current question by comparing multiple fields to ensure uniqueness
                current_body = current_question.get('body', '')
                current_answers = current_question.get('answers', [])
                current_old_text = current_question.get('old_text', '')
                
                q_body = q.get('body', '')
                q_answers = q.get('answers', [])
                q_old_text = q.get('old_text', '')
                
                # Skip if this is the same question (check multiple fields)
                if (q_body == current_body or 
                    q_old_text == current_body or 
                    (current_old_text and q_body == current_old_text) or
                    (q_answers == current_answers and len(current_answers) > 0)):
                    continue
                    
                context_questions.append(q)
        
        # Limit the number of context examples
        context_questions = context_questions[:self.context_limit]
        
        if not context_questions:
            return ""
        
        # Format context examples more safely - only show the reformatted versions
        # to avoid AI model confusion with original texts from other questions
        examples = []
        for i, q in enumerate(context_questions, 1):
            reformatted = q.get('body', 'N/A')
            # Only show the reformatted version, not the original text
            # to prevent AI from mixing up original texts between questions
            examples.append(f"Example {i} (reformatted style): {reformatted}")
        
        return "\n\n".join(examples)
    
    def _parse_reformulations(self, raw_text: str) -> List[str]:
        """Extract reformulated questions from the model output."""
        # Remove markdown code blocks if present
        raw_text = re.sub(r'```json\s*', '', raw_text)
        raw_text = re.sub(r'```\s*$', '', raw_text)
        
        # Try direct JSON parsing
        try:
            data = json.loads(raw_text)
            if isinstance(data, list):
                return [str(item) for item in data]
        except json.JSONDecodeError:
            pass

        # Fallback – find the first JSON array in the string
        # Look for array that starts with [ and ends with ]
        match = re.search(r'\[\s*"[^"]*"(?:\s*,\s*"[^"]*")*\s*\]', raw_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return [str(item) for item in data]
            except json.JSONDecodeError:
                pass

        # Final fallback - try to extract individual quoted strings
        strings = re.findall(r'"([^"]+)"', raw_text)
        if strings and len(strings) >= self.num_variations:
            return strings[:self.num_variations]

        raise ValueError(f"Could not extract JSON array with reformulations from: {raw_text[:200]}...")
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API to generate reformulations."""
        if not self.openai_client:
            raise ValueError("OpenAI API key is not set or OpenAI library not installed")
        
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a specialist in creating alternative formulations of English language exam questions. Your task is to provide diverse reformulations of questions while preserving their meaning and answer. Always respond with a valid JSON array."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 1.0
        }
        
        # Check if it's a reasoning model (o1, o3, etc.)
        if self.model.startswith('o'):
            params['max_completion_tokens'] = 2000
        else:
            params['max_tokens'] = 2000
            params['top_p'] = 0.95
            params['frequency_penalty'] = 0.8
            params['presence_penalty'] = 0.6
        
        response = self.openai_client.chat.completions.create(**params)
        return response.choices[0].message.content.strip()
    
    def _call_gemini(self, prompt: str, question_data: Optional[Dict] = None) -> str:
        """Call Gemini API to generate reformulations."""
        if not genai:
            raise ValueError("Gemini API key is not set or google-generativeai library not installed")
        
        # Configure generation settings
        generation_config = {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2000,
        }
        
        # Configure safety settings to be more permissive for educational content
        try:
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            safety_settings = [
                {
                    "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": HarmBlockThreshold.BLOCK_NONE
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE
                }
            ]
        except ImportError:
            # Fallback to string-based settings
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
        
        # Create the model
        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Add system instruction - more explicit about JSON format
        full_prompt = f"""You are an educational AI assistant helping to create alternative formulations of English language exam questions for educational purposes. This is for legitimate educational content creation.

Your task is to provide diverse reformulations of questions while preserving their meaning and answer options. 

IMPORTANT: Respond ONLY with a valid JSON array of strings. No markdown formatting, no additional text.

{prompt}"""
        
        try:
            response = model.generate_content(full_prompt)
            
            # Check if response has candidates and if they were blocked
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    print(f"Gemini finish_reason: {candidate.finish_reason}")
                    
                if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                    if candidate.content.parts:
                        text_content = candidate.content.parts[0].text
                        return text_content.strip()
            
            # If we get here, the response was likely blocked
            print(f"Gemini response blocked or empty. Response: {response}")
            print(f"Candidates: {getattr(response, 'candidates', None)}")
            
            # Try a simpler, more direct prompt as fallback
            if question_data:
                simple_prompt = f"""Create {self.num_variations} different versions of this English exam question. Keep the same answer options and meaning.

Original: {question_data.get('body', '')}
Correct answer: {question_data.get('answers', [''])[question_data.get('correct_answer', 0)]}

Return only a JSON array like: ["version 1", "version 2", "version 3"]"""
            else:
                simple_prompt = f"""Create {self.num_variations} alternative versions of the given exam question as a JSON array."""
            
            print("Trying simpler prompt...")
            simple_response = model.generate_content(simple_prompt)
            
            if hasattr(simple_response, 'candidates') and simple_response.candidates:
                candidate = simple_response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                    if candidate.content.parts:
                        return candidate.content.parts[0].text.strip()
            
            raise ValueError("Gemini API response was blocked by safety filters")
            
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            raise
    
    def reformat_question(self, question: Dict, all_questions: Optional[List[Dict]] = None) -> List[str]:
        """Generate alternative formulations of the question."""
        # Extract question data
        question_body = question['body']
        answers = question['answers']
        correct_answer_index = question['correct_answer']
        correct_answer = answers[correct_answer_index]
        
        # Prepare context section if enabled
        context_section = ""
        if self.use_context and all_questions:
            context_examples = self.get_context_examples(all_questions, question)
            if context_examples:
                context_section = CONTEXT_PROMPT_SECTION.format(context_examples=context_examples)
        
        # Prepare prompt
        prompt = self.prompt.format(
            correct_answer=correct_answer,
            question_body=question_body,
            answers_json=json.dumps(answers, indent=2),
            num_variations=self.num_variations,
            context_section=context_section
        )
        
        try:
            # Call appropriate API based on model
            if self.is_gemini_model(self.model):
                raw_response = self._call_gemini(prompt, question)
            elif self.is_openai_model(self.model):
                raw_response = self._call_openai(prompt)
            else:
                raise ValueError(f"Unknown model type: {self.model}")
            
            print(f"API Response: {raw_response}")  # Debug log

            # Parse reformulations
            reformulations = self._parse_reformulations(raw_response)
            
            # Ensure we have the right number of reformulations
            if len(reformulations) < self.num_variations:
                # Add fallback reformulations
                while len(reformulations) < self.num_variations:
                    reformulations.append(
                        f"Alternative {len(reformulations)+1}: Fill the gap with '{correct_answer}' in: {question_body}"
                    )
            elif len(reformulations) > self.num_variations:
                reformulations = reformulations[:self.num_variations]

            return reformulations
            
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            # Return meaningful fallback reformulations
            timestamp = int(time.time())
            fallbacks = []
            
            # Create better fallback variations
            base_templates = [
                "Complete the sentence with '{correct_answer}': {question_body}",
                "Fill in the gap using '{correct_answer}': {question_body}",
                "The word '{correct_answer}' belongs in: {question_body}",
                "Choose '{correct_answer}' to complete: {question_body}",
                "Use '{correct_answer}' in the following: {question_body}",
                "The correct word is '{correct_answer}' for: {question_body}",
                "Fill the blank with '{correct_answer}': {question_body}",
                "Insert '{correct_answer}' to complete: {question_body}",
                "The answer '{correct_answer}' fits in: {question_body}",
                "Place '{correct_answer}' in this sentence: {question_body}"
            ]
            
            for i in range(self.num_variations):
                template = base_templates[i % len(base_templates)]
                fallback = template.format(correct_answer=correct_answer, question_body=question_body)
                fallbacks.append(fallback)
            
            return fallbacks