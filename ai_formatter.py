import os
import json
import re
from openai import OpenAI
import time

DEFAULT_PROMPT = """
You are given an English exam question that contains a gap to be filled with the word "{correct_answer}".

Original question text (with the gap already shown as dots): {question_body}

Answer options (index aligned with the original): {answers_json}

The correct answer is "{correct_answer}".

**TASK**  
Write **exactly three** alternative formulations of this question.  
Hard rules:  
1. Keep the gap as seven dots: "......."  
2. Make sure "{correct_answer}" is still the only correct answer.  
3. Each formulation must differ significantly from the original and from each other while maintaining the same difficulty.

**OUTPUT FORMAT (MANDATORY)**  
Respond **only** with a valid JSON array of three strings, for example:  

["First reformulation with .......", "Second reformulation with .......", "Third reformulation with ......."]

No additional JSON keys, Markdown, or comments.
"""

class AIFormatter:
    def __init__(self, api_key=None, model="gpt-4o", prompt=None):
        """Initialize the AI formatter with API key, model, and prompt."""
        self.api_key = api_key
        self.model = model
        self.prompt = prompt or DEFAULT_PROMPT
        self.client = None
        if api_key:
            self.client = OpenAI(api_key=api_key)
        
    def set_api_key(self, api_key):
        """Set or update the OpenAI API key."""
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
    
    def set_model(self, model):
        """Set or update the AI model to use."""
        self.model = model
    
    def set_prompt(self, prompt):
        """Set or update the AI prompt template."""
        self.prompt = prompt or DEFAULT_PROMPT
    
    def _parse_reformulations(self, raw_text):
        """Extract exactly three reformulated questions from the model output.

        The model is asked to reply with a JSON array of three strings, but some
        models prepend/append extra reasoning text.  We try strict JSON first,
        then look for the first JSON array in the text.

        Args:
            raw_text (str): Full text returned by the model.

        Returns:
            list[str]: Three reformulated questions.

        Raises:
            ValueError: If no JSON array can be found.
        """
        # Attempt direct JSON parsing
        try:
            data = json.loads(raw_text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Fallback – find the first JSON array in the string
        match = re.search(r"\[[\s\S]*?\]", raw_text)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not extract JSON array with reformulations")
    
    def reformat_question(self, question):
        """Generate three alternative formulations of the question.
        
        Args:
            question: A dictionary containing the question data.
            
        Returns:
            A list of three reformatted question texts.
        """
        if not self.api_key or not self.client:
            raise ValueError("OpenAI API key is not set")
        
        # Extract question body and answers
        question_body = question['body']
        answers = question['answers']
        correct_answer_index = question['correct_answer']
        correct_answer = answers[correct_answer_index]
        
        # Add a timestamp to ensure unique requests
        timestamp = int(time.time())
        
        # Prepare prompt for the AI
        prompt = self.prompt.format(
            correct_answer=correct_answer,
            question_body=question_body,
            answers_json=json.dumps(answers, indent=2)
        )
        
        try:
            reasoning_models = ['o4-mini', 'o3', 'gpt-4.1']
            is_reasoning_model = any(m in self.model for m in reasoning_models)

            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a specialist in creating alternative formulations of English language exam questions. Your task is to provide diverse reformulations of questions while preserving their meaning and answer."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 1.0
            }

            if is_reasoning_model:
                params['max_completion_tokens'] = 1000
            else:
                # These parameters are not supported by some reasoning models
                params['max_tokens'] = 1000
                params['top_p'] = 0.95
                params['frequency_penalty'] = 0.8
                params['presence_penalty'] = 0.6
            
            response = self.client.chat.completions.create(**params)
            
            # Parse and validate the response
            reformulations_text = response.choices[0].message.content.strip()
            print(f"API Response: {reformulations_text}")  # Debug log

            try:
                reformulations = self._parse_reformulations(reformulations_text)
            except Exception as parse_err:
                print(f"Error extracting JSON array: {parse_err}")
                # Fallback to simple line‑based parsing
                reformulations = [ln.strip() for ln in reformulations_text.splitlines() if ln.strip()]

            # Ensure exactly three reformulations
            if len(reformulations) < 3:
                while len(reformulations) < 3:
                    reformulations.append(f"Fallback reformulation {len(reformulations)+1}: {question_body}")
            elif len(reformulations) > 3:
                reformulations = reformulations[:3]

            # Guarantee all items are strings
            reformulations = [str(r) for r in reformulations]

            return reformulations
            
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            # Return different fallback options instead of identical copies of the original
            return [
                f"Original question: {question_body}",
                f"Alternative formulation ({timestamp}): The gap word {correct_answer} should be used in {question_body}",
                f"Second alternative ({timestamp}): Try using {correct_answer} to complete {question_body}"
            ]