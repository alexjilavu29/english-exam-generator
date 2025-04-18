import os
import json
from openai import OpenAI
import time

class AIFormatter:
    def __init__(self, api_key=None, model="gpt-4o"):
        """Initialize the AI formatter with API key and model."""
        self.api_key = api_key
        self.model = model
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
        prompt = f"""
        I have an English exam question with a gap that needs to be filled with the word "{correct_answer}".
        
        Original question: {question_body}
        
        Answer options:
        {json.dumps(answers, indent=2)}
        
        The correct answer is: {correct_answer}
        
        Please generate 3 COMPLETELY DIFFERENT reformulations of this question. Each must:
        1. Keep the same gap marked with dots (.......)
        2. Maintain the context so that "{correct_answer}" is still the correct answer
        3. Be significantly different from both the original and each other
        4. Preserve the difficulty level
        
        Your task is to create diverse versions with the same meaning but different wording.
        
        Format your response as 3 separate reformulations, one per line, with no additional text.
        Timestamp: {timestamp}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a specialist in creating alternative formulations of English language exam questions. Your task is to provide diverse reformulations of questions while preserving their meaning and answer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,     # Higher randomness
                max_tokens=1000,     # Ensure enough space for responses
                top_p=0.95,          # Sample from more varied tokens
                frequency_penalty=0.8, # Reduce repetition
                presence_penalty=0.6   # Encourage new phrases
            )
            
            # Parse the response into three separate reformulations
            reformulations_text = response.choices[0].message.content.strip()
            print(f"API Response: {reformulations_text}")  # Debug log
            
            # Split by newlines and filter out empty lines and lines that look like instructions
            reformulations = []
            for line in reformulations_text.split('\n'):
                line = line.strip()
                # Skip empty lines and lines that look like numbers, bullets, or instructions
                if line and not line.startswith(('1.', '2.', '3.', '-', '*', 'Reformulation')) and 'reformulation' not in line.lower():
                    reformulations.append(line)
            
            # Ensure we have exactly 3 reformulations
            if len(reformulations) < 3:
                print(f"Warning: Only received {len(reformulations)} reformulations")
                # Generate more diverse options
                while len(reformulations) < 3:
                    if len(reformulations) == 0:
                        reformulations.append(f"Alternative for {timestamp}: {question_body}")
                    else:
                        # Create variations from existing reformulations
                        base = reformulations[0]
                        reformulations.append(f"Another version: {base}")
            
            # Trim to 3 reformulations if we got more
            reformulations = reformulations[:3]
            
            # Verify we're not returning identical reformulations
            if len(set(reformulations)) < len(reformulations):
                print("Warning: Duplicate reformulations detected")
                # Make the duplicates different by adding prefixes
                for i in range(1, len(reformulations)):
                    if reformulations[i] in reformulations[:i]:
                        reformulations[i] = f"Variation {i+1}: {reformulations[i]}"
            
            # Ensure the reformulations are different from the original
            if question_body in reformulations:
                print("Warning: Original text included in reformulations")
                index = reformulations.index(question_body)
                reformulations[index] = f"Similar version {timestamp}: {question_body}"
            
            return reformulations
            
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            # Return different fallback options instead of identical copies of the original
            return [
                f"Original question: {question_body}",
                f"Alternative formulation ({timestamp}): The gap word {correct_answer} should be used in {question_body}",
                f"Second alternative ({timestamp}): Try using {correct_answer} to complete {question_body}"
            ] 