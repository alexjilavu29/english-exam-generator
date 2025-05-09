#!/bin/bash

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    touch .env
    echo "OPENAI_API_KEY=" >> .env
    echo "OPENAI_MODEL=gpt-4o" >> .env
fi

# Run the application
python app.py
