#!/bin/bash

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Create Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    touch .env
    echo "OPENAI_API_KEY=" >> .env
    echo "OPENAI_MODEL=gpt-4o" >> .env
fi

# Run the application
python app.py

# Keep terminal window open if there are errors
read -p "Press enter to close this window..." 