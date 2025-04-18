# English Exam Generator - Project Structure

This document outlines the complete file structure for the English Exam Generator application to help you recreate it correctly.

## Directory Structure

```
english_exam_generator/
├── app.py                 # Main Flask application
├── questions.json         # Database of exam questions
├── word_formatter.py      # Module for Word document formatting
├── start.sh               # Startup script
├── README.md              # User documentation
├── output/                # Directory for generated Word documents
└── templates/             # HTML templates
    ├── index.html             # Home page
    ├── questions.html         # Question listing page
    ├── question_detail.html   # Individual question view
    ├── edit_question.html     # Question editing form
    ├── add_question.html      # New question form
    ├── generate.html          # Exam generation form
    ├── tags.html              # Tag management page
    └── tag_questions.html     # Questions by tag page
```

## Key Files Description

### 1. app.py
The main Flask application that handles all routes and logic. It includes:
- Question loading and saving functions
- Routes for viewing, adding, editing, and deleting questions
- Tag management functionality
- Exam generation with customizable options

### 2. questions.json
The database file containing all exam questions. Each question has:
- Question text (body)
- Answer options
- Correct answer index
- Topic (Vocabulary/Grammar)
- Category (FCE, etc.)
- Year
- Optional tags

### 3. word_formatter.py
Module for creating and formatting Word documents with:
- Custom formatting options (fonts, sizes, etc.)
- Headers and footers
- Answer key generation
- Question grouping by type

### 4. templates/
Directory containing all HTML templates for the web interface.

## How to Recreate the Project

1. Create the directory structure as shown above
2. Copy all files to their respective locations
3. Ensure the templates directory contains all HTML files
4. Make start.sh executable with `chmod +x start.sh`
5. Run the application with `./start.sh` or `python app.py`

## Common Issues and Solutions

### TemplateNotFound Errors
These errors occur when Flask cannot locate the template files. The fixes implemented include:
1. Using absolute paths for the template directory
2. Explicitly setting the template folder in Flask initialization
3. Creating the templates directory if it doesn't exist
4. Using absolute paths for all file operations

### File Not Found Errors
These can occur when the application cannot find the questions.json file or when saving Word documents. The fixes include:
1. Using absolute paths for all file operations
2. Creating necessary directories (output) if they don't exist
3. Ensuring consistent working directory regardless of where the app is started from

## Dependencies

- Python 3.6 or higher
- Flask
- python-docx

Install dependencies with:
```
pip install flask python-docx
```
