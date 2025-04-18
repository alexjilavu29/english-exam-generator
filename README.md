# English Exam Generator - User Guide

## Overview
The English Exam Generator is a comprehensive system for managing and generating English exams based on a database of questions. The application provides three main functionalities:

1. **Question Display and Modification**: View and edit existing questions in the database
2. **Question Management**: Add new questions, remove questions, and tag questions with topics or categories
3. **Exam Generation**: Create customized Word document exams with various filtering options

## Installation and Setup

### Prerequisites
- Python 3.6 or higher
- Flask
- python-docx

### Installation Steps
1. Ensure Python is installed on your system
2. Install required packages:
   ```
   pip install flask python-docx
   ```
3. Place the application files in a directory of your choice
4. Make sure the `questions.json` file is in the same directory as the application files

## Running the Application

1. Navigate to the application directory
2. Run the start script:
   ```
   ./start.sh
   ```
   Or run the application directly:
   ```
   python app.py
   ```
3. Open a web browser and go to `http://localhost:5000`

## Using the Application

### Viewing and Modifying Questions
1. Click on "View Questions" in the navigation menu
2. Browse through the questions or use the filters to find specific questions
3. Click on "View Details" to see the full question information
4. On the question detail page, click "Edit" to modify the question
5. Make your changes and click "Save Changes"

### Managing Questions
1. **Adding Questions**:
   - Click on "Add Question" in the navigation menu
   - Fill in the question text, answer options, and select the correct answer
   - Choose the topic, category, and year
   - Add tags (optional) separated by commas
   - Click "Add Question" to save

2. **Removing Questions**:
   - Navigate to the question detail page
   - Click the "Delete" button
   - Confirm the deletion in the popup dialog

3. **Managing Tags**:
   - Click on "Manage Tags" in the navigation menu
   - View all tags and the number of questions associated with each
   - Click "View Questions" to see all questions with a specific tag
   - Use the "Rename" or "Delete" buttons to modify tags

### Generating Exams
1. Click on "Generate Exam" in the navigation menu
2. Configure the exam parameters:
   - Set the percentage of Vocabulary vs. Grammar questions
   - Choose the number of questions
   - Select specific topics, categories, or tags (optional)
   - Set the year range (optional)
3. Configure document formatting:
   - Set the exam title
   - Choose whether to include an answer key
   - Select font, font size, and page orientation
   - Configure other formatting options
4. Click "Generate Exam" to create and download the Word document

## Features

### Question Display
- View all questions in the database
- Filter questions by topic, category, year, or tag
- Detailed view of individual questions

### Question Management
- Add new questions with all required properties
- Edit existing questions
- Delete questions
- Tag questions with custom tags
- Manage tags (view, rename, delete)

### Exam Generation
- Customize the ratio of Vocabulary to Grammar questions
- Filter questions by topic, category, tag, or year range
- Set the number of questions in the exam
- Format the Word document with various options:
  - Custom title
  - Font selection
  - Page orientation
  - Headers and footers
  - Question grouping
  - Answer key inclusion

## Troubleshooting

- If the application doesn't start, ensure all required packages are installed
- If questions don't appear, check that the `questions.json` file is in the correct location
- If exam generation fails, verify that the python-docx library is properly installed

## Support

For additional help or to report issues, please contact the system administrator.
