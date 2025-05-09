from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, session
import json
import os
import sys
import argparse
from docx import Document
import random
from word_formatter import create_word_document
from ai_formatter import AIFormatter
from dotenv import load_dotenv, set_key

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')

# Initialize Flask app with explicit template folder
app = Flask(__name__, template_folder=template_dir)
app.secret_key = os.urandom(24)  # For session management

# Initialize AI formatter with API key from .env
openai_api_key = os.environ.get('OPENAI_API_KEY', '')
openai_model = os.environ.get('OPENAI_MODEL', 'gpt-4o')
ai_formatter = AIFormatter(api_key=openai_api_key, model=openai_model)

# For storing original texts when reformatting
original_texts = {}

# Load questions from JSON file
def load_questions():
    questions_path = os.path.join(current_dir, 'questions.json')
    try:
        with open(questions_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        # Create an empty questions file if it doesn't exist
        with open(questions_path, 'w') as file:
            json.dump([], file)
        return []

# Save questions to JSON file
def save_questions(questions):
    questions_path = os.path.join(current_dir, 'questions.json')
    with open(questions_path, 'w') as file:
        json.dump(questions, file, indent=2)

# Get unique topics, categories, and years from questions
def get_metadata():
    questions = load_questions()
    topics = sorted(list(set(q['topic'] for q in questions)))
    categories = sorted(list(set(q['category'] for q in questions)))
    years = sorted(list(set(q['year'] for q in questions)))
    
    # Get all tags from questions (if they exist)
    tags = set()
    for question in questions:
        if 'tags' in question and question['tags']:
            for tag in question['tags']:
                tags.add(tag)
    
    return {
        'topics': topics,
        'categories': categories,
        'years': years,
        'tags': sorted(list(tags))
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/questions')
def view_questions():
    all_questions = load_questions()
    metadata = get_metadata()
    
    # Filter questions if parameters are provided
    topic_filter = request.args.get('topic')
    category_filter = request.args.get('category')
    year_filter = request.args.get('year')
    tag_filter = request.args.get('tag')
    
    # Start with all questions
    filtered_questions = all_questions.copy()
    
    if topic_filter:
        filtered_questions = [q for q in filtered_questions if q['topic'] == topic_filter]
    if category_filter:
        filtered_questions = [q for q in filtered_questions if q['category'] == category_filter]
    if year_filter and year_filter.isdigit():
        filtered_questions = [q for q in filtered_questions if q['year'] == int(year_filter)]
    if tag_filter:
        filtered_questions = [q for q in filtered_questions if 'tags' in q and tag_filter in q['tags']]
    
    # Create a list of (question, original_index) tuples
    indexed_questions = []
    for i, q in enumerate(all_questions):
        if q in filtered_questions:
            indexed_questions.append((q, i))
    
    return render_template('questions.html', 
                          indexed_questions=indexed_questions,
                          metadata=metadata,
                          topic_filter=topic_filter,
                          category_filter=category_filter,
                          year_filter=year_filter,
                          tag_filter=tag_filter)

@app.route('/question/<int:index>')
def view_question(index):
    questions = load_questions()
    if 0 <= index < len(questions):
        # Check if we came from a filtered view
        filtered_query = request.args.get('filtered', 'false') == 'true'
        filters = {}
        
        # If we came from a filtered view, also pass the filters back so we can return to the filtered list
        if filtered_query:
            filters = {
                'topic': request.args.get('topic', ''),
                'category': request.args.get('category', ''),
                'year': request.args.get('year', ''),
                'tag': request.args.get('tag', '')
            }
        
        return render_template('question_detail.html', 
                              question=questions[index], 
                              index=index,
                              filters=filters,
                              filtered=filtered_query)
    return redirect(url_for('view_questions'))

@app.route('/question/<int:index>/edit', methods=['GET', 'POST'])
def edit_question(index):
    questions = load_questions()
    metadata = get_metadata()
    
    # Get filter parameters
    filtered_query = request.args.get('filtered', 'false') == 'true'
    filter_params = {}
    if filtered_query:
        filter_params = {
            'topic': request.args.get('topic', ''),
            'category': request.args.get('category', ''),
            'year': request.args.get('year', ''),
            'tag': request.args.get('tag', '')
        }
        filter_query_string = '&'.join([f"{k}={v}" for k, v in filter_params.items() if v])
        if filter_query_string:
            filter_query_string = f"filtered=true&{filter_query_string}"
        else:
            filter_query_string = "filtered=true"
    else:
        filter_query_string = ""
    
    if request.method == 'POST':
        if 0 <= index < len(questions):
            questions[index]['body'] = request.form['body']
            questions[index]['answers'] = [
                request.form['answer0'],
                request.form['answer1'],
                request.form['answer2'],
                request.form['answer3']
            ]
            questions[index]['correct_answer'] = int(request.form['correct_answer'])
            questions[index]['topic'] = request.form['topic']
            questions[index]['category'] = request.form['category']
            questions[index]['year'] = int(request.form['year'])
            
            # Handle tags
            tags = request.form.get('tags', '').strip()
            if tags:
                questions[index]['tags'] = [tag.strip() for tag in tags.split(',') if tag.strip()]
            else:
                # Remove tags field if empty
                if 'tags' in questions[index]:
                    del questions[index]['tags']
            
            save_questions(questions)
            
            # Redirect back with filter parameters if needed
            if filtered_query:
                return redirect(f"{url_for('view_question', index=index)}?{filter_query_string}")
            else:
                return redirect(url_for('view_question', index=index))
    
    if 0 <= index < len(questions):
        return render_template('edit_question.html', 
                             question=questions[index], 
                             index=index, 
                             metadata=metadata,
                             filtered=filtered_query,
                             filters=filter_params)
    return redirect(url_for('view_questions'))

@app.route('/question/add', methods=['GET', 'POST'])
def add_question():
    metadata = get_metadata()
    
    if request.method == 'POST':
        new_question = {
            'body': request.form['body'],
            'answers': [
                request.form['answer0'],
                request.form['answer1'],
                request.form['answer2'],
                request.form['answer3']
            ],
            'correct_answer': int(request.form['correct_answer']),
            'topic': request.form['topic'],
            'category': request.form['category'],
            'year': int(request.form['year'])
        }
        
        # Handle tags
        tags = request.form.get('tags', '').strip()
        if tags:
            new_question['tags'] = [tag.strip() for tag in tags.split(',') if tag.strip()]
        
        questions = load_questions()
        questions.append(new_question)
        save_questions(questions)
        
        return redirect(url_for('view_questions'))
    
    return render_template('add_question.html', metadata=metadata)

@app.route('/question/<int:index>/delete', methods=['POST'])
def delete_question(index):
    questions = load_questions()
    
    if 0 <= index < len(questions):
        del questions[index]
        save_questions(questions)
    
    return redirect(url_for('view_questions'))

@app.route('/tags')
def manage_tags():
    questions = load_questions()
    metadata = get_metadata()
    
    # Count questions per tag
    tag_counts = {}
    for tag in metadata['tags']:
        tag_counts[tag] = sum(1 for q in questions if 'tags' in q and tag in q['tags'])
    
    return render_template('tags.html', tags=metadata['tags'], tag_counts=tag_counts)

@app.route('/tag/<tag>')
def view_tag(tag):
    questions = load_questions()
    metadata = get_metadata()
    
    # Filter questions by tag
    filtered_questions = [q for q in questions if 'tags' in q and tag in q['tags']]
    
    return render_template('tag_questions.html', 
                          questions=filtered_questions, 
                          tag=tag,
                          metadata=metadata)

@app.route('/tag/rename', methods=['POST'])
def rename_tag():
    old_tag = request.form.get('old_tag')
    new_tag = request.form.get('new_tag', '').strip()
    
    if old_tag and new_tag:
        questions = load_questions()
        
        # Replace old tag with new tag in all questions
        for question in questions:
            if 'tags' in question and old_tag in question['tags']:
                question['tags'].remove(old_tag)
                if new_tag not in question['tags']:
                    question['tags'].append(new_tag)
        
        save_questions(questions)
    
    return redirect(url_for('manage_tags'))

@app.route('/tag/delete', methods=['POST'])
def delete_tag():
    tag = request.form.get('tag')
    
    if tag:
        questions = load_questions()
        
        # Remove tag from all questions
        for question in questions:
            if 'tags' in question and tag in question['tags']:
                question['tags'].remove(tag)
                # Remove tags field if empty
                if not question['tags']:
                    del question['tags']
        
        save_questions(questions)
    
    return redirect(url_for('manage_tags'))

@app.route('/generate', methods=['GET', 'POST'])
def generate_exam():
    metadata = get_metadata()
    
    if request.method == 'POST':
        # Get filter parameters
        vocab_percent = int(request.form.get('vocab_percent', 50))
        grammar_percent = 100 - vocab_percent
        selected_topics = request.form.getlist('topics')
        selected_categories = request.form.getlist('categories')
        selected_tags = request.form.getlist('tags')
        min_year = int(request.form.get('min_year', 0))
        max_year = int(request.form.get('max_year', 9999))
        num_questions = int(request.form.get('num_questions', 10))
        
        # Get formatting options
        title = request.form.get('exam_title', 'English Exam')
        include_answers = 'include_answers' in request.form
        font_name = request.form.get('font_name', 'Calibri')
        font_size = int(request.form.get('font_size', 11))
        page_orientation = request.form.get('page_orientation', 'portrait')
        include_header_footer = 'include_header_footer' in request.form
        include_question_type = 'include_question_type' in request.form
        
        # Create formatting options dictionary
        format_options = {
            'font_name': font_name,
            'font_size': font_size,
            'title_font_size': font_size + 5,
            'heading_font_size': font_size + 3,
            'page_orientation': page_orientation,
            'question_spacing': 12,
            'include_header_footer': include_header_footer,
            'include_question_type': include_question_type
        }
        
        # Generate exam
        exam_questions = generate_exam_questions(
            vocab_percent, 
            selected_topics, 
            selected_categories,
            selected_tags,
            min_year, 
            max_year, 
            num_questions
        )
        
        # Create Word document
        doc_path = create_word_document(
            exam_questions,
            title=title,
            include_answers=include_answers,
            format_options=format_options
        )
        
        return send_file(doc_path, as_attachment=True, download_name=f"{title.replace(' ', '_')}.docx")
    
    return render_template('generate.html', metadata=metadata)

def generate_exam_questions(vocab_percent, selected_topics, selected_categories, selected_tags, min_year, max_year, num_questions):
    questions = load_questions()
    
    # Filter questions based on criteria
    filtered_questions = [q for q in questions if 
                         (not selected_topics or q['topic'] in selected_topics) and
                         (not selected_categories or q['category'] in selected_categories) and
                         (min_year <= q['year'] <= max_year) and
                         (not selected_tags or ('tags' in q and any(tag in q['tags'] for tag in selected_tags)))]
    
    # Separate vocabulary and grammar questions
    vocab_questions = [q for q in filtered_questions if q['topic'] == 'Vocabulary']
    grammar_questions = [q for q in filtered_questions if q['topic'] == 'Grammar']
    
    # Calculate number of questions for each type
    vocab_count = int(num_questions * vocab_percent / 100)
    grammar_count = num_questions - vocab_count
    
    # Adjust if not enough questions of a type
    if len(vocab_questions) < vocab_count:
        vocab_count = len(vocab_questions)
        grammar_count = min(num_questions - vocab_count, len(grammar_questions))
    
    if len(grammar_questions) < grammar_count:
        grammar_count = len(grammar_questions)
        vocab_count = min(num_questions - grammar_count, len(vocab_questions))
    
    # Randomly select questions
    selected_vocab = random.sample(vocab_questions, vocab_count) if vocab_count > 0 else []
    selected_grammar = random.sample(grammar_questions, grammar_count) if grammar_count > 0 else []
    
    # Combine and shuffle
    exam_questions = selected_vocab + selected_grammar
    random.shuffle(exam_questions)
    
    return exam_questions

def create_word_document(exam_questions, **kwargs):
    """
    This function is kept for backward compatibility.
    It now calls the more advanced word_formatter.create_word_document function.
    """
    from word_formatter import create_word_document as advanced_create_word_document
    return advanced_create_word_document(exam_questions, **kwargs)

# New route for API key settings
@app.route('/ai_settings', methods=['GET', 'POST'])
def ai_settings():
    if request.method == 'POST':
        api_key = request.form.get('api_key', '').strip()
        model = request.form.get('model', 'gpt-4o').strip()
        
        # Save settings to session and .env file
        session['openai_api_key'] = api_key
        session['openai_model'] = model
        
        # Save to .env file
        if api_key:
            set_key(env_path, 'OPENAI_API_KEY', api_key)
        if model:
            set_key(env_path, 'OPENAI_MODEL', model)
        
        # Update AIFormatter
        ai_formatter.set_api_key(api_key)
        ai_formatter.set_model(model)
        
        return redirect(url_for('index'))
    
    # Get current settings
    api_key = os.environ.get('OPENAI_API_KEY', '') or session.get('openai_api_key', '')
    model = os.environ.get('OPENAI_MODEL', 'gpt-4o') or session.get('openai_model', 'gpt-4o')
    
    # Create a preview of the API key
    api_key_preview = "Not set"
    if api_key:
        if len(api_key) > 8:
            api_key_preview = f"{api_key[:4]}...{api_key[-4:]}"
        else:
            api_key_preview = "Set but too short"
    
    return render_template('ai_settings.html', api_key=api_key, api_key_preview=api_key_preview, model=model)

# Route for AI reformatting
@app.route('/question/<int:index>/reformat', methods=['POST'])
def reformat_question(index):
    questions = load_questions()
    
    if 0 <= index < len(questions):
        question = questions[index]
        
        # Get request data
        request_data = request.get_json(silent=True) or {}
        is_regenerate = request_data.get('regenerate', False)
        
        # Store original text for potential reversion (only if first request)
        if index not in original_texts:
            # If question already has an old_text field, use that as the original
            if 'old_text' in question:
                original_texts[index] = question['old_text']
            else:
                # Otherwise use the current body
                original_texts[index] = question['body']
        
        try:
            # Get API key from .env first, then session as fallback
            api_key = os.environ.get('OPENAI_API_KEY', '') or session.get('openai_api_key', '')
            model = os.environ.get('OPENAI_MODEL', 'gpt-4o') or session.get('openai_model', 'gpt-4o')
            
            if not api_key:
                return jsonify({'error': 'OpenAI API key is not set. Please configure it in AI Settings.'}), 400
            
            # Configure AIFormatter
            ai_formatter.set_api_key(api_key)
            ai_formatter.set_model(model)
            
            # Add a timestamp to ensure we're not getting cached results
            timestamp = request.args.get('t', '')
            print(f"Processing reformatting request at {timestamp}")
            
            if is_regenerate:
                print("This is a regeneration request - forcing new reformulations")
            
            # Get reformulations
            reformulations = ai_formatter.reformat_question(question)
            
            # Ensure we're not returning the same text three times
            unique_reformulations = set(reformulations)
            if len(unique_reformulations) < len(reformulations):
                print("Warning: Received duplicate reformulations from AI")
            
            return jsonify({'reformulations': reformulations})
        
        except Exception as e:
            print(f"Error in reformatting: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Question not found'}), 404

# Route for applying a reformatted question
@app.route('/question/<int:index>/apply_reformat', methods=['POST'])
def apply_reformat(index):
    questions = load_questions()
    
    if 0 <= index < len(questions):
        question = questions[index]
        new_text = request.json.get('text', '')
        
        if new_text:
            # Store original text in the question object if this is the first reformatting
            if 'old_text' not in question:
                # If we already have the original in memory, use that
                if index in original_texts:
                    question['old_text'] = original_texts[index]
                # Otherwise use the current body text
                else:
                    question['old_text'] = question['body']
            
            # Also store it in memory for the current session
            original_texts[index] = question.get('old_text', question['body'])
            
            # Update question body
            question['body'] = new_text
            
            # Add or update tags
            if 'tags' not in question:
                question['tags'] = []
            
            if 'Reformatted with AI' not in question['tags']:
                question['tags'].append('Reformatted with AI')
            
            save_questions(questions)
            
            return jsonify({'success': True})
    
    return jsonify({'error': 'Question not found or invalid text'}), 404

# Route for reverting back to original text
@app.route('/question/<int:index>/revert', methods=['POST'])
def revert_question(index):
    questions = load_questions()
    
    if 0 <= index < len(questions):
        question = questions[index]
        
        # Check if we have the original text stored in the question
        if 'old_text' in question:
            # Restore original text
            question['body'] = question['old_text']
            
            # Remove the old_text field
            del question['old_text']
            
            # Remove reformatting tag if it exists
            if 'tags' in question and 'Reformatted with AI' in question['tags']:
                question['tags'].remove('Reformatted with AI')
                # Remove tags field if empty
                if not question['tags']:
                    del question['tags']
            
            save_questions(questions)
            
            # Remove from original_texts dictionary if present
            if index in original_texts:
                del original_texts[index]
            
            return jsonify({'success': True})
        # Fallback to the in-memory storage if old_text is not in the question
        elif index in original_texts:
            # Restore original text
            question['body'] = original_texts[index]
            
            # Remove reformatting tag if it exists
            if 'tags' in question and 'Reformatted with AI' in question['tags']:
                question['tags'].remove('Reformatted with AI')
                # Remove tags field if empty
                if not question['tags']:
                    del question['tags']
            
            save_questions(questions)
            
            # Remove from original_texts dictionary
            del original_texts[index]
            
            return jsonify({'success': True})
    
    return jsonify({'error': 'Question not found or no original text stored'}), 404

# Route for downloading questions.json file
@app.route('/download_questions')
def download_questions():
    questions_path = os.path.join(current_dir, 'questions.json')
    return send_file(questions_path, as_attachment=True, download_name="questions.json")

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='English Exam Generator')
    parser.add_argument('--port', type=int, default=5001,
                        help='Port number to run the server on (default: 5001)')
    args = parser.parse_args()
    
    # Create templates directory if it doesn't exist
    os.makedirs(template_dir, exist_ok=True)
    
    # Create output directory for Word documents
    output_dir = os.path.join(current_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting application with template directory: {template_dir}")
    print(f"Server running on port: {args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=True)
