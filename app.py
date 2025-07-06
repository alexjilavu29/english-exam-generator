import os
import sys
import json
import random
import argparse
import re
from unicodedata import normalize

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, session
from docx import Document
from word_formatter import create_word_document
from ai_formatter import AIFormatter
from dotenv import load_dotenv, set_key

def slugify(text):
    if text is None:
        return ""
    text = str(text) # Ensure text is a string
    # Replace / with a hyphen
    text = text.replace('/', '-')
    # Normalize unicode characters
    text = normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    # Remove characters that are not alphanumeric, spaces, or hyphens
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    # Replace whitespace and repeated hyphens with a single hyphen
    text = re.sub(r'[-\s]+', '-', text)
    return text

def get_chapter_by_slug(slug):
    tag_data = load_tag_data()
    for chapter_id, chapter in tag_data['chapters'].items():
        if slugify(chapter['name']) == slug:
            return chapter
    return None

def get_chapter_id_from_slug(slug):
    chapter = get_chapter_by_slug(slug)
    if chapter:
        return chapter.get('id')
    return None

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')

# Initialize Flask app with explicit template folder
app = Flask(__name__, template_folder=template_dir)
app.secret_key = os.urandom(24)  # For session management

@app.context_processor
def utility_processor():
    return dict(slugify=slugify)

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

# Load tags and chapters from JSON file
def load_tag_data():
    tags_path = os.path.join(current_dir, 'tags_and_chapters.json')
    try:
        with open(tags_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        # Create an empty tags file if it doesn't exist
        default_data = {"tags": {}, "chapters": {}}
        with open(tags_path, 'w') as file:
            json.dump(default_data, file, indent=2)
        return default_data

# Save tags and chapters to JSON file
def save_tag_data(tag_data):
    tags_path = os.path.join(current_dir, 'tags_and_chapters.json')
    with open(tags_path, 'w') as file:
        json.dump(tag_data, file, indent=2)

# Synchronize tags between questions.json and tags_and_chapters.json
def sync_tags_from_questions():
    """
    Ensures that all tags found in questions.json exist in tags_and_chapters.json.
    This is called whenever tags might be out of sync (e.g., after uploading a new questions file).
    """
    questions = load_questions()
    tag_data = load_tag_data()
    
    # Extract all unique tags from questions
    question_tags = set()
    for question in questions:
        if 'tags' in question and isinstance(question['tags'], list):
            for tag in question['tags']:
                question_tags.add(tag.strip())
    
    # Add any missing tags to tag_data
    added_count = 0
    for tag in question_tags:
        if tag not in tag_data['tags']:
            tag_data['tags'][tag] = {
                "name": tag,
                "description": ""
            }
            added_count += 1
    
    if added_count > 0:
        save_tag_data(tag_data)
    
    return added_count

# Get unique topics, categories, and years from questions
def get_metadata():
    questions = load_questions()
    topics = sorted(list(set(q['topic'] for q in questions)))
    categories = sorted(list(set(q['category'] for q in questions)))
    years = sorted(list(set(q['year'] for q in questions)))
    
    # Get all tags from the centralized tag data
    tag_data = load_tag_data()
    tags = sorted(list(tag_data['tags'].keys()))
    
    # Get all chapters
    chapters = list(tag_data['chapters'].values())
    
    return {
        'topics': topics,
        'categories': categories,
        'years': years,
        'tags': tags,
        'chapters': chapters
    }

@app.route('/')
def index():
    metadata = get_metadata()
    questions = load_questions()
    metadata['questions_count'] = len(questions)
    return render_template('index.html', metadata=metadata)

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
                tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
                questions[index]['tags'] = tag_list
                
                # Sync any new tags to the centralized tag data
                tag_data = load_tag_data()
                added_new_tags = False
                for tag in tag_list:
                    if tag not in tag_data['tags']:
                        tag_data['tags'][tag] = {
                            "name": tag,
                            "description": ""
                        }
                        added_new_tags = True
                
                if added_new_tags:
                    save_tag_data(tag_data)
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
            tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            new_question['tags'] = tag_list
            
            # Sync any new tags to the centralized tag data
            tag_data = load_tag_data()
            added_new_tags = False
            for tag in tag_list:
                if tag not in tag_data['tags']:
                    tag_data['tags'][tag] = {
                        "name": tag,
                        "description": ""
                    }
                    added_new_tags = True
            
            if added_new_tags:
                save_tag_data(tag_data)
        
        questions = load_questions()
        questions.append(new_question)
        save_questions(questions)
        
        return redirect(url_for('view_questions'))
    
    return render_template('add_question.html', metadata=metadata)

@app.route('/question/<int:index>/delete', methods=['POST'])
def delete_question(index):
    questions = load_questions()
    
    # Get filter parameters for redirect
    filtered_query = request.args.get('filtered', 'false') == 'true'
    if filtered_query:
        filter_params = {
            'topic': request.args.get('topic', ''),
            'category': request.args.get('category', ''),
            'year': request.args.get('year', ''),
            'tag': request.args.get('tag', '')
        }
        filter_query_string = '&'.join([f"{k}={v}" for k, v in filter_params.items() if v])
        if filter_query_string:
            filter_query_string = f"?{filter_query_string}"
        else:
            filter_query_string = ""
    else:
        filter_query_string = ""
    
    if 0 <= index < len(questions):
        del questions[index]
        save_questions(questions)
    
    return redirect(f"{url_for('view_questions')}{filter_query_string}")

@app.route('/tags')
def manage_tags():
    questions = load_questions()
    tag_data = load_tag_data()
    
    # Count questions per tag
    tag_counts = {}
    for tag_name in tag_data['tags']:
        tag_counts[tag_name] = sum(1 for q in questions if 'tags' in q and tag_name in q['tags'])
    
    # Sort tags alphabetically for consistent display
    all_tags_info = []
    for tag_name, tag_details in sorted(tag_data['tags'].items()):
        all_tags_info.append({
            'name': tag_name,
            'description': tag_details.get('description', ''),
            'count': tag_counts.get(tag_name, 0)
        })

    used_tags = [tag for tag in all_tags_info if tag['count'] > 0]
    unused_tags = [tag for tag in all_tags_info if tag['count'] == 0]

    # Get feedback messages from session
    tag_message = session.pop('tag_message', None)
    tag_error = session.pop('tag_error', None)
    
    return render_template('tags.html', 
                         used_tags=used_tags, 
                         unused_tags=unused_tags,
                         tag_message=tag_message,
                         tag_error=tag_error)

@app.route('/tag/<path:tag>')
def view_tag(tag):
    questions = load_questions()
    metadata = get_metadata()
    
    # Filter questions by tag
    filtered_questions = [q for q in questions if 'tags' in q and tag in q['tags']]
    
    return render_template('tag_questions.html', 
                          questions=filtered_questions, 
                          tag=tag,
                          metadata=metadata)

@app.route('/tag/update', methods=['POST'])
def update_tag():
    old_name = request.form.get('old_name')
    new_name = request.form.get('new_name', '').strip()
    description = request.form.get('description', '').strip()

    if not old_name or not new_name:
        session['tag_error'] = "Tag name cannot be empty."
        return redirect(url_for('manage_tags'))

    tag_data = load_tag_data()

    if new_name != old_name and new_name in tag_data['tags']:
        session['tag_error'] = f"Tag '{new_name}' already exists."
        return redirect(url_for('manage_tags'))

    if old_name in tag_data['tags']:
        if old_name != new_name:
            tag_data['tags'][new_name] = tag_data['tags'].pop(old_name)
            tag_data['tags'][new_name]['name'] = new_name
            
            questions = load_questions()
            for q in questions:
                if 'tags' in q and old_name in q['tags']:
                    q['tags'] = [new_name if t == old_name else t for t in q['tags']]
            save_questions(questions)

            for chapter in tag_data['chapters'].values():
                if 'tags' in chapter and old_name in chapter['tags']:
                    chapter['tags'] = [new_name if t == old_name else t for t in chapter['tags']]
        
        tag_data['tags'][new_name]['description'] = description
        save_tag_data(tag_data)
        session['tag_message'] = f"Tag '{old_name}' updated successfully to '{new_name}'."
    else:
        session['tag_error'] = f"Tag '{old_name}' not found."

    return redirect(url_for('manage_tags'))

@app.route('/tag/delete', methods=['POST'])
def delete_tag():
    tag = request.form.get('tag')
    
    if tag:
        # Remove tag from centralized tag data
        tag_data = load_tag_data()
        if tag in tag_data['tags']:
            del tag_data['tags'][tag]
            
            # Remove tag from all chapters
            for chapter in tag_data['chapters'].values():
                if tag in chapter['tags']:
                    chapter['tags'].remove(tag)
            
            save_tag_data(tag_data)
        
        # Remove tag from all questions
        questions = load_questions()
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
        selected_chapters = request.form.getlist('chapters')
        min_year = int(request.form.get('min_year', 0))
        max_year = int(request.form.get('max_year', 9999))
        num_questions = int(request.form.get('num_questions'))
        
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
            selected_chapters,
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

def generate_exam_questions(vocab_percent, selected_topics, selected_categories, selected_tags, selected_chapters, min_year, max_year, num_questions):
    questions = load_questions()
    
    # Collect all tags from selected chapters
    chapter_tags = set()
    if selected_chapters:
        tag_data = load_tag_data()
        for chapter_id in selected_chapters:
            if chapter_id in tag_data['chapters']:
                chapter_tags.update(tag_data['chapters'][chapter_id]['tags'])
    
    # Combine selected tags with chapter tags
    all_selected_tags = set(selected_tags) | chapter_tags
    
    # Filter questions based on criteria
    filtered_questions = [q for q in questions if 
                         (not selected_topics or q['topic'] in selected_topics) and
                         (not selected_categories or q['category'] in selected_categories) and
                         (min_year <= q['year'] <= max_year) and
                         (not all_selected_tags or ('tags' in q and any(tag in q['tags'] for tag in all_selected_tags)))]
    
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

# Settings route for API key settings and file uploads
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        # Handle AI Settings
        if 'api_key' in request.form:
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
            
            return redirect(url_for('settings'))
        
        # Handle file uploads
        if 'questions_file' in request.files:
            file = request.files['questions_file']
            if file.filename and file.filename.endswith('.json'):
                try:
                    # Load and validate the uploaded questions file
                    uploaded_data = json.load(file)
                    if isinstance(uploaded_data, list):
                        # Save the new questions first to ensure sync works on the new data
                        save_questions(uploaded_data)
                        
                        # Synchronize tags from the newly uploaded file
                        added_tags = sync_tags_from_questions()
                        
                        session['upload_message'] = f"Questions file uploaded successfully. {added_tags} new tags synchronized."
                    else:
                        session['upload_error'] = "Invalid questions file format. Expected a JSON array."
                except json.JSONDecodeError:
                    session['upload_error'] = "Invalid JSON file."
                except Exception as e:
                    session['upload_error'] = f"Error uploading file: {str(e)}"
            else:
                session['upload_error'] = "Please select a valid JSON file."
            
            return redirect(url_for('settings'))
        
        if 'tags_file' in request.files:
            file = request.files['tags_file']
            if file.filename and file.filename.endswith('.json'):
                try:
                    # Load and validate the uploaded tags file
                    uploaded_data = json.load(file)
                    if isinstance(uploaded_data, dict) and 'tags' in uploaded_data and 'chapters' in uploaded_data:
                        # Save the new tag data
                        save_tag_data(uploaded_data)
                        session['upload_message'] = "Tags and chapters file uploaded successfully."
                    else:
                        session['upload_error'] = "Invalid tags file format. Expected JSON with 'tags' and 'chapters' keys."
                except json.JSONDecodeError:
                    session['upload_error'] = "Invalid JSON file."
                except Exception as e:
                    session['upload_error'] = f"Error uploading file: {str(e)}"
            else:
                session['upload_error'] = "Please select a valid JSON file."
            
            return redirect(url_for('settings'))
    
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
    
    # Get upload messages
    upload_message = session.pop('upload_message', None)
    upload_error = session.pop('upload_error', None)
    
    return render_template('settings.html', 
                         api_key=api_key, 
                         api_key_preview=api_key_preview, 
                         model=model,
                         upload_message=upload_message,
                         upload_error=upload_error)

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

# Route for downloading tags_and_chapters.json file
@app.route('/download_tags')
def download_tags():
    tags_path = os.path.join(current_dir, 'tags_and_chapters.json')
    return send_file(tags_path, as_attachment=True, download_name="tags_and_chapters.json")

# Tag Centralizer page
@app.route('/tag-centralizer')
def tag_centralizer():
    tag_data = load_tag_data()
    questions = load_questions()
    
    # Count questions per tag
    tag_counts = {}
    for tag_name in tag_data['tags']:
        tag_counts[tag_name] = sum(1 for q in questions if 'tags' in q and tag_name in q['tags'])
    
    # Find orphaned tags (tags in questions but not in centralized data)
    question_tags = set()
    for question in questions:
        if 'tags' in question and isinstance(question['tags'], list):
            for tag in question['tags']:
                question_tags.add(tag.strip())
    
    orphaned_tags = question_tags - set(tag_data['tags'].keys())
    
    return render_template('tag_centralizer.html', 
                         tag_data=tag_data,
                         tag_counts=tag_counts,
                         orphaned_tags=list(orphaned_tags))

# Chapter management routes
@app.route('/chapter/create', methods=['POST'])
def create_chapter():
    name = request.form.get('name', '').strip()
    description = request.form.get('description', '').strip()
    
    if name:
        tag_data = load_tag_data()
        chapter_id = name.lower().replace(' ', '-').replace('&', 'and')
        
        # Ensure unique ID
        counter = 1
        original_id = chapter_id
        while chapter_id in tag_data['chapters']:
            chapter_id = f"{original_id}-{counter}"
            counter += 1
        
        tag_data['chapters'][chapter_id] = {
            "id": chapter_id,
            "name": name,
            "description": description,
            "tags": []
        }
        
        save_tag_data(tag_data)
    
    return redirect(url_for('tag_centralizer'))

@app.route('/chapter/<slug>/edit', methods=['POST'])
def edit_chapter(slug):
    tag_data = load_tag_data()
    
    # Find the chapter_id based on slug
    chapter_id = None
    for c_id, c in tag_data['chapters'].items():
        if slugify(c['name']) == slug:
            chapter_id = c_id
            break
            
    if not chapter_id:
        # Handle case where chapter is not found
        return redirect(url_for('tag_centralizer'))

    # Get form data
    new_name = request.form.get('name', '').strip()
    new_description = request.form.get('description', '').strip()
    new_tags = request.form.getlist('tags')

    if new_name:
        # Update chapter details
        tag_data['chapters'][chapter_id]['name'] = new_name
        tag_data['chapters'][chapter_id]['description'] = new_description
        tag_data['chapters'][chapter_id]['tags'] = new_tags
        save_tag_data(tag_data)
            
    return redirect(url_for('tag_centralizer'))

@app.route('/chapter/<slug>/delete', methods=['POST'])
def delete_chapter(slug):
    chapter_id = get_chapter_id_from_slug(slug)
    if not chapter_id:
        return redirect(url_for('tag_centralizer'))

    tag_data = load_tag_data()
    if chapter_id in tag_data['chapters']:
        # Get the name of the chapter before deleting it
        chapter_name_to_delete = tag_data['chapters'][chapter_id].get('name')

        del tag_data['chapters'][chapter_id]
        save_tag_data(tag_data)
        
        # Also remove chapter from any question that has it
        if chapter_name_to_delete:
            questions = load_questions()
            for q in questions:
                if 'chapter' in q and q['chapter'] == chapter_name_to_delete:
                    del q['chapter']
            save_questions(questions)
        
    return redirect(url_for('tag_centralizer'))

# Advanced tag management routes
@app.route('/tag/merge', methods=['POST'])
def merge_tags():
    source_tags = request.form.getlist('source_tags')
    destination_tag = request.form.get('destination_tag', '').strip()
    
    if source_tags and destination_tag:
        try:
            tag_data = load_tag_data()
            questions = load_questions()
            
            # Convert source_tags to set for faster lookup
            source_tags_set = set(source_tags)
            
            # Ensure destination tag exists in centralized data
            if destination_tag not in tag_data['tags']:
                tag_data['tags'][destination_tag] = {
                    "name": destination_tag,
                    "description": ""
                }
            
            # Track progress for large operations
            questions_modified = 0
            total_questions = len(questions)
            
            # Optimized question processing
            for i, question in enumerate(questions):
                if 'tags' in question and question['tags']:
                    # Check if this question has any source tags
                    question_tags_set = set(question['tags'])
                    has_source_tags = bool(source_tags_set & question_tags_set)
                    
                    if has_source_tags:
                        # Remove source tags
                        question['tags'] = [tag for tag in question['tags'] if tag not in source_tags_set]
                        
                        # Add destination tag if not already present
                        if destination_tag not in question['tags']:
                            question['tags'].append(destination_tag)
                        
                        # Clean up empty tags list
                        if not question['tags']:
                            del question['tags']
                        
                        questions_modified += 1
                
                # Progress feedback for large datasets (every 1000 questions)
                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1}/{total_questions} questions, modified {questions_modified}")
            
            # Update centralized tag data and chapters
            for source_tag in source_tags:
                if source_tag in tag_data['tags'] and source_tag != destination_tag:
                    del tag_data['tags'][source_tag]
                
                # Update chapters efficiently
                for chapter in tag_data['chapters'].values():
                    if source_tag in chapter['tags']:
                        chapter['tags'].remove(source_tag)
                        if destination_tag not in chapter['tags']:
                            chapter['tags'].append(destination_tag)
            
            # Save data efficiently
            print(f"Merge complete: Modified {questions_modified} questions")
            save_tag_data(tag_data)
            save_questions(questions)
            
            # Add success message to session
            session['merge_success'] = f"Successfully merged {len(source_tags)} tag(s) into '{destination_tag}'. Modified {questions_modified} questions."
            
        except Exception as e:
            print(f"Error during tag merge: {str(e)}")
            session['merge_error'] = f"Error during tag merge: {str(e)}"
    
    return redirect(url_for('tag_centralizer'))

@app.route('/tag/create', methods=['POST'])
def create_tag():
    tag_name = request.form.get('tag_name', '').strip()
    tag_description = request.form.get('tag_description', '').strip()

    if tag_name:
        tag_data = load_tag_data()
        if tag_name not in tag_data['tags']:
            tag_data['tags'][tag_name] = {
                "name": tag_name,
                "description": tag_description
            }
            save_tag_data(tag_data)
            session['tag_message'] = f"Tag '{tag_name}' created successfully."
        else:
            session['tag_error'] = f"Tag '{tag_name}' already exists."

    return redirect(url_for('tag_centralizer'))

@app.route('/tag/add-orphaned', methods=['POST'])
def add_orphaned_tags():
    orphaned_tags = request.form.getlist('orphaned_tags')
    
    if orphaned_tags:
        tag_data = load_tag_data()
        
        for tag in orphaned_tags:
            if tag not in tag_data['tags']:
                tag_data['tags'][tag] = {
                    "name": tag,
                    "description": ""
                }
        
        save_tag_data(tag_data)
    
    return redirect(url_for('tag_centralizer'))

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
