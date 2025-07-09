
import json
import re


def process_body(body):
    new_body = body.replace("\u2026", ".")
    new_body = new_body.replace("\u2019", "'")
    new_body = new_body.replace("\u2018", "'")
    new_body = new_body.replace("\u2013", "-")
    new_body = new_body.replace('\u00A0', " ")
    new_body = new_body.replace("\u201c", "'")
    new_body = new_body.replace("\u201d", "'")
    new_body = new_body.replace("\u00a3", "£")
    return new_body


def load_and_process_questions(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed = []
    for item in data:
        body = item.get("body", "")
        answers = item.get("answers", [])
        tags = item.get("tags", [])
        correct_answer = item.get("correct_answer", 0)

        correct_text = answers[correct_answer] if 0 <= correct_answer < len(
            answers) else ""

        body_processed = process_body(body)
        body_processed = re.sub(r"\.{2,}", correct_text, body_processed)

        tags = [tag for tag in tags if tag != "Reformatted with AI"]

        processed.append({
            "body": body_processed,
            "answers": answers,
            "tags": tags,
            "correct_answer": correct_answer
        })

    return processed


processed_questions = load_and_process_questions("questions.json")

print("Questions length:")
print(len(processed_questions))

# print(json.dumps(processed_questions, indent=2))

questions_with_tags = [q for q in processed_questions if q["tags"]]
questions_without_tags = [
    q for q in processed_questions if not q["tags"]]

# print("Questions with tags:\n")
# print(json.dumps(questions_with_tags, indent=2))

# print("Questions without tags:\n")
# print(json.dumps(questions_without_tags, indent=2))

print("Questions with tags length:")
print(len(questions_with_tags))

print("Questions without tags length:")
print(len(questions_without_tags))


def contains_non_ascii(text):
    return any(ord(char) > 127 and char != '£' for char in text)


questions_with_funny_chars = [
    q for q in processed_questions if contains_non_ascii(q["body"])
]

print("Questions with funny chars length:")
print(len(questions_with_funny_chars))

print(json.dumps(questions_with_funny_chars, indent=2, ensure_ascii=False))
