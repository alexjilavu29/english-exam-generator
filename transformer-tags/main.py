import json


def load_and_process_questions(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed = []
    for item in data:
        body = item.get("body", "")
        answers = item.get("answers", [])
        tags = item.get("tags", [])
        correct_answer = item.get("correct_answer", 0)

        processed.append({
            "body": body,
            "answers": answers,
            "tags": tags,
            "correct_answer": correct_answer
        })

    return processed


processed_questions = load_and_process_questions("questions.json")

print(json.dumps(processed_questions, indent=2))

questions_with_tags = [q for q in processed_questions if q["tags"]]
questions_without_tags = [q for q in processed_questions if not q["tags"]]

print("Questions with tags:\n")
print(json.dumps(questions_with_tags, indent=2))

print("Questions without tags:\n")
print(json.dumps(questions_without_tags, indent=2))

print("Questions with tags length:")
print(len(questions_with_tags))

print("Questions without tags length:")
print(len(questions_without_tags))
