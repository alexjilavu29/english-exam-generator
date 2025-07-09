import pickle
import json
import re
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np


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

texts = [item["body"] for item in questions_with_tags]
tag_lists = [item["tags"] for item in questions_with_tags]

all_tags = sorted(list({tag for tags in tag_lists for tag in tags}))

mlb = MultiLabelBinarizer()
tag_matrix = mlb.fit_transform(tag_lists)

dataset = Dataset.from_dict({
    "text": texts,
    "labels": tag_matrix.astype(np.float32)
})

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)


dataset = dataset.map(tokenize, batched=True)


# def convert_labels_to_float(item):
#     item["labels"] = torch.tensor(item["labels"], dtype=torch.float)
#     return item


# dataset = dataset.map(convert_labels_to_float)

dataset.set_format(type="torch", columns=[
                        "input_ids", "attention_mask", "labels"])

# 20% sa testam
split = dataset.train_test_split(test_size=0.2)
train_dataset = split["train"]
eval_dataset = split["test"]

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(mlb.classes_),
    problem_type="multi_label_classification"
)


def compute_metrics(pred):
    logits, labels = pred
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.tensor(logits))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= 0.5)] = 1  # threshold de 0.5

    from sklearn.metrics import f1_score, precision_score, recall_score

    return {
        "f1_micro": f1_score(labels, y_pred, average="micro"),
        "precision_micro": precision_score(labels, y_pred, average="micro"),
        "recall_micro": recall_score(labels, y_pred, average="micro"),
    }


training_args = TrainingArguments(
    output_dir="./bert",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("./bert")
tokenizer.save_pretrained("./bert")

with open("./bert/tag_classes.pkl", "wb") as f:
    pickle.dump(mlb.classes_, f)
