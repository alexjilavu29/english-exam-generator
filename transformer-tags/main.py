import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np

from data import questions_with_tags

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
