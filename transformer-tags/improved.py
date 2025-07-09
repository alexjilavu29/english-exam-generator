from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers import DataCollatorWithPadding
from datasets import Dataset
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import pickle
from collections import Counter
import re
from data import questions_with_tags

texts = [item["body"] for item in questions_with_tags]
tag_lists = [item["tags"] for item in questions_with_tags]

all_tags = [tag for tags in tag_lists for tag in tags]
tag_counter = Counter(all_tags)

MIN_TAG_COUNT = 5
filtered_tags = [tag for tag, count in tag_counter.items()
                 if count >= MIN_TAG_COUNT]
filtered_tag_lists = [
    [tag for tag in tags if tag in filtered_tags]
    for tags in tag_lists
]

texts, filtered_tag_lists = zip(*[
    (text, tags) for text, tags in zip(texts, filtered_tag_lists)
    if len(tags) > 0
])

mlb = MultiLabelBinarizer()
tag_matrix = mlb.fit_transform(filtered_tag_lists)

label_counts = np.sum(tag_matrix, axis=0)
class_weights = torch.tensor(
    (len(tag_matrix) - label_counts) / (label_counts + 1e-6),
    dtype=torch.float32
)

model_name = "roberta-large"
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=256,
        padding=False
    )


class FocalLossTrainer(Trainer):
    def __init__(self, class_weights, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.device)
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, labels)
        probs = torch.sigmoid(logits)
        pt = torch.where(labels == 1, probs, 1 - probs)
        focal_weight = (1 - pt).pow(self.gamma)

        weighted_loss = focal_weight * bce_loss * torch.where(
            labels == 1,
            self.class_weights,
            torch.tensor(1.0).to(self.model.device)
        )

        loss = weighted_loss.mean()
        return (loss, outputs) if return_outputs else loss


dataset = Dataset.from_dict({
    "text": texts,
    "labels": tag_matrix.astype(np.float32)
})
dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = RobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(mlb.classes_),
    problem_type="multi_label_classification",
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2
)


def optimize_threshold(logits, labels):
    best_thresholds = []
    for i in range(logits.shape[1]):
        thresholds = np.linspace(0.1, 0.9, 50)
        f1_scores = []
        for thresh in thresholds:
            preds = (logits[:, i] >= thresh).astype(int)
            report = classification_report(
                labels[:, i], preds,
                output_dict=True, zero_division=0
            )
            f1_scores.append(report['macro avg']['f1-score'])
        best_thresholds.append(thresholds[np.argmax(f1_scores)])
    return best_thresholds


training_args = TrainingArguments(
    output_dir="./roberta_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    num_train_epochs=8,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    greater_is_better=True,
    fp16=True,
    dataloader_num_workers=4,
    logging_steps=100,
    report_to="none",
    save_total_limit=2,
)


def compute_metrics(pred):
    logits, labels = pred
    sigmoid = nn.Sigmoid()
    probs = sigmoid(torch.tensor(logits))

    y_pred = np.zeros(probs.shape)
    y_pred[probs >= 0.5] = 1

    report = classification_report(
        labels, y_pred,
        output_dict=True,
        zero_division=0,
        target_names=mlb.classes_
    )

    return {
        "f1_micro": report["micro avg"]["f1-score"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_micro": report["micro avg"]["precision"],
        "recall_micro": report["micro avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"]
    }


trainer = FocalLossTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    class_weights=class_weights,
    gamma=2.0
)

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

trainer.train()

val_preds = trainer.predict(dataset["test"])
logits = val_preds.predictions
labels = val_preds.label_ids

optimal_thresholds = optimize_threshold(logits, labels)

model.save_pretrained("./roberta_model")
tokenizer.save_pretrained("./roberta_model")

with open("./roberta_model/tag_classes.pkl", "wb") as f:
    pickle.dump(mlb.classes_, f)

with open("./roberta_model/optimal_thresholds.pkl", "wb") as f:
    pickle.dump(optimal_thresholds, f)
