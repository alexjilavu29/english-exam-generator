from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from transformers import RobertaTokenizerFast
import torch.nn as nn
from transformers import Trainer
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import transformers
import torch
from torch.nn import BCEWithLogitsLoss
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from data import questions_with_tags

model = "roberta-large"

texts = [item["body"] for item in questions_with_tags]
tag_lists = [item["tags"] for item in questions_with_tags]

all_tags = sorted(list({tag for tags in tag_lists for tag in tags}))

mlb = MultiLabelBinarizer()
tag_matrix = mlb.fit_transform(tag_lists)

label_counts = np.sum(tag_matrix, axis=0)
neg_counts = tag_matrix.shape[0] - label_counts
weights = torch.tensor(
    neg_counts / (label_counts + 1e-6), dtype=torch.float32)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.device)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


class_weights = torch.tensor(weights, dtype=torch.float32)

dataset = Dataset.from_dict({
    "text": texts,
    "labels": tag_matrix.astype(np.float32)
})

# tokenizer = BertTokenizerFast.from_pretrained(model)
tokenizer = RobertaTokenizerFast.from_pretrained(model)
data_collator = transformers.DataCollatorWithPadding(tokenizer)


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

model = RobertaForSequenceClassification.from_pretrained(
    model,
    num_labels=len(mlb.classes_),
    problem_type="multi_label_classification"
)


def compute_metrics(pred):
    logits, labels = pred
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.tensor(logits))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= 0.5)] = 1  # threshold de 0.5

    from sklearn.metrics import classification_report

    report = classification_report(
        labels, y_pred, output_dict=True, zero_division=0)

    return {
        "f1_micro": report["micro avg"]["f1-score"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_micro": report["micro avg"]["precision"],
        "recall_micro": report["micro avg"]["recall"],
    }


training_args = TrainingArguments(
    output_dir="./bert",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=16,
    weight_decay=0.005,
    warmup_steps=500,
    lr_scheduler_type="linear",
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# trainer = WeightedTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
#     class_weights=class_weights,
# )

trainer.train()

model.save_pretrained("./bert")
tokenizer.save_pretrained("./bert")

with open("./bert/tag_classes.pkl", "wb") as f:
    pickle.dump(mlb.classes_, f)
