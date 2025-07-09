import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import pickle
import numpy as np

model = BertForSequenceClassification.from_pretrained("./bert")
tokenizer = BertTokenizerFast.from_pretrained("./bert")

with open("./bert/tag_classes.pkl", "rb") as f:
    classes = pickle.load(f)

model.eval()


def predict_tags(texts, threshold=0.4):
    inputs = tokenizer(texts, padding=True, truncation=True,
                       max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).cpu().numpy()
    for i, prob in enumerate(probs):
        print(f"Input: {texts[i]}")
        for tag, p in zip(classes, prob):
            if p >= 0.1:
                print(f"  {tag}: {p:.2f}")
        print()
    preds = (probs >= threshold).astype(int)

    results = []
    for pred in preds:
        results.append([cls for cls, flag in zip(classes, pred) if flag])
    return results


new_questions = [
    "Throughout the years, we have remained on good grips with our neighbors."]
print(predict_tags(new_questions))
