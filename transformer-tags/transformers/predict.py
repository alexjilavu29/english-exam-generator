import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from transformers import BertTokenizerFast, BertForSequenceClassification
import pickle
import numpy as np

model = RobertaForSequenceClassification.from_pretrained("./bert")
tokenizer = RobertaTokenizerFast.from_pretrained("./bert")

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


def predict_top_k(texts, k=2):
    inputs = tokenizer(texts, padding=True, truncation=True,
                       max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).cpu().numpy()
    topk_indices = np.argsort(probs, axis=1)[:, -k:]

    results = []
    for indices in topk_indices:
        results.append([classes[i] for i in indices])
    return results


new_questions = [
    "We had to take the car to work because the bus drivers are on strike."]

print(predict_top_k(new_questions, 2))

# threshold = 0.6
# count = 100
# while (count >= 3):
#     tags = predict_tags(new_questions, threshold)
#     count = len(tags)
#     threshold = threshold - 0.01
#     print(tags)
