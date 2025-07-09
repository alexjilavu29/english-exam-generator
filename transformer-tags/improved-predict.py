import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
import pickle
import numpy as np

model = RobertaForSequenceClassification.from_pretrained("./roberta_model")
tokenizer = RobertaTokenizerFast.from_pretrained("./roberta_model")

with open("./roberta_model/tag_classes.pkl", "rb") as f:
    classes = pickle.load(f)

with open("./roberta_model/optimal_thresholds.pkl", "rb") as f:
    optimal_thresholds = pickle.load(f)

model.eval()


def predict_tags(texts, thresholds=None, show_probs=False):
    if thresholds is None:
        thresholds = optimal_thresholds

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).cpu().numpy()

    if show_probs:
        for i, prob in enumerate(probs):
            print(f"Input: {texts[i]}")
            for tag, p, thresh in zip(classes, prob, thresholds):
                if p >= 0.1:
                    print(f"  {tag}: {p:.2f} (threshold: {thresh:.2f})")
            print()

    preds = np.array([
        [p >= thresh for p, thresh in zip(sample, thresholds)]
        for sample in probs
    ])

    results = []
    for pred in preds:
        results.append([cls for cls, flag in zip(classes, pred) if flag])
    return results


def predict_top_k(texts, k=3, min_prob=0.1):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).cpu().numpy()

    results = []
    for prob in probs:
        valid_indices = [i for i, p in enumerate(prob) if p >= min_prob]
        valid_probs = [prob[i] for i in valid_indices]
        valid_tags = [classes[i] for i in valid_indices]

        sorted_indices = np.argsort(valid_probs)[-k:]
        top_tags = [valid_tags[i]
                    for i in sorted_indices[::-1]]
        results.append(top_tags)

    return results


new_questions = [
    "We had to take the car to work because the bus drivers are on strike.",

]

print("Predictions with optimal thresholds:")
print(predict_tags(new_questions, show_probs=True))

print("\nTop 3 predictions:")
print(predict_top_k(new_questions, k=3))
