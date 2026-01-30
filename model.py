import torch
import csv
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from labels import LABEL_MAP

MODEL_PATH = "./saved_model"
OUTPUT_FILE = "predictions.csv"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


def predict_single(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

    class_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0][class_id].item()

    label = LABEL_MAP[class_id]

    print("DEBUG → logits:", logits)
    print("DEBUG → probs:", probs)
    print("DEBUG → predicted:", class_id, label, confidence)

    return label, round(confidence * 100, 2)



def predict_batch(texts: list):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    class_ids = torch.argmax(outputs.logits, dim=1).tolist()
    labels = [LABEL_MAP[i] for i in class_ids]

    for t, l in zip(texts, labels):
        save_to_file(t, l)

    return labels


def save_to_file(text, label):
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), text, label])

