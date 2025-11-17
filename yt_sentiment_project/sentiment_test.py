from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm  

import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict
import pandas as pd


model_path = "./sentiment_distilbert_imdb/best_model"


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

label_names = ["negative", "positive"]   # IMDb LABEL

def predict_sentiment(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze(0)

    pred_id = int(torch.argmax(probs))
    label = label_names[pred_id]
    score = float(probs[pred_id])
    return label, score


def predict_sentiment_batch(text_list):
    results = []
    for text in tqdm(text_list, desc="Sentiment inference"):
        label, score = predict_sentiment(text)
        results.append((label, score))
    return results

def analyze_youtube_comments(comments: List[Dict[str, str]],
                             excel_path: str = "yt_sentiment_results.xlsx"):
    """
    Takes a YouTube comment list (each item: {"author", "text"}) and performs:
    1) Sentiment analysis (positive/negative)
    2) Console summary output
    3) Bar chart visualization
    4) Excel export (detailed comments + sentiment summary)
    """

    if not comments:
        print("⚠ The comment list is empty.")
        return

    texts = [c["text"] for c in comments]

    print(f"Total comments: {len(texts)}")
    print("Running sentiment analysis...")

    # (label, score) per comment
    results = predict_sentiment_batch(texts)

    # ===== 1) Build detailed result table =====
    rows = []
    for c, (label, score) in zip(comments, results):
        rows.append({
            "author": c["author"],
            "text": c["text"],
            "sentiment": label,
            "score": score,
        })

    df = pd.DataFrame(rows)

    # ===== 2) Compute summary statistics =====
    labels = [label for (label, score) in results]
    counter = Counter(labels)

    num_pos = counter.get("positive", 0)
    num_neg = counter.get("negative", 0)
    total = num_pos + num_neg

    print("\n=== Sentiment Analysis Summary ===")
    print(f"positive: {num_pos}")
    print(f"negative: {num_neg}")
    print(f"total   : {total}")

    if total == 0:
        print("⚠ No valid sentiment labels found.")
        return

    pos_ratio = num_pos / total * 100
    neg_ratio = num_neg / total * 100
    print(f"positive ratio: {pos_ratio:.2f}%")
    print(f"negative ratio: {neg_ratio:.2f}%")

    summary_data = [
        {"sentiment": "positive", "count": num_pos, "ratio_percent": pos_ratio},
        {"sentiment": "negative", "count": num_neg, "ratio_percent": neg_ratio},
        {"sentiment": "total",    "count": total,   "ratio_percent": 100.0},
    ]
    df_summary = pd.DataFrame(summary_data)

    # ===== 3) Write to Excel =====
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="comments", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)

    print(f"\n📁 Excel file saved: {excel_path}")

    # ===== 4) Draw bar chart =====
    draw_sentiment_bar_chart(num_pos, num_neg, total)


def draw_sentiment_bar_chart(num_pos: int, num_neg: int, total: int):
    categories = ["positive", "negative"]
    counts = [num_pos, num_neg]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(categories, counts)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.title(f"Youtube Comments Sentiment (n={total})")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("yt_sentiment_summary.png", dpi=150)
    plt.show()
