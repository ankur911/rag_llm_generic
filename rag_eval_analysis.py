"""
RAG Evaluation Analysis (script version)
- Loads rag_eval_results.jsonl
- Summarizes metrics per category
- Saves box plots and mean bar charts to ./plots
- Writes category summary to category_summary.csv
"""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_PATH = Path("rag_eval_results.jsonl")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

METRICS = [
    "qa_cosine", "q_ctx_cosine", "a_ctx_cosine",
    "jaccard_q_ctx", "jaccard_a_ctx",
    "rougeL_recall_a_ctx", "answer_novelty_vs_ctx"
]

assert RESULTS_PATH.exists(), f"Results file not found: {RESULTS_PATH}"

# Load JSONL
records = []
summary = None
with RESULTS_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "summary" in obj:
            summary = obj["summary"]
            continue
        records.append(obj)

df = pd.json_normalize(records)
print(f"Loaded {len(df)} records.")
if summary is not None:
    print("Run summary:", summary)

# Prep columns
for m in METRICS:
    col = f"metrics.{m}"
    if col in df.columns:
        df[m] = df[col]
    else:
        df[m] = None

if "category" not in df.columns:
    df["category"] = "unknown"

df_metrics = df[["id","category","response_length","within_char_cap"] + METRICS].copy()

# Summary by category
summary_cols = ["response_length"] + METRICS
summary_by_cat = df_metrics.groupby("category")[summary_cols].agg(["count","mean","std"]).round(4)
summary_csv = Path("category_summary.csv")
summary_by_cat.to_csv(summary_csv)
print(f"Saved {summary_csv}")

# Box plots per metric
cats = sorted(df_metrics["category"].dropna().unique())
for m in METRICS:
    plt.figure()
    data = [df_metrics[df_metrics["category"]==c][m].dropna().values for c in cats]
    plt.boxplot(data, labels=cats, showfliers=True)
    plt.title(f"Box plot of {m} by category")
    plt.ylabel(m)
    plt.xticks(rotation=30, ha="right")
    out_path = PLOTS_DIR / f"box_{m}.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    print(f"Saved {out_path}")

# Mean bar charts per metric
means = df_metrics.groupby("category")[METRICS].mean().reset_index()
for m in METRICS:
    plt.figure()
    x = means["category"]
    y = means[m]
    plt.bar(x, y)
    plt.title(f"Mean {m} by category")
    plt.ylabel(m)
    plt.xticks(rotation=30, ha="right")
    out_path = PLOTS_DIR / f"mean_{m}.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    print(f"Saved {out_path}")
