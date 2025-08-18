"""
Batch evaluation for RAG pipeline (direct, no server).
Reads a JSON file of questions (with categories), optional category filter,
then writes JSONL with results + a summary footer row.

Usage examples:
  python -m app.eval_rag --qfile iycf_eval_questions.json
  python -m app.eval_rag --qfile iycf_eval_questions.json --category fact_check --outfile eval_fact.jsonl
"""

import argparse
import json
import math
import statistics as stats
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.vectorstore import VectorStoreManager
from src.llm import get_llm
from src.pipeline import retrieve_and_generate
from src.config import (
    MAX_CHAR_LEN_RESP,
    TOP_K_RESULTS,
    PER_DOC_CHARS_ALLOWED,
    OVERALL_CHARS_ALLOWED,
)

DEFAULT_Q_PATH = Path("iycf_eval_questions.json")


# ---------------------- helpers: loading & CLI ----------------------

def load_questions(path: Path, only_category: Optional[str] = None) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if only_category:
        data = [q for q in data if q.get("category") == only_category]
    return data


def parse_args():
    p = argparse.ArgumentParser(description="Batch evaluation for RAG pipeline.")
    p.add_argument("--qfile", type=str, default=str(DEFAULT_Q_PATH),
                   help="Path to JSON file with questions.")
    p.add_argument("--category", type=str, default=None,
                   help="Optional category filter "
                        "(fact_check | summarization | noisy_grammar | multilingual | "
                        "semantic_paraphrase | on_topic_outside_context).")
    p.add_argument("--outfile", type=str, default="rag_eval_results.jsonl",
                   help="Path to save evaluation results (JSONL).")
    return p.parse_args()


# ---------------------- helpers: metrics ----------------------

def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)


def tokenize(s: str) -> List[str]:
    return [t for t in (s or "").lower().split() if t]


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def lcs_len(a: List[str], b: List[str]) -> int:
    # Longest Common Subsequence length (for ROUGE-L style recall proxy)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]


def rouge_l_recall(answer: str, context: str) -> float:
    a, c = tokenize(answer), tokenize(context)
    if not a:
        return 0.0
    return lcs_len(a, c) / len(a)


def novelty(answer: str, context: str) -> float:
    # Fraction of answer unigrams not present in context (lower => more grounded)
    a, c = tokenize(answer), tokenize(context)
    if not a:
        return 0.0
    cset = set(c)
    missing = [w for w in a if w not in cset]
    return len(missing) / len(a)


# ---------------------- main run ----------------------

def run():
    args = parse_args()
    qpath = Path(args.qfile)
    out_path = Path(args.outfile)

    questions = load_questions(qpath, args.category)
    if not questions:
        raise SystemExit(f"No questions found (file={qpath}, category={args.category}).")

    # Init pipeline components
    manager = VectorStoreManager()
    if not manager.load():
        raise RuntimeError("Vector store not loaded. Build once via app/main.py (builder).")
    llm = get_llm()
    emb_model = manager.embedding_model

    # Prepare retriever mirroring pipeline
    retriever = manager.vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

    records: List[Dict[str, Any]] = []

    for qitem in questions:
        qtext = qitem["text"]
        qcat = qitem.get("category", "unknown")
        qid = qitem.get("id")

        # Get model answer via pipeline (without requiring return_docs)
        res = retrieve_and_generate(manager, llm, qtext)
        ans = res.get("answer", "")
        sources = res.get("sources", [])

        # Reconstruct the exact-ish context as the pipeline does
        rdocs = retriever.invoke(qtext) or []
        # slice per doc, then cap overall
        ctx_chunks = [(d.page_content or "")[:PER_DOC_CHARS_ALLOWED] for d in rdocs]
        context = "\n---\n".join(ctx_chunks)[:OVERALL_CHARS_ALLOWED]
        retrieved_sources = sorted({d.metadata.get("source", "Unknown") for d in rdocs})

        # Response stats
        resp_len = len(ans)
        within_cap = resp_len <= MAX_CHAR_LEN_RESP

        # Embedding-based metrics (use same model as vector store)
        q_emb = emb_model.embed_query(qtext)
        a_emb = emb_model.embed_query(ans)
        c_emb = emb_model.embed_query(context) if context else None

        qa_cosine = cosine_sim(q_emb, a_emb)
        q_ctx_cosine = cosine_sim(q_emb, c_emb) if c_emb else 0.0
        a_ctx_cosine = cosine_sim(a_emb, c_emb) if c_emb else 0.0

        # Lexical proxies
        jaccard_q_ctx = jaccard(tokenize(qtext), tokenize(context))
        jaccard_a_ctx = jaccard(tokenize(ans), tokenize(context))
        rougeL_recall_a_ctx = rouge_l_recall(ans, context)
        answer_novelty_vs_ctx = novelty(ans, context)

        record = {
            "id": qid,
            "category": qcat,
            "query": qtext,
            "answer": ans,
            "response_length": resp_len,
            "within_char_cap": within_cap,
            "sources": sources,                  # pipeline-reported sources
            "retrieved_sources": list(retrieved_sources),  # from eval retriever
            "context": context,                  # reconstructed context string
            "metrics": {
                "qa_cosine": qa_cosine,
                "q_ctx_cosine": q_ctx_cosine,
                "a_ctx_cosine": a_ctx_cosine,
                "jaccard_q_ctx": jaccard_q_ctx,
                "jaccard_a_ctx": jaccard_a_ctx,
                "rougeL_recall_a_ctx": rougeL_recall_a_ctx,
                "answer_novelty_vs_ctx": answer_novelty_vs_ctx,
            },
        }
        records.append(record)

    # Write JSONL
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    def avg(metric_key: str) -> float:
        vals = [r["metrics"][metric_key] for r in records]
        return round(stats.mean(vals), 4) if vals else 0.0

    summary = {
        "n": len(records),
        "avg_qa_cosine": avg("qa_cosine"),
        "avg_q_ctx_cosine": avg("q_ctx_cosine"),
        "avg_a_ctx_cosine": avg("a_ctx_cosine"),
        "avg_jaccard_q_ctx": avg("jaccard_q_ctx"),
        "avg_jaccard_a_ctx": avg("jaccard_a_ctx"),
        "avg_rougeL_recall_a_ctx": avg("rougeL_recall_a_ctx"),
        "avg_answer_novelty": avg("answer_novelty_vs_ctx"),
        "pct_within_char_cap": round(100.0 * sum(r["within_char_cap"] for r in records) / len(records), 1),
        "avg_response_length": round(stats.mean([r["response_length"] for r in records]), 1),
    }

    print("== Summary ==")
    for k, v in summary.items():
        print(f"{k}: {v}")

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run()
