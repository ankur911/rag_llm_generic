# User Guide: Makefile for RAG+LLM Project

This project includes a `Makefile` to simplify setup, running, evaluation, and quality checks.  
Use it to avoid remembering long commands and keep workflows consistent.

---

## 🔹 Basics

Run commands with:

```bash
make <target>
```

List all available targets:

```bash
make help
```

---

## 🔹 Setup & Environment

| Target        | Description |
|---------------|-------------|
| `make setup`      | Install runtime dependencies (`requirements.txt`) |
| `make dev-setup`  | Install runtime + dev tools (`requirements.txt + requirements-dev.txt`) |
| `make venv`       | Create a virtual environment `.venv` (activate manually after) |
| `make freeze`     | Freeze current environment into `requirements-lock.txt` |

---

## 🔹 Application & API

| Target        | Description |
|---------------|-------------|
| `make build-vstore` | Build the FAISS vector store from `DOCUMENT_SOURCES` (required once before queries) |
| `make plan-a`       | Run CLI test (direct RAG pipeline) |
| `make plan-b`       | Run API smoke test: starts server, hits `/health` + `/query`, stops |
| `make api`          | Run FastAPI locally on `127.0.0.1:8000` (with `--reload`) |
| `make health`       | Check `/health` endpoint (requires server running, uses `curl` + `jq`) |

---

## 🔹 Evaluation

| Target        | Description |
|---------------|-------------|
| `make eval` | Run batch evaluation (`app/eval_rag.py`) on predefined questions. Outputs results to `rag_eval_results.jsonl` and prints summary stats. |

---

## 🔹 Quality & Testing

| Target        | Description |
|---------------|-------------|
| `make test`      | Run test suite with `pytest` |
| `make lint`      | Check code style with `flake8` |
| `make format`    | Auto-format code with `black` and `isort` |
| `make typecheck` | Run static type checks with `mypy` |
| `make clean`     | Remove caches, temp files, and eval artifacts |

---

## 🔹 Example Workflows

### 1. First-time setup
```bash
make dev-setup
make build-vstore
```

### 2. Run the API locally
```bash
make api
# In another terminal
make health
```

### 3. Evaluate pipeline
```bash
make eval
# Check rag_eval_results.jsonl for details
```

### 4. Code quality checks
```bash
make lint
make typecheck
make test
```

### 5. Clean workspace
```bash
make clean
```

---

## 🔹 Customizing Evaluation

The evaluation script lives in `app/eval_rag.py`.  
It feeds a set of questions into the pipeline and records answers, context, and metrics.

### Steps to customize:

1. Open `app/eval_rag.py`.
2. Locate the `QUESTIONS` list:
   ```python
   QUESTIONS = [
       "List key WHO recommendations for infant and young child feeding.",
       "When should I introduce solids to a 6-month-old?",
       # Add or replace with your own queries here
   ]
   ```
3. Modify or extend this list with your test questions.
4. Run the evaluation again:
   ```bash
   make eval
   ```
5. Results will appear in `rag_eval_results.jsonl` with one JSON record per question and a summary line.

This allows you to benchmark the pipeline with domain-specific queries and track improvements over time.

---

✅ With this Makefile + guide, you can easily manage development, testing, and evaluation without memorizing long commands.
