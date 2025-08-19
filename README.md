# Genric RAG Pipeline, with WHO page as Source for knowlege basr

## Objective
This repository implements a robust, modular Retrieval-Augmented Generation (RAG) pipeline for parenting advice. It supports both CLI and API orchestration, enabling scalable, maintainable, and testable workflows for document retrieval and answer generation.

## Repo Structure
```
├── app/
│   ├── main.py        # CLI orchestration (PLAN-A, PLAN-B)
│   └── api.py         # FastAPI app for API orchestration
├── src/
│   ├── config.py      # Configuration and constants
│   ├── vectorstore.py # Vector store management
│   ├── embedding.py   # Embedding model loader
│   ├── llm.py         # LLM loader with fallback logic
│   ├── pipeline.py    # Query classification and RAG orchestration
├── requirements.txt   # Python dependencies
├── .gitignore         # Git ignore rules
└── README.md          # Project documentation
```

## Code Overview
### app/
- **main.py**: Entry point for CLI orchestration. Supports different pipeline plans and interacts with core logic in `src/`.
- **api.py**: FastAPI application exposing endpoints for health check, query, and test query. Uses core logic from `src/` for answer generation.

### src/
- **config.py**: Centralized configuration for models, vector store, prompt templates, and pipeline parameters.
- **vectorstore.py**: Handles loading, saving, and querying the FAISS vector store.
- **embedding.py**: Loads and manages the embedding model for document processing.
- **llm.py**: Loads the LLM, with fallback to a local model if remote endpoint is unavailable.
- **pipeline.py**: Orchestrates query classification, document retrieval, prompt construction, and answer generation.

## Pipeline Logic Flow
1. **Orchestration**: CLI (`main.py`) or API (`api.py`) receives a query.
2. **Classification**: Query is checked for toxicity/profanity.
3. **Retrieval**: Vector store retrieves relevant documents.
4. **Prompt Construction**: Context and question are formatted using a configurable prompt template.
5. **Answer Generation**: LLM generates a concise, factual answer, respecting character limits and context.
6. **Response**: Answer and sources are returned to the user via CLI or API.

## Setup Instructions
### 1. Clone the Repository
```powershell
git clone <repo-url>
cd rag_pipeline/
```

### 2. Set Up Virtual Environment
```powershell
python -m venv .venv
```

### 3. Activate Virtual Environment
- **Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```
- **Windows (cmd):**
```cmd
.venv\Scripts\activate.bat
```
- **Linux/macOS:**
```bash
source .venv/bin/activate
```

### 4. Upgrade pip
```powershell
python -m pip install --upgrade pip
```

### 5. Install Requirements
```powershell
pip install -r requirements.txt
```

## How to Run
### CLI Orchestration
```powershell
python -m app.main plan-a
```

### API Orchestration (FastAPI)
```powershell
uvicorn app.api:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

---
For further details, see comments in each source file or reach out to the maintainer.
