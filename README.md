# Agentic-Like Memory with Category-Aware Prompts and LLM Reranking for Long-Term Conversational QA
This repository packages a long-context conversational QA pipeline for the LoCoMo benchmark. Dialogs are chunked into memory windows, indexed with dense embeddings, and queried with category-aware prompts plus LLM-based reranking to surface the right context for each question.
You can preprocess LoCoMo, build a FAISS index with Qwen3 embeddings, answer questions with Claude (via Bedrock) or local Qwen, and export predictions for evaluation.


## 🔥 Highlights
- LoCoMo conversations flattened and chunked with temporal cues via `src/prepare_data.py` (category 5 questions skipped by default).
- Dense retrieval powered by Qwen3-Embedding-8B and FAISS, built with `src/build_index.py`.
- Category-aware question typing using dataset labels with LLM/rule fallback in `src/question_classifier.py`.
- Memory retrieval + LLM reranking pipeline in `src.retriever.py` to pick the smallest useful context.
- End-to-end QA runner in `src/qa_pipeline.py` with Bedrock Claude (default) or local Qwen3-235B backends.
- Prediction export to the Mem0 eval JSON format via `src.convert_for_mem0_eval.py`.



## 🚀 Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU strongly recommended for Qwen3 embedding/QA models (CPU/`faiss-cpu` works but is slow)
- AWS Bedrock access for Anthropic Claude if using the default QA provider

```
git clone <REPO_URL>
cd FDU-2025NLP-PJ
conda create -n memory python=3.10
conda activate memory
pip install -U torch transformers faiss-cpu boto3 botocore numpy
```

Choose `faiss-gpu` and a GPU-enabled PyTorch build if you want GPU indexing/search.

### Environment
- AWS credentials and region for Bedrock Claude: set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION` (matches `CLAUDE_MODEL_ID` in `src/config.py`).
- Hugging Face access for model downloads: `EMBED_MODEL_NAME` points to `Qwen3-Embedding-8B` and `QWEN_QA_MODEL_NAME` is `Qwen/Qwen3-235B-A22B-Instruct-2507`; update `src/config.py` to a local path/model you have available.
- Switch QA backend by editing `QA_MODEL_PROVIDER` in `src/config.py` (`claude` or `qwen`).
- Sanity check after install:
  ```
  python - <<'PY'
  from src import config
  print("Raw data path:", config.LOCOMO_PATH)
  PY
  ```


## Quick Start
1) Prepare LoCoMo splits (regenerates `data/processed/` from `data/raw/locomo10.json`):
```
python -m src.prepare_data
```

2) Build the FAISS index and embedding cache (writes to `indices/`):
```
python -m src.build_index
```
Make sure `EMBED_MODEL_NAME` in `src/config.py` points to a reachable Qwen3 embedding checkpoint.

3) Run QA to produce predictions (writes to `results/result.json` and logs retrieved context):
```
python -m src.qa_pipeline
```
Default provider uses Bedrock Claude; set `QA_MODEL_PROVIDER="qwen"` in `src/config.py` if you prefer the local model (very large).

4) Convert predictions to the Mem0-style eval input:
```
python -m src.convert_for_mem0_eval
```
This creates `results/locomo10_for_mem0_eval.json`, which you can feed to the official LoCoMo/Mem0 evaluator (not included here).


## 📊 Datasets
- Supported benchmark: LoCoMo (long-term conversational memory).
- Raw input expected at `data/raw/locomo10.json` (already included here for convenience).
- Preprocessing (`python -m src.prepare_data`) flattens sessions, detects temporal cues, slides a window over turns (`WINDOW_TURNS`, `STRIDE_TURNS` in `src/config.py`), drops very short tails, and writes:
  - `data/processed/chunks.jsonl`: memory chunks with turn ranges and temporal flags.
  - `data/processed/questions.jsonl`: flattened QA with category ids (adversarial category 5 skipped).
- Data helper: `data/locomo_loader.py` parses the raw file into structured objects if you need programmatic access.

Expected layout:
```
data/
  raw/
    locomo10.json
  processed/
    chunks.jsonl
    questions.jsonl
    questions_10.jsonl  # sample slice
  locomo_loader.py
```




## Evaluation
- Main entrypoint: `python -m src.qa_pipeline` reads `data/processed/questions.jsonl`, retrieves memories from the FAISS index (`indices/`), reranks with `src.retriever.MemoryRetriever`, and queries the QA LLM. Outputs go to `results/result.json` (contains predictions plus retrieved context for debugging).
- Export for official scoring: `python -m src.convert_for_mem0_eval` merges predictions with question metadata and writes `results/locomo10_for_mem0_eval.json` (and versioned variants if you keep multiple runs). Use the official LoCoMo/Mem0 evaluator on that file to obtain metrics (not included in this repo).
- Retrieval knobs live in `src/config.py` (`NUM_CANDIDATES`, `NUM_FINAL_CONTEXT`, and temporal variants). Change `QA_MODEL_PROVIDER`/model ids there to switch QA backends.


## Models and Results
- Default models (configurable in `src/config.py`):
  - Embeddings: `Qwen3-Embedding-8B` (`EMBED_MODEL_NAME`, local path by default).
  - QA: Claude Sonnet on Bedrock (`CLAUDE_MODEL_ID`) or local `Qwen/Qwen3-235B-A22B-Instruct-2507`.
- Existing prediction artifacts (no metrics computed in-repo): `results/result_v1.json`, `results/result_v2.json`, `results/result_v2_verbose.json`, `results/locomo10_for_mem0_eval_v1.json`, `results/locomo10_for_mem0_eval_v2.json`, and the default `results/result.json` if you run the pipeline.



## 📁 Project Structure
```
.
├── README.md
├── .gitignore
├── data/
|   ├── locomo_loader.py
|   ├── raw/
|   |   └── locomo10.json
|   └── processed/
|       ├── chunks.jsonl
|       ├── questions.jsonl
|       └── questions_10.jsonl
├── results/
|   ├── locomo10_for_mem0_eval_v1.json
|   ├── locomo10_for_mem0_eval_v2.json
|   ├── result_v1.json
|   ├── result_v2.json
|   ├── result_v2_verbose.json
|   └── result.json
├── src/
|   ├── __init__.py
|   ├── build_index.py
|   ├── config.py
|   ├── convert_for_mem0_eval.py
|   ├── embedding_qwen3.py
|   ├── llm_client.py
|   ├── prepare_data.py
|   ├── qa_pipeline.py
|   ├── question_classifier.py
|   ├── retriever.py
|   └── test_claude.py
└── report/
    └── report.pdf
```





