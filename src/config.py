import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
INDICES_DIR = os.path.join(PROJECT_ROOT, "indices")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Input data
LOCOMO_PATH = os.path.join(RAW_DIR, "locomo10.json")

# Processed paths
CHUNKS_PATH = os.path.join(PROCESSED_DIR, "chunks.jsonl")
QUESTIONS_PATH = os.path.join(PROCESSED_DIR, "questions.jsonl")

# Index files
FAISS_INDEX_PATH = os.path.join(INDICES_DIR, "faiss_chunks.index")
EMBEDDINGS_PATH = os.path.join(INDICES_DIR, "chunk_embeddings.npy")
META_PATH = os.path.join(INDICES_DIR, "meta.json")

# Result file for official eval
RESULT_JSON_PATH = os.path.join(RESULTS_DIR, "result.json")
# 如果你想基于助教提供的 baseline result.json 拷贝结构再替换答案,
# 可以在这里加一个路径 BASELINE_RESULT_PATH

# Embedding model
EMBED_MODEL_NAME = "/home/efs/nwang60/models/Qwen3-Embedding-8B"

# QA LLM config
QA_MODEL_PROVIDER = "claude"

# Claude on AWS Bedrock
CLAUDE_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
BEDROCK_REGION = "us-west-2"
ANTHROPIC_VERSION = "bedrock-2023-05-31"

# Open source Qwen3 for QA
QWEN_QA_MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"

# Question type classification
USE_LLM_FOR_QTYPE = True

# Chunking hyperparameters
WINDOW_TURNS = 64          # number of turns per chunk
STRIDE_TURNS = 32          # stride between chunks
MIN_TURNS_PER_CHUNK = 4    # ignore very small tails

# Retrieval hyperparameters
NUM_CANDIDATES = 8
NUM_FINAL_CONTEXT = 2

TEMPORAL_NUM_CANDIDATES = 10
TEMPORAL_NUM_FINAL_CONTEXT = 4

# How many characters of each memory snippet to show the QA model
MAX_SNIPPET_CHARS = 400
