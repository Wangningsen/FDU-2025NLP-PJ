import json
import os

import faiss
import numpy as np

from .config import (
    CHUNKS_PATH,
    INDICES_DIR,
    FAISS_INDEX_PATH,
    EMBEDDINGS_PATH,
    META_PATH,
)
from .embedding_qwen3 import embed_texts


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    ensure_dir(INDICES_DIR)

    texts = []
    metas = []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            metas.append(
                {
                    "conv_id": rec["conv_id"],
                    "chunk_id": rec["chunk_id"],
                    "start_turn": rec["start_turn"],
                    "end_turn": rec["end_turn"],
                    "has_datetime": rec.get("has_datetime", False),
                    "text": rec["text"],
                }
            )

    print(f"Loaded {len(texts)} chunks, computing embeddings...")
    embs = embed_texts(texts, batch_size=8)
    dim = embs.shape[1]
    print(f"Vector dimension: {dim}")

    # inner product index (embeddings already L2 normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(EMBEDDINGS_PATH, embs)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    print(f"Wrote FAISS index to {FAISS_INDEX_PATH}")
    print(f"Wrote meta to {META_PATH}")


if __name__ == "__main__":
    main()
