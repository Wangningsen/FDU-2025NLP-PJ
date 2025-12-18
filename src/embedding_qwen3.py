from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from .config import EMBED_MODEL_NAME

_embed_tokenizer = None
_embed_model = None


def _get_embed_model():
    global _embed_tokenizer, _embed_model
    if _embed_model is None:
        _embed_tokenizer = AutoTokenizer.from_pretrained(
            EMBED_MODEL_NAME,
            trust_remote_code=True,
        )
        _embed_model = AutoModel.from_pretrained(
            EMBED_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        _embed_model.eval()
    return _embed_tokenizer, _embed_model


def embed_texts(texts: List[str], batch_size: int = 8) -> np.ndarray:
    tok, model = _get_embed_model()
    all_vecs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            inputs = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            ).to(model.device)

            outputs = model(**inputs)
            hidden = outputs.last_hidden_state  # (b, seq, dim), likely bfloat16

            # mean pooling
            vec = hidden.mean(dim=1)

            # cast to float32 to keep FAISS happy and avoid bfloat16→numpy问题
            vec = vec.to(torch.float32)

            # L2 normalize
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)

            # now safe to move to cpu and numpy
            all_vecs.append(vec.cpu().numpy())

    return np.concatenate(all_vecs, axis=0)

def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
