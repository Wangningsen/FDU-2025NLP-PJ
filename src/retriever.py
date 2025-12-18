import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import faiss

from .config import (
    FAISS_INDEX_PATH,
    META_PATH,
    NUM_CANDIDATES,
    NUM_FINAL_CONTEXT,
    TEMPORAL_NUM_CANDIDATES,
    TEMPORAL_NUM_FINAL_CONTEXT,
    MAX_SNIPPET_CHARS,
)
from .embedding_qwen3 import embed_query
from .llm_client import chat
from .question_classifier import QuestionType


@dataclass
class MemoryChunk:
    index: int
    conv_id: str
    chunk_id: str
    start_turn: int
    end_turn: int
    text: str
    has_datetime: bool
    score: float


class MemoryRetriever:
    def __init__(self):
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.metas: List[Dict[str, Any]] = json.load(f)
        assert self.index.ntotal == len(
            self.metas
        ), "FAISS index size does not match meta entries."

    def _search_raw(self, query: str, top_k: int) -> List[MemoryChunk]:
        q_vec = embed_query(query)
        q_vec = q_vec.reshape(1, -1)
        scores, idxs = self.index.search(q_vec, top_k)
        res = []
        for score, idx in zip(scores[0], idxs[0]):
            meta = self.metas[int(idx)]
            res.append(
                MemoryChunk(
                    index=int(idx),
                    conv_id=meta["conv_id"],
                    chunk_id=meta["chunk_id"],
                    start_turn=meta["start_turn"],
                    end_turn=meta["end_turn"],
                    text=meta["text"],
                    has_datetime=bool(meta.get("has_datetime", False)),
                    score=float(score),
                )
            )
        return res

    def search_candidates(
        self,
        question: str,
        conv_id: Optional[str],
        qtype: QuestionType,
    ) -> List[MemoryChunk]:
        if qtype == QuestionType.TEMPORAL:
            top_k_raw = TEMPORAL_NUM_CANDIDATES
        else:
            top_k_raw = NUM_CANDIDATES

        raw = self._search_raw(question, top_k=top_k_raw * 3)

        # restrict to same conversation if conv_id is provided
        if conv_id is not None:
            raw = [c for c in raw if c.conv_id == conv_id]

        # for temporal questions, prefer chunks that contain date-like expressions
        if qtype == QuestionType.TEMPORAL:
            with_date = [c for c in raw if c.has_datetime]
            without_date = [c for c in raw if not c.has_datetime]
            # ensure we still have enough
            ordered = with_date + without_date
        else:
            ordered = raw

        # truncate to top_k_raw
        return ordered[:top_k_raw]

    def rerank_with_llm(
        self,
        question: str,
        candidates: List[MemoryChunk],
        qtype: QuestionType,
    ) -> List[MemoryChunk]:
        if not candidates:
            return []

        # choose final context size
        if qtype == QuestionType.TEMPORAL:
            k_final = TEMPORAL_NUM_FINAL_CONTEXT
        else:
            k_final = NUM_FINAL_CONTEXT

        # if very few candidates, skip rerank
        if len(candidates) <= k_final:
            return candidates

        # construct a prompt listing candidates
        snippet_lines = []
        for i, c in enumerate(candidates):
            text = c.text
            if len(text) > MAX_SNIPPET_CHARS:
                text = text[:MAX_SNIPPET_CHARS] + " ..."
            snippet_lines.append(
                f"[{i}] (conv={c.conv_id}, turns={c.start_turn}-{c.end_turn})\n{text}"
            )
        snippets_block = "\n\n".join(snippet_lines)

        system_prompt = (
            "You are a long-context memory selector.\n"
            "Given a question and a set of candidate memory snippets, "
            "you must choose the smallest subset of snippets that together "
            "contain enough information to answer the question.\n"
            "Return a JSON list of integer indices corresponding to the chosen snippets.\n"
        )
        type_hint = f"Question type: {qtype.value}"
        user_prompt = (
            f"{type_hint}\n\n"
            f"Question:\n{question}\n\n"
            f"Candidate snippets:\n{snippets_block}\n\n"
            "Please return ONLY a JSON list of integer indices, for example:\n"
            "[0, 2, 5]\n"
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]

        try:
            resp = chat(messages, max_tokens=256, temperature=0.0)
            idx_list = json.loads(resp)
            if not isinstance(idx_list, list):
                raise ValueError("LLM did not return a list.")
            idx_set = set(int(i) for i in idx_list)
        except Exception:
            # fallback to vector similarity order
            return candidates[:k_final]

        selected = []
        for i, c in enumerate(candidates):
            if i in idx_set:
                selected.append(c)
        # enforce max size
        selected = selected[:k_final]
        # if empty, fallback
        if not selected:
            selected = candidates[:k_final]
        return selected
