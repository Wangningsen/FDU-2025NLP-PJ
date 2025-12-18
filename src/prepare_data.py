import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

from .config import (
    LOCOMO_PATH,
    CHUNKS_PATH,
    QUESTIONS_PATH,
    WINDOW_TURNS,
    STRIDE_TURNS,
    MIN_TURNS_PER_CHUNK,
)


# Lightweight date/time heuristic: used as a signal for temporal routing.
_DATE_PATTERN = re.compile(
    r"("
    r"\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b"
    r"|\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}\b"
    r"|\b\d{4}-\d{1,2}-\d{1,2}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b"
    r"|\b(today|yesterday|tomorrow|last week|next week|last month|next month|last year|next year)\b"
    r"|\b\d{1,2}:\d{2}(\s*[ap]m|\s*AM|\s*PM)?\b"
    r")",
    flags=re.IGNORECASE,
)


def _load_locomo(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load the LoCoMo JSON file.

    The file is a list of samples; each sample has:
      - sample_id
      - conversation (sessions and metadata)
      - qa (question-answer annotations)
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected LoCoMo data to be a list of samples")
    return data


def _flatten_sessions(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten all sessions in a conversation into an ordered list of turns.

    Each element in the returned list has:
      - text: rendered turn text (including speaker if present)
      - session_num: int
      - dia_id: original dialog id (if any)
      - has_datetime: whether this line clearly contains temporal info
    """
    sessions: List[tuple[int, str, List[Dict[str, Any]]]] = []

    for key, value in conversation.items():
        if not key.startswith("session_"):
            continue
        # We only want keys like "session_1", not "session_1_summary"
        suffix = key[len("session_") :]
        if not suffix.isdigit():
            continue

        num = int(suffix)
        dialog = value  # list of dialog turns
        if not isinstance(dialog, list):
            continue

        dt_key = f"session_{num}_date_time"
        dt_str = conversation.get(dt_key, "")
        sessions.append((num, dt_str, dialog))

    # Sort sessions by numeric id to preserve chronology
    sessions.sort(key=lambda x: x[0])

    turns: List[Dict[str, Any]] = []
    for num, dt_str, dialog in sessions:
        # Synthetic meta-turn with the session timestamp
        if dt_str:
            meta_text = f"[Session {num} on {dt_str}]"
            turns.append(
                {
                    "text": meta_text,
                    "session_num": num,
                    "dia_id": None,
                    "has_datetime": True,
                }
            )

        for utt in dialog:
            speaker = utt.get("speaker", "")
            text = (utt.get("text") or "").strip()
            dia_id = utt.get("dia_id")
            if not text:
                continue

            line = f"{speaker}: {text}" if speaker else text
            has_dt = bool(_DATE_PATTERN.search(text))
            turns.append(
                {
                    "text": line,
                    "session_num": num,
                    "dia_id": dia_id,
                    "has_datetime": has_dt,
                }
            )

    return turns


def _iter_chunks(
    turns: List[Dict[str, Any]],
    sample_id: str,
):
    """Yield chunk records from a list of flattened turns."""
    total = len(turns)
    if total == 0:
        return

    start = 0
    while start < total:
        end = min(start + WINDOW_TURNS, total)
        window = turns[start:end]
        if len(window) < MIN_TURNS_PER_CHUNK:
            # Drop very short trailing window
            break

        chunk_text = "\n".join(t["text"] for t in window)
        has_dt = any(t["has_datetime"] for t in window)

        yield {
            "sample_id": sample_id,
            "conv_id": sample_id,  # use sample_id as conversation id
            "chunk_id": f"{sample_id}_{start}_{end-1}",
            "start_turn": start,
            "end_turn": end - 1,
            "has_datetime": has_dt,
            "text": chunk_text,
        }

        start += STRIDE_TURNS


def main() -> None:
    # Ensure parent dirs exist
    Path(CHUNKS_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(QUESTIONS_PATH).parent.mkdir(parents=True, exist_ok=True)

    data = _load_locomo(LOCOMO_PATH)

    with Path(CHUNKS_PATH).open("w", encoding="utf-8") as chunks_f, Path(
        QUESTIONS_PATH
    ).open("w", encoding="utf-8") as q_f:
        for sample in data:
            sample_id = str(sample.get("sample_id"))
            conversation = sample.get("conversation")
            if not sample_id or not isinstance(conversation, dict):
                # Skip malformed entries
                continue

            # 1) Build chunks from dialogs
            turns = _flatten_sessions(conversation)
            for chunk in _iter_chunks(turns, sample_id):
                chunks_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            # 2) Flatten QA annotations with full metadata
            qa_list = sample.get("qa", [])
            if not isinstance(qa_list, list):
                continue

            for idx, qa in enumerate(qa_list):
                question = qa.get("question")
                if not question:
                    continue

                # 规范化 category, 并且跳过 category 5 (adversarial)
                raw_cat = qa.get("category")
                cat_id = None
                try:
                    cat_id = int(raw_cat)
                except (TypeError, ValueError):
                    cat_id = None

                if cat_id == 5:
                    # 完全从下游 pipeline 中剔除 adversarial 类问题
                    continue

                qid = f"{sample_id}_q{idx}"
                q_rec: Dict[str, Any] = {
                    "id": qid,
                    "sample_id": sample_id,
                    "conv_id": sample_id,
                    "question": question,
                    # Not used at eval time but useful for analysis / debugging
                    "answer": qa.get("answer"),
                    "category": cat_id,
                    "evidence": qa.get("evidence", []),
                }
                q_f.write(json.dumps(q_rec, ensure_ascii=False) + "\n")
         
if __name__ == "__main__":
    main()
