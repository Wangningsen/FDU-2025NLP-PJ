import json
from enum import Enum
from typing import Tuple

from .llm_client import chat


class QuestionType(Enum):
    SINGLE_FACT = "single_fact"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"
    PREFERENCE = "preference"
    SUMMARY = "summary"
    OTHER = "other"


def _rule_based_classify(question: str) -> QuestionType:
    """Simple keyword-based fallback classifier."""
    q = question.lower().strip()

    # Temporal cues
    temporal_keywords = [
        "when ",
        " what day",
        " what date",
        "which day",
        "how long",
        " how many days",
        "first time",
        "last time",
        "before ",
        "after ",
        "earlier",
        "later",
        "since ",
        "until ",
    ]
    if any(kw in q for kw in temporal_keywords):
        return QuestionType.TEMPORAL

    # Preference / persona cues
    pref_keywords = [
        "favorite",
        "favourite",
        "like most",
        "likes most",
        "enjoy",
        "enjoys",
        "prefer",
        "prefers",
        "interested in",
        "usually does",
        "tends to",
        "hobbies",
        "hobby",
    ]
    if any(kw in q for kw in pref_keywords):
        return QuestionType.PREFERENCE

    # Summarization cues
    if q.startswith("summarize ") or q.startswith("summarise "):
        return QuestionType.SUMMARY
    if "summary of" in q or "overall summary" in q:
        return QuestionType.SUMMARY
    if q.startswith("what happened in") and "overall" in q:
        return QuestionType.SUMMARY

    # Very rough multi-hop cue
    if " and " in q or " or " in q:
        return QuestionType.MULTI_HOP

    return QuestionType.SINGLE_FACT


def _classify_with_llm(question: str) -> Tuple[QuestionType, str]:
    """Ask an LLM to classify the question.

    Returns (QuestionType, reason). Falls back to rule-based if parsing fails.
    """
    system_prompt = (
        "You are an expert annotator for a long-term conversational memory benchmark.\n"
        "Your task is to classify user questions into one of these categories:\n"
        "- SINGLE_FACT: can be answered from a single short span in the dialogue.\n"
        "- MULTI_HOP: requires combining multiple facts from different parts of the dialogue.\n"
        "- TEMPORAL: asks about when something happened, the order of events, or durations.\n"
        "- PREFERENCE: asks about stable likes, dislikes, or habits of a person.\n"
        "- SUMMARY: asks for a high-level recap or summary over many events.\n"
        "- OTHER: anything that does not clearly match the above.\n"
        "Always respond with a single JSON object with fields 'type' and 'reason'."
    )

    user_prompt = (
        f"Question: {question}\n\n"
        "Respond ONLY with valid JSON, no extra text. For example:\n"
        "{\"type\": \"TEMPORAL\", \"reason\": \"Asks when an event occurred.\"}"
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}],
        },
    ]

    raw = chat(messages, max_tokens=256, temperature=0.0)
    text = raw.strip()

    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        obj = json.loads(text[start:end])
        t = str(obj.get("type", "")).upper()
        reason = str(obj.get("reason", "")).strip()

        mapping = {
            "SINGLE_FACT": QuestionType.SINGLE_FACT,
            "MULTI_HOP": QuestionType.MULTI_HOP,
            "TEMPORAL": QuestionType.TEMPORAL,
            "PREFERENCE": QuestionType.PREFERENCE,
            "SUMMARY": QuestionType.SUMMARY,
            "OTHER": QuestionType.OTHER,
        }
        qtype = mapping.get(t)
        if qtype is None:
            qtype = _rule_based_classify(question)
            if not reason:
                reason = "Fell back to rule-based classification due to unknown type label."
        return qtype, reason or f"LLM-based classification as {qtype.value}."
    except Exception:
        # Parsing / JSON error → fall back to rules on the ORIGINAL question
        qtype = _rule_based_classify(question)
        return qtype, "LLM classification failed; used rule-based heuristics instead."


def classify_question(question: str) -> Tuple[QuestionType, str]:
    """Public entry point.

    For LoCoMo, we usually already know the category from the dataset and
    do not need this function. For unseen questions we use:
      1) rule-based classification first for obvious cases;
      2) otherwise an LLM-based classifier with robust fallback.
    """
    rule_type = _rule_based_classify(question)
    if rule_type in (QuestionType.TEMPORAL, QuestionType.PREFERENCE, QuestionType.SUMMARY):
        return rule_type, f"Rule-based classification as {rule_type.value} based on keywords."

    return _classify_with_llm(question)
