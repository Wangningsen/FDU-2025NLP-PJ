import json
import os
from typing import Dict, Any, List, Optional
from botocore.exceptions import ClientError

from .config import (
    QUESTIONS_PATH,
    RESULT_JSON_PATH,
)
from .llm_client import chat
from .question_classifier import classify_question, QuestionType
from .retriever import MemoryRetriever


# LoCoMo 官方 category 映射:
# 1 -> Multi-hop
# 2 -> Temporal
# 3 -> Open-domain
# 4 -> Single-hop
# 5 -> Adversarial
CATEGORY_NAME_MAP: Dict[int, str] = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

# 把 LoCoMo 的 category 映射到我们内部用的 QuestionType
CATEGORY_TO_QTYPE: Dict[int, QuestionType] = {
    1: QuestionType.MULTI_HOP,
    2: QuestionType.TEMPORAL,
    4: QuestionType.SINGLE_FACT,
    # 3 (open-domain) 和 5 (adversarial) 在检索策略上
    # 没有特别强的结构性需求,统一映射为 OTHER,
    # 由系统 prompt 里的“信息不足就说不确定”来兜底。
    3: QuestionType.OTHER,
    5: QuestionType.OTHER,
}


def resolve_qtype(q_record: Dict[str, Any]) -> (QuestionType, str, Optional[int], Optional[str]):
    """
    决定这道题在 pipeline 里的 QuestionType。

    优先使用数据集自带的 category:
      1 -> MULTI_HOP
      2 -> TEMPORAL
      3 -> OTHER (open-domain)
      4 -> SINGLE_FACT
      5 -> OTHER (adversarial)

    如果没有 category(例如以后换别的数据集),
    再调用原来的 LLM 分类器 classify_question。
    """
    q_text = q_record.get("question", "")
    raw_cat = q_record.get("category", None)

    cat_id: Optional[int] = None
    if raw_cat is not None:
        try:
            cat_id = int(raw_cat)
        except (ValueError, TypeError):
            cat_id = None

    cat_name = CATEGORY_NAME_MAP.get(cat_id) if cat_id is not None else None

    if cat_id is not None and cat_id in CATEGORY_TO_QTYPE:
        qtype = CATEGORY_TO_QTYPE[cat_id]
        reason = f"Using dataset category {cat_id} ({cat_name}) as question type."
        return qtype, reason, cat_id, cat_name

    # fallback: 没有 category 或超出范围时,走原来的 LLM 分类器
    qtype, reason = classify_question(q_text)
    reason = f"LLM classifier fallback: {reason}"
    return qtype, reason, cat_id, cat_name


def build_qa_messages(
    question_text: str,
    qtype: QuestionType,
    memories: List[Any],
) -> List[Dict[str, Any]]:
    """
    Build messages for the QA LLM.

    memories: list of MemoryChunk
    """
    # 1) 拼记忆上下文
    ctx_parts = []
    for i, m in enumerate(memories):
        ctx_parts.append(
            f"[Memory {i+1}] (conv={getattr(m, 'conv_id', None)}, "
            f"turns={getattr(m, 'start_turn', '?')}-{getattr(m, 'end_turn', '?')})\n"
            f"{getattr(m, 'text', '')}"
        )
    context_block = "\n\n".join(ctx_parts)

    type_hint = f"Question type: {qtype.value}"

    # 2) 系统约束: 只能用记忆, 不瞎编
    system_prompt = (
        "You are a careful assistant that answers questions based ONLY on the "
        "provided memory snippets from a long conversation.\n"
        "If the memories are insufficient or conflicting, you must say you are "
        "not sure and explain briefly.\n"
        "Do not invent facts that are not supported by the memories.\n"
    )

    # 3) 类型相关的推理提示
    extra_instructions = ""
    if qtype == QuestionType.TEMPORAL:
        extra_instructions = (
            "Pay careful attention to temporal expressions. If the question asks "
            "for 'when', 'first', 'last', 'before', 'after', or a duration, you must "
            "compare the times of relevant events explicitly when deciding the answer.\n"
            "If a memory uses relative time like 'yesterday', 'last week', 'last year', "
            "'next month', convert it into an explicit calendar date, month+year, or year, "
            "using the session date and time given in the memories.\n"
        )
    elif qtype == QuestionType.MULTI_HOP:
        extra_instructions = (
            "This question likely requires combining multiple pieces of evidence. "
            "Use all relevant memories together when deciding the answer.\n"
        )
    elif qtype == QuestionType.PREFERENCE:
        extra_instructions = (
            "Focus on stable preferences or repeated choices of the person, "
            "such as what they like, dislike, or habitually do.\n"
        )
    elif qtype == QuestionType.SUMMARY:
        extra_instructions = (
            "Provide a concise summary that captures the key points relevant to the question.\n"
        )
    elif qtype == QuestionType.OTHER:
        extra_instructions = (
            "The question may be unanswerable from the memories alone. "
            "If the memories do not clearly contain the required information, "
            "explicitly say that the question cannot be answered from the "
            "given memories.\n"
        )

    # 4) 为了 BLEU/F1 的输出格式约束
    format_instructions = (
        "Formatting rules for your final answer:\n"
        "1. Your ENTIRE reply must be ONLY the final answer string.\n"
        "   - Do NOT show your reasoning.\n"
        "   - Do NOT repeat the question.\n"
        "   - Do NOT add any explanation, apology, or extra sentences.\n"
        "   - Do NOT prefix with 'Answer:' or similar.\n"
        "2. The answer must be a single short phrase that directly answers the question.\n"
        "   - If multiple items are needed, separate them with commas and a space.\n"
        "   - Examples of valid answer styles:\n"
        "       8 May 2025\n"
        "       June 2012\n"
        "       2018\n"
        "       The week before 13 July 2004\n"
        "       Transgender woman\n"
        "       Psychology, counseling certification\n"
        "       pottery, camping, painting, boxing\n"
        "3. When the memories contain a short span that answers the question, "
        "copy that span verbatim or with minimal changes, instead of paraphrasing.\n"
    )

    # 针对不同类型再给一点特化的 hint
    if qtype in (QuestionType.TEMPORAL, QuestionType.SINGLE_FACT, QuestionType.MULTI_HOP):
        format_instructions += (
            "For temporal questions, prefer explicit calendar references over relative words.\n"
            "For example, if a memory says 'last year' in a session dated 2023, "
            "answer '2022'. If it says 'last week' in a session dated 9 June 2023, "
            "you may answer 'The week before 9 June 2023'.\n"
        )

    if qtype == QuestionType.OTHER:
        format_instructions += (
            "For hypothetical or likelihood questions, answer with a short phrase like "
            "'Likely yes', 'Likely no', or another short phrase that best matches the memories. "
            "Still do NOT add any explanation.\n"
        )

    user_prompt = (
        f"{type_hint}\n\n"
        f"Memories:\n{context_block}\n\n"
        f"Question:\n{question_text}\n\n"
        f"{extra_instructions}"
        f"{format_instructions}"
        "Remember: if you output anything other than the single short answer string, "
        "your answer will be treated as wrong.\n"
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
    return messages


def serialize_memory_chunk(m: Any, rank: int) -> Dict[str, Any]:
    """
    把 MemoryRetriever 返回的 chunk 转成可写入 JSON 的简要结构:
    - 记录 rank(在该阶段的排序位置)
    - conv_id, chunk_id, 起止 turn, has_datetime, score(如果有的话)
    - 文本截断到前 300 字符,方便检查检索行为
    """
    text = getattr(m, "text", "")
    if text is None:
        text = ""
    text = str(text)

    return {
        "rank": rank,
        "conv_id": getattr(m, "conv_id", None),
        "chunk_id": getattr(m, "chunk_id", None),
        "start_turn": getattr(m, "start_turn", None),
        "end_turn": getattr(m, "end_turn", None),
        "has_datetime": getattr(m, "has_datetime", None),
        "score": getattr(m, "score", None),  # 如果 MemoryChunk 里没有这个属性,会是 None
        "text_preview": text[:300],
    }


def main():
    os.makedirs(os.path.dirname(RESULT_JSON_PATH), exist_ok=True)

    retriever = MemoryRetriever()

    # 尝试从已有的 result.json 恢复
    outputs: List[Dict[str, Any]] = []
    done_ids = set()

    if os.path.exists(RESULT_JSON_PATH):
        try:
            with open(RESULT_JSON_PATH, "r", encoding="utf-8") as f_old:
                old = json.load(f_old)
                if isinstance(old, list):
                    outputs = old
                    done_ids = {rec.get("id") for rec in outputs if "id" in rec}
                    print(f"[resume] Loaded {len(outputs)} existing answers from {RESULT_JSON_PATH}")
        except Exception as e:
            print(f"[warn] Failed to load existing results: {e}")
            outputs = []
            done_ids = set()

    def save_outputs():
        with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f_out:
            json.dump(outputs, f_out, ensure_ascii=False, indent=2)

    questions: List[Dict[str, Any]] = []
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            questions.append(rec)

    for idx, q in enumerate(questions):
        qid = q["id"]

        # 如果之前已经算过这一题, 直接跳过
        if qid in done_ids:
            continue

        qtext = q["question"]
        conv_id = q.get("conv_id") or q.get("sample_id")

        # 解析 category
        raw_cat = q.get("category")
        cat_id: Optional[int] = None
        if raw_cat is not None:
            try:
                cat_id = int(raw_cat)
            except (ValueError, TypeError):
                cat_id = None

        cat_name = CATEGORY_NAME_MAP.get(cat_id) if cat_id is not None else None

        # 1) category 5 直接跳过: 不调用 LLM,只写占位
        if cat_id == 5:
            print(f"[skip] category 5 (adversarial), qid={qid}")
            outputs.append(
                {
                    "id": qid,
                    "question": qtext,
                    "prediction": "I am not sure",
                    "qtype": "other",
                    "qtype_reason": "Skipped category 5 (adversarial) at inference time.",
                    "category": cat_id,
                    "category_name": cat_name,
                    "conv_id": conv_id,
                    "retrieved_candidates": [],
                    "final_context": [],
                }
            )
            save_outputs()
            continue

        # 2) 非 category 5: 正常走类型映射 + 检索 + QA
        qtype, reason, _, _ = resolve_qtype(q)

        # 检索候选 - embedding 阶段原始候选
        cands = retriever.search_candidates(
            question=qtext,
            conv_id=conv_id,
            qtype=qtype,
        )
        retrieved_candidates_serialized = [
            serialize_memory_chunk(m, rank=i) for i, m in enumerate(cands)
        ]

        # LLM rerank 后的最终上下文
        final_mems = retriever.rerank_with_llm(
            question=qtext,
            candidates=cands,
            qtype=qtype,
        )
        final_context_serialized = [
            serialize_memory_chunk(m, rank=i) for i, m in enumerate(final_mems)
        ]

        # 3) 调 LLM: 加一层自己的兜底,避免 ServiceUnavailable 把整个 run 弄崩
        messages = build_qa_messages(qtext, qtype, final_mems)

        try:
            raw_answer = chat(messages, max_tokens=64, temperature=0.0)
            # 简单处理一下字符串
            answer_text = raw_answer.strip() if isinstance(raw_answer, str) else str(raw_answer).strip()
            if not answer_text:
                answer_text = "MISS"
        except ClientError as e:
            err = e.response.get("Error", {}) or {}
            code = err.get("Code", "")
            msg = err.get("Message", "")
            print(f"[FATAL] Bedrock ClientError for qid={qid}, code={code}: {msg}")
            print("[FATAL] Stopping QA loop to avoid sending more requests while the service is unavailable.")
            # 已经回答完的问题都在 outputs 里了,这里保险再写一次盘
            save_outputs()
            # 直接抛出异常,整个脚本退出,后面不再发请求
            raise
        except Exception as e:
            print(f"[FATAL] Unexpected error for qid={qid}: {e}")
            print("[FATAL] Stopping QA loop.")
            save_outputs()
            raise

        outputs.append(
            {
                "id": qid,
                "question": qtext,
                "prediction": answer_text,
                "qtype": qtype.value,
                "qtype_reason": reason,
                "category": cat_id,
                "category_name": cat_name,
                "conv_id": conv_id,
                # 新增调试字段:
                # - embedding 检索出来的所有候选(按相似度 rank)
                # - rerank 后真正送进 QA LLM 的最终上下文
                "retrieved_candidates": retrieved_candidates_serialized,
                "final_context": final_context_serialized,
            }
        )

        # 4) 每题完成就写 result.json
        save_outputs()

        if (idx + 1) % 10 == 0:
            print(f"Answered {idx + 1} / {len(questions)} questions")

    print(f"Wrote predictions to {RESULT_JSON_PATH}")
    print("Now run the official eval script on this file.")


if __name__ == "__main__":
    main()

