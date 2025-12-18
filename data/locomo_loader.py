# data/locomo_loader.py
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

CATEGORY_MAP = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


@dataclass
class LocomoQuestion:
    # 用于在整个数据集中唯一定位一条题目
    global_qid: str          # 例如 "sample3_q17"
    sample_id: str           # locomo 原始 sample_id
    local_qidx: int          # 在该 sample 中的第几题
    question: str
    answer: str              # 统一转 string,方便比较
    category_id: int
    category_name: str
    evidence: List[str]      # ["D1:3", ...]


def load_locomo(locomo_path: str | Path) -> Dict[str, Any]:
    """
    读取官方 locomo10.json,返回:
      - conversations: 原始对话对象列表
      - qa_list: 扁平化后的 LocomoQuestion 列表
    """
    locomo_path = Path(locomo_path)
    with locomo_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = []
    qa_list: List[LocomoQuestion] = []

    for sample in data:
        sample_id = str(sample.get("sample_id"))
        conversations.append(sample["conversation"])

        qa_items = sample.get("qa", [])
        for qidx, qa in enumerate(qa_items):
            q_text = qa["question"]
            ans = str(qa["answer"])  # 有些是数字,统一成字符串
            cat_id = int(qa["category"])
            cat_name = CATEGORY_MAP.get(cat_id, "unknown")
            evidence = qa.get("evidence", [])

            global_qid = f"{sample_id}_q{qidx}"

            qa_list.append(
                LocomoQuestion(
                    global_qid=global_qid,
                    sample_id=sample_id,
                    local_qidx=qidx,
                    question=q_text,
                    answer=ans,
                    category_id=cat_id,
                    category_name=cat_name,
                    evidence=evidence,
                )
            )

    return {
        "conversations": conversations,
        "qa_list": qa_list,
    }
