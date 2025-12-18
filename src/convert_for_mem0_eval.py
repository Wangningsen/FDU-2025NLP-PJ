import json
import os

from .config import QUESTIONS_PATH, RESULT_JSON_PATH


def main():
    # 1) 读 questions.jsonl, 建 id -> meta
    meta_by_id = {}
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec["id"]
            meta_by_id[qid] = rec

    print(f"Loaded {len(meta_by_id)} questions from {QUESTIONS_PATH}")

    # 2) 读 qa_pipeline 的预测结果
    with open(RESULT_JSON_PATH, "r", encoding="utf-8") as f:
        preds = json.load(f)

    print(f"Loaded {len(preds)} predictions from {RESULT_JSON_PATH}")

    grouped = {"locomo10_ours": []}
    missing = 0

    for p in preds:
        qid = p["id"]
        meta = meta_by_id.get(qid)
        if meta is None:
            missing += 1
            continue

        grouped["locomo10_ours"].append(
            {
                # 保留 id 和 conv_id 方便后续分析
                "id": qid,
                "conv_id": meta.get("conv_id") or meta.get("sample_id"),

                # eval 真正关心的四个字段
                "question": meta.get("question", ""),
                "answer": str(meta.get("answer", "")),
                "response": str(p.get("prediction", "")),
                "category": meta.get("category", 0),
            }
        )

    if missing > 0:
        print(f"Warning: {missing} predictions had no matching question metadata")

    out_path = os.path.join(
        os.path.dirname(RESULT_JSON_PATH),
        "locomo10_for_mem0_eval.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)

    print(f"Wrote eval input to {out_path}")


if __name__ == "__main__":
    main()

