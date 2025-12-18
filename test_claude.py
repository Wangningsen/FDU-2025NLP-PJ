import json
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError, ConnectTimeoutError

# 你可以按需改这里
LIST_REGIONS = ["us-east-1", "us-west-2"]         # 用来 list_foundation_models 的 region
RUNTIME_REGIONS = ["us-west-2", "us-east-1"]      # 用来 invoke_model 的 region（按顺序优先尝试）
PROVIDER = "anthropic"
ONLY_ACTIVE = True                               # 只测试 ACTIVE 模型，想全测就改 False
MAX_TOKENS = 64                                  # 探测用，尽量小，省钱

def get_identity():
    try:
        sts = boto3.client("sts")
        ident = sts.get_caller_identity()
        return f"Account={ident.get('Account')} Arn={ident.get('Arn')}"
    except Exception as e:
        return f"(STS unavailable) {e}"

def list_anthropic_models(region: str):
    br = boto3.client("bedrock", region_name=region)
    resp = br.list_foundation_models(byProvider=PROVIDER)
    return resp.get("modelSummaries", [])

def _extract_text_from_anthropic_response(obj):
    # 新版 messages: {"content":[{"type":"text","text":"..."}]}
    if isinstance(obj, dict) and isinstance(obj.get("content"), list):
        parts = []
        for block in obj["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts).strip()
    # 旧版 completion: {"completion":"..."}
    if isinstance(obj, dict) and isinstance(obj.get("completion"), str):
        return obj["completion"].strip()
    return str(obj)[:2000]

CFG = Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 1, "mode": "standard"})

def invoke_once(runtime_region: str, model_id: str):
    brt = boto3.client("bedrock-runtime", region_name=runtime_region, config=CFG)

    body_messages = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 64,
        "temperature": 0,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Reply with exactly: OK"}]}],
    }

    print(f"  -> invoke region={runtime_region} modelId={model_id}", flush=True)
    try:
        resp = brt.invoke_model(
            modelId=model_id,
            body=json.dumps(body_messages),
            contentType="application/json",
            accept="application/json",
        )
        out = json.loads(resp["body"].read())
        return True, out
    except (EndpointConnectionError, ReadTimeoutError, ConnectTimeoutError) as e:
        return False, f"NETWORK/TIMEOUT: {repr(e)}"
    except ClientError as e:
        return False, f"{e.response['Error'].get('Code')}: {e.response['Error'].get('Message','')}"

def try_invoke_anywhere(base_model_id: str):
    # 同一个模型，尝试两种写法：原始 modelId + 带 us. 前缀（你示例里就是 us.anthropic...）
    candidates = [base_model_id]
    if not base_model_id.startswith("us."):
        candidates.append("us." + base_model_id)

    attempts = []
    for r in RUNTIME_REGIONS:
        for mid in candidates:
            ok, info = invoke_once(r, mid)
            attempts.append((r, mid, ok, info))
            if ok:
                return True, attempts
    return False, attempts

def main():
    print("Caller identity:", get_identity())
    print("=" * 80)

    # 1) list 模型（多 region 汇总去重）
    model_map = {}
    for region in LIST_REGIONS:
        try:
            models = list_anthropic_models(region)
            print(f"[list] region={region} found={len(models)}")
            for m in models:
                mid = m.get("modelId")
                if not mid:
                    continue
                status = (m.get("modelLifecycle") or {}).get("status", "N/A")
                if ONLY_ACTIVE and status != "ACTIVE":
                    continue
                # 去重：保留第一条即可
                model_map.setdefault(mid, {"status": status, "from_region": region})
        except ClientError as e:
            print(f"[list] region={region} failed: {e.response['Error'].get('Code')}: {e.response['Error'].get('Message')}")

    model_ids = sorted(model_map.keys())
    print("=" * 80)
    print(f"Models to test: {len(model_ids)} (ONLY_ACTIVE={ONLY_ACTIVE})")
    for mid in model_ids:
        meta = model_map[mid]
        print(f"  - {mid} (status={meta['status']} listed_from={meta['from_region']})")

    print("=" * 80)

    # 2) 逐个 invoke 探测
    results = []
    for mid in model_ids:
        ok, attempts = try_invoke_anywhere(mid)
        if ok:
            # 找到第一个成功的尝试
            success = next(a for a in attempts if a[2] is True)
            r, used_mid, _, text = success
            print(f"[OK] base={mid}")
            print(f"     runtime_region={r} used_modelId={used_mid}")
            print(f"     sample_output={text!r}")
        else:
            print(f"[FAIL] base={mid}")
            for (r, used_mid, _, err) in attempts:
                print(f"     tried region={r} modelId={used_mid} -> {err}")

        results.append((mid, ok))

    print("=" * 80)
    ok_cnt = sum(1 for _, ok in results if ok)
    print(f"Summary: {ok_cnt}/{len(results)} callable")

if __name__ == "__main__":
    main()

