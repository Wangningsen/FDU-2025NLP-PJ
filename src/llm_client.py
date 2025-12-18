import json
from typing import List, Dict

from .config import (
    QA_MODEL_PROVIDER,
    CLAUDE_MODEL_ID,
    BEDROCK_REGION,
    ANTHROPIC_VERSION,
    QWEN_QA_MODEL_NAME,
)

# Claude via AWS Bedrock
try:
    import boto3
except ImportError:
    boto3 = None

# Qwen via transformers (local)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

_bedrock_client = None
_qwen_tokenizer = None
_qwen_model = None




def _messages_to_prompt_for_qwen(messages: List[Dict]) -> str:
    """
    Very simple conversion from 'messages' (system/user/assistant with text content)
    to a plain text prompt for a chat style Qwen model.

    You may want to replace this with the official Qwen chat template in production.
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        # Bedrock style: content is a list of dicts, each may be text or others
        content_pieces = []
        for c in m.get("content", []):
            if c.get("type") == "text":
                content_pieces.append(c.get("text", ""))
        if not content_pieces:
            continue
        text = "\n".join(content_pieces)
        if role == "system":
            parts.append(f"[system]\n{text}")
        elif role == "assistant":
            parts.append(f"[assistant]\n{text}")
        else:
            parts.append(f"[user]\n{text}")
    # add assistant tag to signal generation
    parts.append("[assistant]\n")
    return "\n\n".join(parts)


def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=BEDROCK_REGION,
        )
    return _bedrock_client


def _chat_claude(messages, max_tokens=512, temperature=0.2) -> str:
    """
    Call Anthropic Claude via AWS Bedrock Messages API.

    我们内部的 messages 里可能包含:
      - role="system" 一条
      - role="user" 一条
    Bedrock 不接受 messages 里有 system 角色
    所以这里把 system 文本和 user 文本拼成一个大 user 提示发出去
    格式对齐你 test_claude.py 的用法
    """
    client = _get_bedrock_client()

    # 1) 抽取 system 文本和 user 文本
    system_texts = []
    user_texts = []

    for m in messages:
        role = m.get("role")
        content = m.get("content") or []
        if role == "system":
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    system_texts.append(c.get("text", ""))
        elif role == "user":
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    user_texts.append(c.get("text", ""))

    # 2) 合并成一个 user 提示
    #    [系统指令]
    #    <system_prompt>
    #
    #    <原来的 user prompt>
    combined = ""
    if system_texts:
        combined += "[System instructions]\n"
        combined += "\n\n".join(system_texts)
        combined += "\n\n"
    if user_texts:
        combined += "\n\n".join(user_texts)

    if not combined.strip():
        combined = "You are a helpful assistant."

    body = {
        "anthropic_version": ANTHROPIC_VERSION,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": combined,
                    }
                ],
            }
        ],
    }

    resp = client.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(resp["body"].read())

    # 新版 messages API: content 是一个 block list
    pieces = []
    for block in payload.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            pieces.append(block.get("text", ""))

    return "".join(pieces).strip()

def _load_qwen_model():
    global _qwen_tokenizer, _qwen_model
    if _qwen_model is None:
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError(
                "transformers is not installed but QA_MODEL_PROVIDER='qwen'"
            )
        _qwen_tokenizer = AutoTokenizer.from_pretrained(
            QWEN_QA_MODEL_NAME,
            trust_remote_code=True,
        )
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_QA_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        _qwen_model.eval()
    return _qwen_tokenizer, _qwen_model


def _chat_qwen(
    messages: List[Dict],
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    tok, model = _load_qwen_model()
    prompt = _messages_to_prompt_for_qwen(messages)
    device = model.device

    enc = tok(prompt, return_tensors="pt").to(device)
    input_len = enc["input_ids"].shape[1]

    gen_ids = model.generate(
        **enc,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 0.01),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    out_ids = gen_ids[0, input_len:]
    text = tok.decode(out_ids, skip_special_tokens=True)
    return text.strip()


def chat(
    messages: List[Dict],
    max_tokens: int = 512,
    temperature: float = 0.2,
    provider: str | None = None,
) -> str:
    """
    Unified chat entry point for the QA pipeline.

    messages: Bedrock style messages:
      [{"role": "system"|"user"|"assistant", "content": [{"type": "text", "text": "..."}]}]

    provider:
      - "claude": use Claude via Bedrock
      - "qwen": use local Qwen3 model via transformers
      - None: use QA_MODEL_PROVIDER from config
    """
    backend = provider or QA_MODEL_PROVIDER
    if backend == "claude":
        return _chat_claude(messages, max_tokens=max_tokens, temperature=temperature)
    elif backend == "qwen":
        return _chat_qwen(messages, max_tokens=max_tokens, temperature=temperature)
    else:
        raise ValueError(f"Unknown QA_MODEL_PROVIDER: {backend}")
