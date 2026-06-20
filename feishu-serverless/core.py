"""
飞书事件 webhook 的纯逻辑（仅依赖标准库，便于跨平台与单测）。

被 app.py（Flask Web 函数）调用，可部署在阿里云函数计算 / 腾讯云 SCF 等国内 serverless，
飞书国内入口可稳定访问。重活（OCR+LLM）不在这里做，而是触发 GitHub repository_dispatch
交给 GitHub Actions，结果由 Actions 用飞书 app 推回原会话。
"""

import base64
import hashlib
import json
import re

ARXIV_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5})(v\d+)?(?:\.pdf)?", re.I)


def parse_text(message: dict) -> str:
    if (message or {}).get("message_type") != "text":
        return ""
    try:
        return json.loads(message.get("content") or "{}").get("text", "")
    except Exception:
        return ""


def extract_arxiv_url(text: str) -> str:
    if not text:
        return ""
    m = ARXIV_RE.search(text)
    if not m:
        return ""
    # 统一回传 abs 链接；下游脚本会再归一化为 pdf 交给 MinerU
    return f"https://arxiv.org/abs/{m.group(1)}"


def decide_task(text: str) -> str:
    """含「精读」走精读，否则默认实验配置。"""
    return "reading" if "精读" in (text or "") else "setup"


def decrypt_feishu(encrypt: str, encrypt_key: str) -> str:
    """飞书事件加密：AES-256-CBC，key = SHA256(encryptKey)，密文 base64，前 16 字节为 IV。"""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    digest = hashlib.sha256(encrypt_key.encode("utf-8")).digest()
    data = base64.b64decode(encrypt)
    iv, ciphertext = data[:16], data[16:]
    decryptor = Cipher(algorithms.AES(digest), modes.CBC(iv)).decryptor()
    plain = decryptor.update(ciphertext) + decryptor.finalize()
    plain = plain[: -plain[-1]]  # 去除 PKCS7 padding
    return plain.decode("utf-8")


def handle_event(payload, env: dict, dispatch, ack=None) -> tuple[int, dict]:
    """
    处理一条飞书事件，返回 (status_code, response_body)。

    dispatch(url, task, chat_id): 必填，触发后端（GitHub Actions）的回调。
    ack(chat_id, task, url): 可选，发即时回执的回调。
    """
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload or "{}")

    # 事件加密
    if "encrypt" in payload:
        payload = json.loads(decrypt_feishu(payload["encrypt"], env["FEISHU_ENCRYPT_KEY"]))

    # URL 验证
    if payload.get("type") == "url_verification":
        return 200, {"challenge": payload.get("challenge")}

    # 来源 token 校验（v2 在 header.token，v1 在 body.token）
    token = (payload.get("header") or {}).get("token") or payload.get("token")
    verification = env.get("FEISHU_VERIFICATION_TOKEN")
    if verification and token != verification:
        return 403, {"msg": "forbidden"}

    event_type = (payload.get("header") or {}).get("event_type") or (payload.get("event") or {}).get("type")
    if event_type == "im.message.receive_v1":
        message = (payload.get("event") or {}).get("message") or {}
        chat_id = message.get("chat_id")
        text = parse_text(message)
        url = extract_arxiv_url(text)
        if url and chat_id:
            task = decide_task(text)
            if ack:
                try:
                    ack(chat_id, task, url)
                except Exception:
                    pass
            dispatch(url, task, chat_id)

    return 200, {"code": 0}
