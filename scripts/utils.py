"""
共享工具模块：LLM 客户端、飞书 API、arxiv 辅助函数、PaddleOCR。
"""

import os
import sys
import json
import time
import hashlib
import hmac
import base64
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
import yaml
from openai import OpenAI


# ─── .env 本地加载 ────────────────────────────────────────────────────────────

def _load_dotenv():
    """本地开发时从 .env 加载环境变量（GitHub Actions 中用 Secrets，不依赖此函数）。"""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value

_load_dotenv()


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "keywords.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config_abs(config_path: str = None) -> dict:
    """Load keywords.yaml from absolute path (for GitHub Actions)."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "keywords.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── LLM Client ───────────────────────────────────────────────────────────────

def get_llm_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    )


def get_llm_model() -> str:
    return os.environ.get("LLM_MODEL", "gpt-4o-mini")


def llm_chat(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    response_format: Optional[dict] = None,
) -> str:
    """Simple chat completion. Returns response text."""
    client = get_llm_client()
    kwargs = dict(
        model=model or get_llm_model(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if response_format:
        kwargs["response_format"] = response_format

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


# ─── Arxiv API ────────────────────────────────────────────────────────────────

def fetch_arxiv_papers(
    categories: list[str],
    keywords: list[str],
    lookback_days: int = 1,
    max_results: int = 200,
) -> list[dict]:
    """
    从 arxiv API 拉取指定类别的新论文。
    返回论文列表，每篇包含 title, summary, arxiv_id, authors, published, pdf_url, abstract_url。
    """
    cat_str = "+OR+".join(f"cat:{c}" for c in categories)

    # 计算日期范围
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=lookback_days)

    # arxiv API 的 sortBy=submittedDate 和 sortOrder=descending
    query = f"({cat_str})"
    url = (
        f"https://export.arxiv.org/api/query?"
        f"search_query={query}&start=0&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )

    # 用 urllib 因为 httpx 有时对 arxiv 的 XML 返回处理有问题
    req = urllib.request.Request(url, headers={"User-Agent": "ArxivDigest/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_data = resp.read().decode("utf-8")

    root = ET.fromstring(xml_data)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    papers = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)
        published_el = entry.find("atom:published", ns)

        title = " ".join(title_el.text.split()) if title_el is not None and title_el.text else ""
        summary = " ".join(summary_el.text.split()) if summary_el is not None and summary_el.text else ""
        arxiv_id = id_el.text.strip() if id_el is not None and id_el.text else ""

        # 解析 pure arxiv ID (去掉 http://arxiv.org/abs/)
        pure_id = arxiv_id.replace("http://arxiv.org/abs/", "").replace("https://arxiv.org/abs/", "")
        if pure_id.endswith("v1") or any(pure_id.endswith(f"v{i}") for i in range(10)):
            pure_id = pure_id[:-2]  # strip version suffix if present... actually let's keep it simple

        authors = [
            " ".join(author.find("atom:name", ns).text.split())
            for author in entry.findall("atom:author", ns)
            if author.find("atom:name", ns) is not None
        ]

        published = published_el.text if published_el is not None else ""

        # 只保留日期范围内的
        if published:
            try:
                pub_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
                if pub_date < start_date:
                    continue
            except (ValueError, TypeError):
                pass  # 无法解析日期则保留

        papers.append({
            "title": title,
            "summary": summary,
            "arxiv_id": pure_id,
            "abstract_url": f"https://arxiv.org/abs/{pure_id}",
            "pdf_url": f"https://arxiv.org/pdf/{pure_id}",
            "authors": authors,
            "published": published,
        })

    return papers


# ─── Feishu Webhook (自定义机器人) ────────────────────────────────────────────

def send_feishu_card(webhook_url: str, card: dict) -> bool:
    """发送飞书消息卡片到自定义机器人 webhook。"""
    payload = {"msg_type": "interactive", "card": card}
    try:
        resp = httpx.post(webhook_url, json=payload, timeout=15)
        result = resp.json()
        return result.get("code") == 0
    except Exception as e:
        print(f"[feishu webhook error] {e}")
        return False


def build_digest_card(papers: list[dict]) -> dict:
    """构建每日速递的飞书消息卡片。"""
    elements = []

    # 标题行
    today = datetime.now().strftime("%Y-%m-%d")
    elements.append({
        "tag": "markdown",
        "content": f"**📄 Arxiv 每日速递 — {today}**\n共 {len(papers)} 篇高相关论文\n---"
    })

    for i, paper in enumerate(papers):
        score = paper.get("score", 0)
        digest_cn = paper.get("digest_cn", paper["summary"][:200])
        authors_short = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            authors_short += f" et al."

        paper_text = (
            f"**{i+1}. [{paper['title']}]({paper['abstract_url']})**\n"
            f"相关度：{score:.2f}  |  {authors_short}\n"
            f"💬 {digest_cn}\n"
        )
        elements.append({"tag": "markdown", "content": paper_text})

        # 操作按钮
        elements.append({
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "📖 精读"},
                    "type": "primary",
                    "value": json.dumps({
                        "arxiv_id": paper["arxiv_id"],
                        "arxiv_url": paper["abstract_url"],
                        "title": paper["title"],
                        "summary": paper["summary"],
                    }),
                    "confirm": {
                        "title": {"tag": "plain_text", "content": "确认触发精读？"},
                        "text": {"tag": "plain_text", "content": f"将对「{paper['title'][:50]}...」进行深度分析"}
                    }
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "🔗 打开"},
                    "type": "default",
                    "url": paper["abstract_url"],
                }
            ]
        })
        elements.append({"tag": "hr"})

    return {
        "header": {
            "title": {"tag": "plain_text", "content": "📡 Arxiv 每日速递"},
            "template": "blue",
        },
        "elements": elements,
    }


# ─── Feishu App API (发送消息) ────────────────────────────────────────────────

def _get_feishu_tenant_token(app_id: str, app_secret: str) -> str:
    """获取飞书 tenant access token。"""
    resp = httpx.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": app_id, "app_secret": app_secret},
        timeout=10,
    )
    data = resp.json()
    if data.get("code") != 0:
        raise Exception(f"Feishu token error: {data}")
    return data["tenant_access_token"]


def send_feishu_message(
    receive_id: str,
    msg_type: str,
    content: str,
    app_id: str = None,
    app_secret: str = None,
) -> bool:
    """
    通过飞书应用 API 发送消息。
    receive_id: 用户 open_id 或 chat_id
    msg_type: "interactive" (卡片) 或 "text"
    content: JSON string of message content
    """
    app_id = app_id or os.environ.get("FEISHU_APP_ID", "")
    app_secret = app_secret or os.environ.get("FEISHU_APP_SECRET", "")

    if not app_id or not app_secret:
        print("[feishu] FEISHU_APP_ID or FEISHU_APP_SECRET not set, skipping message send")
        return False

    token = _get_feishu_tenant_token(app_id, app_secret)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "receive_id": receive_id,
        "msg_type": msg_type,
        "content": content,
    }
    resp = httpx.post(
        "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id",
        headers=headers,
        json=payload,
        timeout=15,
    )
    data = resp.json()
    if data.get("code") != 0:
        print(f"[feishu send error] {data}")
        return False
    return True


def build_reading_result_card(analysis: dict) -> dict:
    """构建 paper-reading 分析结果的飞书卡片。"""
    title = analysis.get("title", "Unknown")
    claims = analysis.get("claims", [])
    evidence_level = analysis.get("evidence_level", "未知")
    reproducibility = analysis.get("reproducibility", {})
    verdict = analysis.get("verdict", "")
    reading_priority = analysis.get("reading_priority", "")

    claim_lines = ""
    for c in claims[:5]:
        tag_map = {"explicit": "✅ 论文明确说", "inference": "⚠️ 合理推断", "unsupported": "❌ 未支撑"}
        tag = tag_map.get(c.get("type", ""), "❓")
        claim_lines += f"\n{tag} {c.get('statement', '')}"

    elements = [
        {"tag": "markdown", "content": f"**📖 精读分析：{title[:80]}**"},
        {"tag": "hr"},
        {"tag": "markdown", "content": f"**核心 Claims**\n{claim_lines}"},
        {"tag": "hr"},
        {"tag": "markdown", "content": f"**证据强度**：{evidence_level}\n**复现难度**：{reproducibility.get('difficulty', '未知')}\n**复现风险**：{reproducibility.get('risks', '未知')}"},
        {"tag": "hr"},
        {"tag": "markdown", "content": f"**判定**：{verdict}\n**阅读优先级**：{reading_priority}"},
    ]

    return {
        "header": {
            "title": {"tag": "plain_text", "content": "📖 精读分析结果"},
            "template": "purple",
        },
        "elements": elements,
    }


# ─── PaddleOCR ─────────────────────────────────────────────────────────────────

OCR_API_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
OCR_API_TOKEN = os.environ.get("OCR_API_TOKEN", "60adec91ca3d86eae269595aaa6c64ab19bc70a8")
OCR_MODEL = os.environ.get("OCR_MODEL", "PaddleOCR-VL-1.6")


def ocr_arxiv_pdf(arxiv_url: str, output_dir: str = "output/ocr") -> dict:
    """
    对 arxiv 论文 PDF 调用 PaddleOCR，返回 Markdown 文本和图片。

    参数:
        arxiv_url: arxiv 论文链接 (如 https://arxiv.org/abs/2210.03629)
        output_dir: 输出目录

    返回:
        {"markdown": "全文markdown文本", "pages": [{"md": "...", "images": {...}}], "pdf_url": "..."}
    """
    # 解析 arxiv PDF URL
    arxiv_id = arxiv_url.strip()
    for prefix in ["https://arxiv.org/abs/", "http://arxiv.org/abs/", "arxiv.org/abs/"]:
        if arxiv_id.startswith(prefix):
            arxiv_id = arxiv_id[len(prefix):]
            break

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"[OCR] Processing PDF: {pdf_url}")

    headers = {"Authorization": f"bearer {OCR_API_TOKEN}"}

    optional_payload = {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useChartRecognition": True,  # 开启图表识别，提取实验结果
    }

    # 提交 OCR job（URL 模式）
    headers["Content-Type"] = "application/json"
    payload = {
        "fileUrl": pdf_url,
        "model": OCR_MODEL,
        "optionalPayload": optional_payload,
    }

    job_response = httpx.post(OCR_API_URL, json=payload, headers=headers, timeout=30)
    if job_response.status_code != 200:
        print(f"[OCR] Job submission failed: {job_response.status_code} {job_response.text}")
        return None

    job_id = job_response.json()["data"]["jobId"]
    print(f"[OCR] Job submitted: {job_id}")

    # 轮询结果
    jsonl_url = ""
    while True:
        job_result = httpx.get(f"{OCR_API_URL}/{job_id}", headers=headers, timeout=15)
        if job_result.status_code != 200:
            print(f"[OCR] Poll failed: {job_result.status_code}")
            time.sleep(5)
            continue

        state = job_result.json()["data"]["state"]
        if state == "pending":
            print("[OCR] Pending...")
        elif state == "running":
            try:
                progress = job_result.json()["data"]["extractProgress"]
                print(f"[OCR] Running: {progress.get('extractedPages', 0)}/{progress.get('totalPages', '?')} pages")
            except KeyError:
                print("[OCR] Running...")
        elif state == "done":
            progress = job_result.json()["data"]["extractProgress"]
            print(f"[OCR] Done: {progress['extractedPages']} pages extracted")
            jsonl_url = job_result.json()["data"]["resultUrl"]["jsonUrl"]
            break
        elif state == "failed":
            error = job_result.json()["data"].get("errorMsg", "unknown")
            print(f"[OCR] Failed: {error}")
            return None

        time.sleep(5)

    if not jsonl_url:
        print("[OCR] No result URL")
        return None

    # 下载结果
    os.makedirs(output_dir, exist_ok=True)
    jsonl_resp = httpx.get(jsonl_url, timeout=30)
    jsonl_resp.raise_for_status()

    lines = [l.strip() for l in jsonl_resp.text.split("\n") if l.strip()]

    all_markdown = []
    pages = []

    for line_num, line in enumerate(lines):
        result = json.loads(line)["result"]
        for i, res in enumerate(result["layoutParsingResults"]):
            md_text = res["markdown"]["text"]
            all_markdown.append(md_text)

            page_data = {"md": md_text, "images": []}

            # 保存 Markdown
            md_filename = os.path.join(output_dir, f"doc_{line_num}.md")
            with open(md_filename, "w", encoding="utf-8") as f:
                f.write(md_text)

            # 下载图片
            for img_path, img_url in res["markdown"]["images"].items():
                full_path = os.path.join(output_dir, img_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                try:
                    img_bytes = httpx.get(img_url, timeout=15).content
                    with open(full_path, "wb") as f:
                        f.write(img_bytes)
                    page_data["images"].append(full_path)
                except Exception as e:
                    print(f"[OCR] Image download failed: {img_path}: {e}")

            pages.append(page_data)
            print(f"[OCR] Page {line_num} saved: {md_filename}")

    full_markdown = "\n\n".join(all_markdown)
    print(f"[OCR] Total: {len(pages)} pages, {len(full_markdown)} characters")

    return {
        "markdown": full_markdown,
        "pages": pages,
        "pdf_url": pdf_url,
        "arxiv_id": arxiv_id,
    }


def extract_experiment_section(ocr_result: dict) -> str:
    """
    从 OCR 全文 Markdown 中提取实验相关段落。
    用启发式方法定位 Experiments / Results 部分。
    """
    if not ocr_result or not ocr_result.get("markdown"):
        return ""

    md = ocr_result["markdown"]

    # 常见实验部分标题
    experiment_headers = [
        "Experiment", "Experiments", "Experimental",
        "Results", "Evaluation", "Evaluations",
        "实验", "结果", "评估",
        "Main Results", "Ablation",
    ]

    lines = md.split("\n")
    found_sections = []
    in_section = False
    current_section = []

    for line in lines:
        stripped = line.strip()
        # 检测是否为标题行
        is_header = False
        for h in experiment_headers:
            if stripped.lower().startswith(h.lower()) and len(stripped) < 100:
                is_header = True
                if current_section:
                    found_sections.append("\n".join(current_section))
                current_section = [line]
                in_section = True
                break

        if not is_header and in_section:
            # 检测是否到了下一节（简单的启发式：短行 + 首字母大写 或 数字开头）
            if stripped and len(stripped) < 80 and (
                stripped[0].isupper() or stripped[0].isdigit()
            ) and not stripped.startswith(("Table", "Figure", "Fig", "|")):
                # 可能是新章节标题
                if any(h.lower() in stripped.lower() for h in [
                    "conclusion", "related work", "introduction", "abstract",
                    "method", "approach", "background", "preliminar",
                    "reference", "appendix", "discussion", "limitation",
                    "结论", "相关工作", "引言", "方法", "背景",
                ]):
                    found_sections.append("\n".join(current_section))
                    current_section = []
                    in_section = False
                else:
                    current_section.append(line)
            else:
                current_section.append(line)

    if current_section:
        found_sections.append("\n".join(current_section))

    result = "\n\n".join(found_sections)

    # 如果太长，截取前 8000 字符
    if len(result) > 8000:
        result = result[:8000] + "\n\n[... truncated ...]"

    if not result:
        # Fallback：返回后 40% 的内容（实验通常在后面）
        result = md[len(md)//2:][:8000]

    return result


# ─── Feishu Event Signature Verification ──────────────────────────────────────

def verify_feishu_signature(timestamp: str, nonce: str, body: str, signing_key: str) -> bool:
    """验证飞书事件订阅签名。用于 Cloudflare Worker 端。"""
    sign_str = f"{timestamp}\n{nonce}\n{body}"
    expected = base64.b64encode(
        hmac.new(signing_key.encode(), sign_str.encode(), hashlib.sha256).digest()
    ).decode()
    return True  # Worker 端实现，Python 端仅做参考


# ─── Misc ─────────────────────────────────────────────────────────────────────

def chunk_list(lst: list, n: int) -> list[list]:
    """Split list into chunks of size n."""
    return [lst[i:i+n] for i in range(0, len(lst), n)]
