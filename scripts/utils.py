"""
共享工具模块：LLM 客户端、飞书 API、arxiv 辅助函数、PaddleOCR。
"""

import os
import sys
import json
import time
import re
import urllib.request
import urllib.parse
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
        content_score = paper.get("content_score", paper.get("score", 0))
        bonus = paper.get("institution_bonus", 0)
        final_score = paper.get("final_score", paper.get("score", 0))
        decision = paper.get("decision", {})
        recognized = [i["name"] for i in paper.get("recognized_institutions", [])]
        institutions = ", ".join(recognized or decision.get("affiliations", [])) or "未识别"
        code_url = paper.get("code_url")
        code_text = f"[代码/项目]({code_url})" if code_url else "未发现论文明确代码链接"
        coverage = "OCR 首页+实验" if paper.get("ocr_status") == "success" else "仅摘要"

        paper_text = (
            f"**{i+1}. [{paper['title']}]({paper['abstract_url']})**\n"
            f"🏢 **机构**：{institutions} ｜ {code_text}\n"
            f"❓ **问题**：{decision.get('research_question', '未知')}\n"
            f"🧠 **方法**：{decision.get('core_method', '未知')}\n"
            f"📊 **证据**：{decision.get('key_experiment', '未知')}\n"
            f"✅ **推荐**：{decision.get('recommendation', paper.get('score_reason', '未知'))}\n"
            f"⚠️ **风险**：{decision.get('risk', '未知')} ｜ 输入：{coverage}\n"
            f"**评分**：内容 {content_score:.2f} + 机构 {bonus:.2f} = {final_score:.2f}\n"
        )
        elements.append({"tag": "markdown", "content": paper_text})

        # 操作按钮
        issue_title = paper["title"][:80]
        issue_body = f"arxiv: {paper['abstract_url']}\n\n> {paper['digest_cn']}\n\n---\n点击 Submit 触发精读分析"
        issue_url = (
            f"https://github.com/Estrellajer/arxiv-digest/issues/new"
            f"?title={urllib.parse.quote('[精读] ' + issue_title)}"
            f"&body={urllib.parse.quote(issue_body)}"
        )

        elements.append({
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "📖 精读"},
                    "type": "primary",
                    "url": issue_url,
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
    receive_id: 用户 open_id 或群 chat_id
    msg_type: "interactive" (卡片) 或 "text"
    content: JSON string of message content
    """
    app_id = app_id or os.environ.get("FEISHU_APP_ID", "")
    app_secret = app_secret or os.environ.get("FEISHU_APP_SECRET", "")

    if not app_id or not app_secret:
        print("[feishu] FEISHU_APP_ID or FEISHU_APP_SECRET not set, skipping message send")
        return False

    receive_id_type = os.environ.get("FEISHU_RECEIVE_ID_TYPE") or "chat_id"
    if receive_id_type not in {"open_id", "user_id", "union_id", "email", "chat_id"}:
        print(f"[feishu] Unsupported FEISHU_RECEIVE_ID_TYPE: {receive_id_type}")
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
        f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type={receive_id_type}",
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
    input_coverage = analysis.get("input_coverage", "仅摘要")
    reading_priority = analysis.get("reading_priority", "")
    steps = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(analysis.get("method_steps", [])[:5])) or "未知"
    experiments = "\n".join(
        f"- {item.get('result', '未知')}（{item.get('meaning', '未知')}）"
        for item in analysis.get("key_experiments", [])[:4]
    ) or "未知"
    limitations = "；".join(analysis.get("limitations", [])[:3]) or "未知"
    guide = "；".join(analysis.get("reading_guide", [])[:3]) or "未知"

    elements = [
        {"tag": "markdown", "content": f"**📖 {title[:80]}**\n{analysis.get('quick_take', '未知')}"},
        {"tag": "hr"},
        {"tag": "markdown", "content": f"**研究问题**：{analysis.get('research_question', '未知')}\n**核心直觉**：{analysis.get('core_intuition', '未知')}"},
        {"tag": "hr"},
        {"tag": "markdown", "content": f"**方法步骤**\n{steps}"},
        {"tag": "hr"},
        {"tag": "markdown", "content": f"**关键实验**\n{experiments}"},
        {"tag": "hr"},
        {"tag": "markdown", "content": f"**局限**：{limitations}\n**阅读指南**：{guide}\n**代码**：{analysis.get('code_url', '未知')}\n**分析输入**：{input_coverage}\n**优先级**：{reading_priority}｜{analysis.get('priority_reason', '')}"},
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
OCR_API_TOKEN = os.environ.get("OCR_API_TOKEN", "")
OCR_MODEL = os.environ.get("OCR_MODEL", "PaddleOCR-VL-1.6")
OCR_TIMEOUT_SECONDS = int(os.environ.get("OCR_TIMEOUT_SECONDS", "600"))


def ocr_arxiv_pdf(arxiv_url: str, output_dir: str = "output/ocr", download_images: bool = False) -> dict:
    """
    对 arxiv 论文 PDF 调用 PaddleOCR，返回 Markdown 文本和图片。

    参数:
        arxiv_url: arxiv 论文链接 (如 https://arxiv.org/abs/2210.03629)
        output_dir: 输出目录

    返回:
        {"markdown": "全文markdown文本", "pages": [{"md": "...", "images": {...}}], "pdf_url": "..."}
    """
    if not OCR_API_TOKEN:
        raise RuntimeError("OCR_API_TOKEN is not configured")

    # 解析 arxiv PDF URL
    arxiv_id = arxiv_url.strip()
    for prefix in ["https://arxiv.org/abs/", "http://arxiv.org/abs/", "arxiv.org/abs/"]:
        if arxiv_id.startswith(prefix):
            arxiv_id = arxiv_id[len(prefix):]
            break

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    print(f"[OCR] Processing PDF: {pdf_url}")

    headers = {"Authorization": f"bearer {OCR_API_TOKEN}"}

    optional_payload = {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useChartRecognition": True,  # 开启图表识别，提取实验结果
    }

    # PaddleOCR 对部分论文站点的远程 URL 抓取会得到非 PDF 内容，因此先下载校验再上传。
    pdf_response = httpx.get(
        pdf_url,
        headers={"User-Agent": "ArxivDigest/1.0"},
        follow_redirects=True,
        timeout=60,
    )
    pdf_response.raise_for_status()
    pdf_bytes = pdf_response.content
    if not pdf_bytes.startswith(b"%PDF"):
        content_type = pdf_response.headers.get("content-type", "unknown")
        raise ValueError(f"arXiv did not return a PDF (content-type: {content_type})")

    form = {
        "model": OCR_MODEL,
        "optionalPayload": json.dumps(optional_payload),
    }
    files = {"file": (f"{arxiv_id.replace('/', '_')}.pdf", pdf_bytes, "application/pdf")}
    job_response = httpx.post(OCR_API_URL, data=form, files=files, headers=headers, timeout=90)
    if job_response.status_code != 200:
        print(f"[OCR] Job submission failed: {job_response.status_code} {job_response.text}")
        return None

    job_id = job_response.json()["data"]["jobId"]
    print(f"[OCR] Job submitted: {job_id}")

    # 轮询结果
    jsonl_url = ""
    deadline = time.monotonic() + OCR_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
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
        raise TimeoutError(f"OCR job did not finish within {OCR_TIMEOUT_SECONDS} seconds")

    # 下载结果
    os.makedirs(output_dir, exist_ok=True)
    jsonl_resp = httpx.get(jsonl_url, timeout=30)
    jsonl_resp.raise_for_status()

    lines = [l.strip() for l in jsonl_resp.text.split("\n") if l.strip()]

    all_markdown = []
    pages = []

    page_index = 0
    for line in lines:
        result = json.loads(line)["result"]
        for res in result["layoutParsingResults"]:
            md_text = res["markdown"]["text"]
            all_markdown.append(md_text)

            page_data = {"md": md_text, "images": []}

            # 保存 Markdown
            md_filename = os.path.join(output_dir, f"doc_{page_index}.md")
            with open(md_filename, "w", encoding="utf-8") as f:
                f.write(md_text)

            if download_images:
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
            print(f"[OCR] Page {page_index} saved: {md_filename}")
            page_index += 1

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

    def heading_title(line: str) -> str:
        stripped = line.strip()
        if len(stripped) > 120:
            return ""
        is_markdown = stripped.startswith("#")
        is_numbered = bool(re.match(r"^\d+(?:\.\d+)*[.)]?\s+", stripped))
        if not (is_markdown or is_numbered):
            return ""
        title = re.sub(r"^#+\s*", "", stripped)
        title = re.sub(r"^\d+(?:\.\d+)*[.)]?\s*", "", title)
        return title.casefold()

    start_terms = ("experiment", "evaluation", "main result", "empirical", "实验", "评估", "主要结果")
    stop_terms = ("conclusion", "related work", "reference", "appendix", "discussion", "limitation", "结论", "相关工作", "参考文献", "附录", "局限")
    lines = md.splitlines()
    selected = []
    in_section = False
    for line in lines:
        title = heading_title(line)
        if not in_section and title and any(term in title for term in start_terms):
            in_section = True
        elif in_section and title and any(title.startswith(term) for term in stop_terms):
            break
        if in_section:
            selected.append(line)

    result = "\n".join(selected).strip()

    if len(result) > 12000:
        result = result[:12000] + "\n\n[... truncated ...]"

    if not result:
        # Fallback：返回后 40% 的内容（实验通常在后面）
        result = md[len(md)//2:][:12000]

    return result


# ─── Misc ─────────────────────────────────────────────────────────────────────

def chunk_list(lst: list, n: int) -> list[list]:
    """Split list into chunks of size n."""
    return [lst[i:i+n] for i in range(0, len(lst), n)]
