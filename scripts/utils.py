"""
共享工具模块：LLM 客户端、飞书 API、arxiv 辅助函数、MinerU OCR。
"""

import os
import sys
import io
import json
import time
import re
import zipfile
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
        issue_body = f"arxiv: {paper['abstract_url']}\n\n> {paper['digest_cn']}\n\n---\n点击 Submit 触发分析"
        reading_issue_url = (
            f"https://github.com/Estrellajer/arxiv-digest/issues/new"
            f"?title={urllib.parse.quote('[精读] ' + issue_title)}"
            f"&body={urllib.parse.quote(issue_body)}"
        )
        setup_issue_url = (
            f"https://github.com/Estrellajer/arxiv-digest/issues/new"
            f"?title={urllib.parse.quote('[实验配置] ' + issue_title)}"
            f"&body={urllib.parse.quote(issue_body)}"
        )

        elements.append({
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "📖 精读"},
                    "type": "primary",
                    "url": reading_issue_url,
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "🔬 实验配置"},
                    "type": "default",
                    "url": setup_issue_url,
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


def build_setup_result_card(setup: dict) -> dict:
    """构建 experiment-setup 实验配置抽取结果的飞书卡片。"""
    title = setup.get("title", "Unknown")

    datasets = "\n".join(
        f"- {d.get('name', '未知')}｜划分：{d.get('split', '未知')}｜规模：{d.get('size', '未知')}"
        for d in (setup.get("datasets") or [])[:5]
    ) or "未知"

    model = setup.get("model", {}) or {}
    model_text = (
        f"{model.get('name', '未知')}｜架构：{model.get('architecture', '未知')}"
        f"｜参数量：{model.get('params', '未知')}｜权重：{model.get('init_weights', '未知')}"
    )

    hp = setup.get("hyperparameters", {}) or {}
    hp_text = (
        f"lr：{hp.get('learning_rate', '未知')}｜batch：{hp.get('batch_size', '未知')}"
        f"｜轮数/步数：{hp.get('epochs_or_steps', '未知')}｜优化器：{hp.get('optimizer', '未知')}"
        f"｜调度：{hp.get('scheduler', '未知')}｜其它：{hp.get('other', '未知')}"
    )

    hw = setup.get("hardware", {}) or {}
    hw_text = (
        f"{hw.get('accelerator', '未知')} × {hw.get('count', '未知')}"
        f"｜时长：{hw.get('training_time', '未知')}｜成本：{hw.get('cost', '未知')}"
    )

    ev = setup.get("evaluation", {}) or {}
    metrics = "、".join(ev.get("metrics", []) or []) or "未知"
    eval_text = (
        f"指标：{metrics}｜协议：{ev.get('protocol', '未知')}"
        f"｜few-shot：{ev.get('few_shot', '未知')}｜温度：{ev.get('temperature', '未知')}"
        f"｜解码：{ev.get('decoding', '未知')}｜种子：{ev.get('seeds', '未知')}"
    )

    ablations = "\n".join(
        f"- {a.get('setting', '未知')} → {a.get('finding', '未知')}"
        for a in (setup.get("ablations") or [])[:4]
    ) or "未知"

    repro = setup.get("reproducibility", {}) or {}
    code_flag = {"true": "✅", "false": "❌", "unknown": "❓"}.get(str(repro.get("code_available", "unknown")), "❓")
    code_url = repro.get("code_url", "未知")
    key_points = "；".join(repro.get("key_to_reproduce", []) or []) or "未知"
    missing = "；".join(repro.get("missing_details", []) or []) or "未知"
    repro_text = (
        f"代码：{code_flag} {code_url}｜数据：{repro.get('data_available', 'unknown')}"
        f"｜权重：{repro.get('checkpoints', '未知')}\n"
        f"**复现要点**：{key_points}\n**缺口**：{missing}"
    )

    elements = [
        {"tag": "markdown", "content": f"**🔬 实验配置 — {title[:80]}**\n输入：{setup.get('input_coverage', '未知')}"},
        {"tag": "hr"},
        {"tag": "markdown", "content": f"**数据集**\n{datasets}"},
        {"tag": "hr"},
        {"tag": "markdown", "content": f"**模型**：{model_text}\n**超参**：{hp_text}\n**硬件**：{hw_text}"},
        {"tag": "hr"},
        {"tag": "markdown", "content": f"**评测**：{eval_text}"},
        {"tag": "hr"},
        {"tag": "markdown", "content": f"**消融**\n{ablations}"},
        {"tag": "hr"},
        {"tag": "markdown", "content": f"**复现性**\n{repro_text}"},
    ]

    return {
        "header": {
            "title": {"tag": "plain_text", "content": "🔬 实验配置抽取"},
            "template": "turquoise",
        },
        "elements": elements,
    }


# ─── MinerU OCR ─────────────────────────────────────────────────────────────

# MinerU 提供两条解析路径，都把 arXiv PDF 链接交给服务端远程下载并解析，本地 / CI 不下载 PDF：
#   1) 精准解析 API（v4，需 Token）：vlm 模型，质量更高，每账号每天 1000 页高优先级额度。
#   2) Agent 轻量解析 API（v1，免登录、IP 限频）：作为兜底，无 Token 时也能用。
# 配置了 MINERU_TOKEN 则优先精准解析，失败自动回退 Agent 轻量解析。
MINERU_TOKEN = os.environ.get("MINERU_TOKEN", "")
MINERU_MODEL_VERSION = os.environ.get("MINERU_MODEL_VERSION", "vlm")
MINERU_PRECISION_URL = os.environ.get("MINERU_PRECISION_URL", "https://mineru.net/api/v4/extract/task")
MINERU_AGENT_URL = os.environ.get("MINERU_AGENT_URL", "https://mineru.net/api/v1/agent/parse/url")
MINERU_AGENT_QUERY_URL = os.environ.get("MINERU_AGENT_QUERY_URL", "https://mineru.net/api/v1/agent/parse")
# 只解析前 N 页：论文正文通常 <=20 页，之后多为补充材料；也正好是轻量 API 的页数上限。
OCR_PAGE_LIMIT = int(os.environ.get("OCR_PAGE_LIMIT", "20"))
OCR_LANGUAGE = os.environ.get("OCR_LANGUAGE", "en")
OCR_TIMEOUT_SECONDS = int(os.environ.get("OCR_TIMEOUT_SECONDS", "600"))


def _normalize_arxiv_pdf_url(arxiv_url: str) -> tuple[str, str]:
    """接受 abs / pdf / 纯 ID 输入，统一归一化为 (arxiv_id, pdf_url)。"""
    arxiv_id = arxiv_url.strip()
    for prefix in (
        "https://arxiv.org/abs/", "http://arxiv.org/abs/", "arxiv.org/abs/",
        "https://arxiv.org/pdf/", "http://arxiv.org/pdf/", "arxiv.org/pdf/",
    ):
        if arxiv_id.startswith(prefix):
            arxiv_id = arxiv_id[len(prefix):]
            break
    if arxiv_id.lower().endswith(".pdf"):
        arxiv_id = arxiv_id[:-4]
    arxiv_id = arxiv_id.rstrip("/")
    return arxiv_id, f"https://arxiv.org/pdf/{arxiv_id}"


def _markdown_from_zip(zip_url: str) -> Optional[str]:
    """从精准解析结果 ZIP（CDN 链接）中取出 full.md 文本。"""
    resp = httpx.get(zip_url, timeout=120, follow_redirects=True)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        md_name = next((n for n in zf.namelist() if n.endswith("full.md")), None)
        if not md_name:
            print("[OCR] full.md not found in result zip")
            return None
        return zf.read(md_name).decode("utf-8", errors="replace")


def _mineru_precision_parse(pdf_url: str, arxiv_id: str) -> Optional[str]:
    """精准解析 API（v4，需 Token）。返回 Markdown 文本，失败返回 None 以便回退。"""
    if not MINERU_TOKEN:
        return None

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {MINERU_TOKEN}"}
    payload = {
        "url": pdf_url,
        "model_version": MINERU_MODEL_VERSION,
        "page_ranges": f"1-{OCR_PAGE_LIMIT}",
        "is_ocr": False,
        "enable_formula": False,
        "enable_table": True,
        "language": OCR_LANGUAGE,
        "data_id": arxiv_id.replace("/", "_"),
    }
    submit = httpx.post(MINERU_PRECISION_URL, json=payload, headers=headers, timeout=60)
    if submit.status_code != 200:
        print(f"[OCR] Precision submit failed: {submit.status_code} {submit.text[:200]}")
        return None
    body = submit.json()
    if body.get("code") != 0:
        print(f"[OCR] Precision rejected: code={body.get('code')} msg={body.get('msg')}")
        return None
    task_id = (body.get("data") or {}).get("task_id")
    if not task_id:
        print("[OCR] Precision response missing task_id")
        return None
    print(f"[OCR] Precision task submitted: {task_id}")

    zip_url = ""
    deadline = time.monotonic() + OCR_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        poll = httpx.get(f"{MINERU_PRECISION_URL}/{task_id}", headers=headers, timeout=30)
        if poll.status_code != 200:
            print(f"[OCR] Precision poll failed: {poll.status_code}")
            time.sleep(5)
            continue
        data = poll.json().get("data") or {}
        state = data.get("state", "")
        if state == "done":
            zip_url = data.get("full_zip_url") or ""
            print("[OCR] Precision done")
            break
        if state == "failed":
            print(f"[OCR] Precision failed: {data.get('err_msg', 'unknown')}")
            return None
        prog = data.get("extract_progress", {}) or {}
        print(f"[OCR] Precision {state or 'pending'}: {prog.get('extracted_pages', '?')}/{prog.get('total_pages', '?')}")
        time.sleep(5)

    if not zip_url:
        print(f"[OCR] Precision did not finish within {OCR_TIMEOUT_SECONDS} seconds")
        return None
    return _markdown_from_zip(zip_url)


def _mineru_agent_parse(pdf_url: str, arxiv_id: str) -> Optional[str]:
    """Agent 轻量解析 API（v1，免登录）。返回 Markdown 文本，失败返回 None。"""
    payload = {
        "url": pdf_url,
        "file_name": f"{arxiv_id.replace('/', '_')}.pdf",
        "language": OCR_LANGUAGE,
        "page_range": f"1-{OCR_PAGE_LIMIT}",
        "enable_table": True,
        "is_ocr": False,
        "enable_formula": False,
    }
    submit = httpx.post(MINERU_AGENT_URL, json=payload, timeout=60)
    if submit.status_code == 429:
        print("[OCR] Agent rate limited (HTTP 429)")
        return None
    if submit.status_code != 200:
        print(f"[OCR] Agent submit failed: {submit.status_code} {submit.text[:200]}")
        return None
    body = submit.json()
    if body.get("code") != 0:
        print(f"[OCR] Agent rejected: code={body.get('code')} msg={body.get('msg')}")
        return None
    task_id = (body.get("data") or {}).get("task_id")
    if not task_id:
        print("[OCR] Agent response missing task_id")
        return None
    print(f"[OCR] Agent task submitted: {task_id}")

    markdown_url = ""
    deadline = time.monotonic() + OCR_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        poll = httpx.get(f"{MINERU_AGENT_QUERY_URL}/{task_id}", timeout=30)
        if poll.status_code != 200:
            print(f"[OCR] Agent poll failed: {poll.status_code}")
            time.sleep(5)
            continue
        data = poll.json().get("data") or {}
        state = data.get("state", "")
        if state == "done":
            markdown_url = data.get("markdown_url") or data.get("markdownUrl") or ""
            print("[OCR] Agent done")
            break
        if state == "failed":
            print(f"[OCR] Agent failed: {data.get('err_msg') or data.get('error_msg', 'unknown')}")
            return None
        print(f"[OCR] Agent {state or 'pending'}...")
        time.sleep(5)

    if not markdown_url:
        print(f"[OCR] Agent did not finish within {OCR_TIMEOUT_SECONDS} seconds")
        return None

    md_resp = httpx.get(markdown_url, timeout=60)
    md_resp.raise_for_status()
    return md_resp.text


def ocr_arxiv_pdf(arxiv_url: str, output_dir: str = "output/ocr", download_images: bool = False) -> dict:
    """
    通过 MinerU 解析 arxiv 论文 PDF，返回 Markdown 文本。

    设计要点：
    - 直接把 arXiv PDF 链接交给 MinerU，由服务端远程下载并解析，本地 / CI 不下载 PDF。
    - 仅解析前 OCR_PAGE_LIMIT 页（默认 20）。
    - 优先精准解析 API（需 MINERU_TOKEN，vlm 模型），失败自动回退 Agent 轻量解析 API（免登录）。

    参数:
        arxiv_url: arxiv 链接或 ID（abs / pdf / 纯 ID 均可）
        output_dir: 结果 Markdown 的输出目录
        download_images: 兼容旧签名；当前实现只取文本，此参数不再使用

    返回:
        {"markdown": "全文markdown文本", "pages": [{"md": "...", "images": []}], "pdf_url": "...", "arxiv_id": "...", "ocr_source": "precision|agent"}
    """
    arxiv_id, pdf_url = _normalize_arxiv_pdf_url(arxiv_url)

    markdown = None
    source = ""
    if MINERU_TOKEN:
        print(f"[OCR] MinerU precision parsing: {pdf_url} (pages 1-{OCR_PAGE_LIMIT}, model={MINERU_MODEL_VERSION})")
        markdown = _mineru_precision_parse(pdf_url, arxiv_id)
        if markdown is not None:
            source = "precision"
        else:
            print("[OCR] Precision parse failed; falling back to Agent lightweight API")

    if markdown is None:
        print(f"[OCR] MinerU agent parsing: {pdf_url} (pages 1-{OCR_PAGE_LIMIT})")
        markdown = _mineru_agent_parse(pdf_url, arxiv_id)
        if markdown is not None:
            source = "agent"

    if markdown is None:
        print("[OCR] All MinerU parse paths failed")
        return None

    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, "doc.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"[OCR] Saved markdown via {source}: {md_path} ({len(markdown)} characters)")

    # MinerU 返回整篇 Markdown，不分页；以单页形式回传以兼容下游 build_ocr_evidence。
    return {
        "markdown": markdown,
        "pages": [{"md": markdown, "images": []}],
        "pdf_url": pdf_url,
        "arxiv_id": arxiv_id,
        "ocr_source": source,
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
