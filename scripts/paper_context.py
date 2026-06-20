"""OCR evidence, institution matching, and prompt loading."""

import re
from pathlib import Path

import httpx
import yaml


ROOT = Path(__file__).resolve().parent.parent
URL_RE = re.compile(r"https?://[^\s<>()\]\[\]{}\"']+")
CODE_HOSTS = ("github.com", "gitlab.com", "codeberg.org", "huggingface.co", "modelscope.cn")


def load_prompt(name: str) -> str:
    return (ROOT / "prompts" / f"{name}.md").read_text(encoding="utf-8").strip()


def load_institutions(path: str = None) -> dict:
    source = Path(path) if path else ROOT / "config" / "institutions.yaml"
    return yaml.safe_load(source.read_text(encoding="utf-8"))


def match_institutions(text: str, config: dict = None) -> list[dict]:
    """Match configured aliases against OCR text, longest aliases first."""
    config = config or load_institutions()
    haystack = f" {text.casefold()} "
    matches = []
    seen = set()
    for kind, key in (("university", "universities"), ("company", "companies")):
        for item in config.get(key, []):
            aliases = sorted({item["name"], *item.get("aliases", [])}, key=len, reverse=True)
            if any(_alias_present(haystack, alias) for alias in aliases):
                if item["name"] not in seen:
                    matches.append({"name": item["name"], "type": kind, "country": item.get("country", "")})
                    seen.add(item["name"])
    return matches


def extract_affiliation_region(first_page: str) -> str:
    """Keep the paper header and stop before abstract/body where model names cause false matches."""
    if not first_page:
        return ""
    match = re.search(r"(?im)^\s*#{0,3}\s*(abstract|摘要)\s*$", first_page)
    region = first_page[:match.start() if match else min(len(first_page), 4000)]
    lines = region.splitlines()
    for index, line in enumerate(lines):
        if line.strip():
            if line.lstrip().startswith("#"):
                lines[index] = ""
            break
    return "\n".join(lines)


def _alias_present(haystack: str, alias: str) -> bool:
    needle = alias.casefold().strip()
    if len(needle) <= 3 and needle.isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None
    return needle in haystack


def extract_evidence_urls(*texts: str) -> list[str]:
    urls = []
    for text in texts:
        for raw in URL_RE.findall(text or ""):
            url = raw.rstrip(".,;:!?)")
            if url.endswith(".git"):
                url = url[:-4]
            if any(host in url.casefold() for host in CODE_HOSTS) and url not in urls:
                urls.append(url)
    priority = {"github.com": 0, "gitlab.com": 1, "codeberg.org": 2, "huggingface.co": 3, "modelscope.cn": 4}
    return sorted(urls, key=lambda url: next((rank for host, rank in priority.items() if host in url.casefold()), 99))


def fetch_arxiv_evidence_urls(abstract_url: str) -> list[str]:
    try:
        response = httpx.get(abstract_url, follow_redirects=True, timeout=15)
        response.raise_for_status()
        return extract_evidence_urls(response.text)
    except Exception as exc:
        print(f"[context] arXiv link extraction failed: {exc}")
        return []


def build_ocr_evidence(ocr_result: dict, abstract_url: str = "") -> dict:
    pages = (ocr_result or {}).get("pages", [])
    first_page = pages[0].get("md", "") if pages else ""
    from utils import extract_experiment_section

    experiment = extract_experiment_section(ocr_result) if ocr_result else ""
    urls = extract_evidence_urls(first_page, (ocr_result or {}).get("markdown", ""))
    for url in fetch_arxiv_evidence_urls(abstract_url) if abstract_url else []:
        if url not in urls:
            urls.append(url)
    return {
        "first_page": first_page[:8000],
        "experiment": experiment,
        "institutions": match_institutions(extract_affiliation_region(first_page)),
        "code_urls": urls,
        "ocr_status": "success" if first_page else "empty",
    }


def apply_institution_bonus(content_score: float, institutions: list[dict], max_bonus: float = 0.08) -> tuple[float, float]:
    bonus = max_bonus if content_score >= 0.5 and institutions else 0.0
    return round(min(1.0, content_score + bonus), 4), bonus
