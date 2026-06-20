"""
paper-feishu-digest: arxiv 每日速递
抓取 → LLM 保守打分 → 中文一句话速递 → 飞书卡片推送

Usage:
    python digest.py                          # 使用 config/keywords.yaml
    python digest.py --categories cs.CL,cs.AI --keywords "agent,RAG"
    python digest.py --dry-run                # 不推送到飞书，只打印结果
    python digest.py --lookback 3             # 查最近 3 天
"""

import os
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent dir to path for GitHub Actions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_config, load_config_abs,
    get_llm_client, get_llm_model, llm_chat,
    fetch_arxiv_papers,
    send_feishu_card, build_digest_card,
    chunk_list,
)


# ─── Scoring (保守打分) ───────────────────────────────────────────────────────

SCORING_SYSTEM_PROMPT = """You are a strict research paper reviewer screening arxiv submissions for relevance to specific research interests.

Your job is to assign a relevance score (0-1) to each paper based on whether it genuinely relates to the user's keywords.

**CRITICAL RULES — conservative scoring**:
1. DEFAULT to LOW scores. Only give high scores when you are genuinely confident the paper is relevant.
2. A keyword appearing in passing or as background context does NOT make a paper relevant.
3. The paper's CORE contribution must relate to the keywords.
4. If the abstract is vague or you're unsure, score LOW (0.35 or below).
5. "Better to miss a relevant paper than to flood the user with noise."
6. Title-only matches with no abstract substance → score 0.4 max.

Output format: a JSON object with:
{
  "scores": [
    {"arxiv_id": "xxxx.xxxxx", "score": 0.85, "reason": "one line Chinese reason"},
    ...
  ]
}
"""


def score_papers(papers: list[dict], keywords: list[str], model: str = None) -> list[dict]:
    """
    用 LLM 对论文进行保守打分。
    每批处理 20 篇，并行调用。
    """
    if not model:
        model = get_llm_model()

    keyword_str = ", ".join(keywords)

    # 构建论文列表供 LLM 阅读
    def score_batch(batch: list[dict]) -> list[dict]:
        papers_text = ""
        for p in batch:
            summary_short = p["summary"][:800]  # 只取前 800 字符
            papers_text += (
                f"ID: {p['arxiv_id']}\n"
                f"Title: {p['title']}\n"
                f"Abstract: {summary_short}\n\n"
            )

        user_prompt = (
            f"Keywords: {keyword_str}\n\n"
            f"Papers to score:\n\n{papers_text}\n"
            f"Score each paper 0-1 for relevance to the keywords. Be conservative (default low)."
        )

        try:
            result = llm_chat(
                messages=[
                    {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            data = json.loads(result)
            return data.get("scores", [])
        except Exception as e:
            print(f"[scoring error] {e}, returning empty scores for batch")
            return []

    batches = chunk_list(papers, 20)
    all_scores = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(score_batch, batch): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            try:
                scores = future.result()
                for s in scores:
                    all_scores[s["arxiv_id"]] = s
            except Exception as e:
                print(f"[scoring batch error] {e}")

    # Attach scores to papers
    scored_papers = []
    for p in papers:
        score_info = all_scores.get(p["arxiv_id"], None)
        if score_info:
            p["score"] = score_info.get("score", 0.0)
            p["score_reason"] = score_info.get("reason", "")
        else:
            p["score"] = 0.0
            p["score_reason"] = ""
        scored_papers.append(p)

    return scored_papers


# ─── Chinese Digest Generation ────────────────────────────────────────────────

DIGEST_SYSTEM_PROMPT = """你是一个论文速递编辑。用**一句话中文**总结论文的核心贡献。

要求：
1. 只写一句话，不超过 80 个字
2. 只描述论文做了什么，不评价好坏
3. 用中文技术术语，确保领域内的人一看就懂
4. 不要用"本文""这篇论文"开头，直接说方法/发现
5. 如果论文提出了新方法：说方法名 + 解决什么问题
6. 如果是 benchmark/数据集：说规模和特点
7. 如果是理论分析：说主要结论
"""


def generate_digests(papers: list[dict], model: str = None, max_workers: int = 5) -> list[dict]:
    """为高分论文生成中文一句话速递。"""
    if not model:
        model = get_llm_model()

    def digest_one(paper: dict) -> dict:
        try:
            content = llm_chat(
                messages=[
                    {"role": "system", "content": DIGEST_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Title: {paper['title']}\nAbstract: {paper['summary'][:1200]}"},
                ],
                model=model,
                temperature=0.2,
                max_tokens=200,
            )
            paper["digest_cn"] = content.strip()
        except Exception as e:
            paper["digest_cn"] = paper["summary"][:100] + "..."
            print(f"[digest error for {paper['arxiv_id']}] {e}")
        return paper

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(digest_one, p): p["arxiv_id"] for p in papers}
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"[digest future error] {e}")

    # Restore order
    id_to_paper = {p["arxiv_id"]: p for p in results}
    return [id_to_paper.get(p["arxiv_id"], p) for p in papers]


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_digest(
    categories: list[str],
    keywords: list[str],
    threshold: float = 0.75,
    max_papers: int = 10,
    lookback_days: int = 1,
    dry_run: bool = False,
    webhook_url: str = None,
    scoring_model: str = None,
    digest_model: str = None,
    config_path: str = None,
):
    print(f"[digest] Fetching papers from categories: {categories}")
    print(f"[digest] Keywords: {keywords} | Threshold: {threshold} | Max: {max_papers} | Lookback: {lookback_days}d")

    # Step 1: Fetch
    papers = fetch_arxiv_papers(
        categories=categories,
        keywords=keywords,
        lookback_days=lookback_days,
        max_results=max_papers * 20,  # overfetch then filter
    )
    print(f"[digest] Fetched {len(papers)} papers from arxiv")

    if not papers:
        print("[digest] No papers found in the lookback window. Skipping.")
        return

    # Step 2: Score
    scored = score_papers(papers, keywords, model=scoring_model)
    high_score = [p for p in scored if p.get("score", 0) >= threshold]
    high_score.sort(key=lambda p: p.get("score", 0), reverse=True)
    high_score = high_score[:max_papers]

    if not high_score:
        print(f"[digest] No papers scored >= {threshold}. Top scores found:")
        top5 = sorted(scored, key=lambda p: p.get("score", 0), reverse=True)[:5]
        for p in top5:
            print(f"  [{p['score']:.2f}] {p['title'][:80]}")
        print("[digest] Nothing to push. Done.")
        return

    print(f"[digest] {len(high_score)} papers passed threshold. Top 3:")
    for p in high_score[:3]:
        print(f"  [{p['score']:.2f}] {p['title'][:80]}")

    # Step 3: Chinese digest
    high_score = generate_digests(high_score, model=digest_model)

    # Step 4: Build card & push
    card = build_digest_card(high_score)

    if dry_run:
        print("\n[DRY RUN] Would send the following card:")
        print(json.dumps(card, ensure_ascii=False, indent=2).encode('utf-8', errors='replace').decode('utf-8'))
        print("\n[DRY RUN] Papers that would be pushed:")
        for p in high_score:
            print(f"  [{p['score']:.2f}] {p['digest_cn']}")
        return

    webhook = webhook_url or os.environ.get("FEISHU_WEBHOOK", "")
    if not webhook:
        print("[digest] FEISHU_WEBHOOK not set. Printing results only:")
        for p in high_score:
            print(f"  [{p['score']:.2f}] {p['digest_cn']}")
        return

    ok = send_feishu_card(webhook, card)
    if ok:
        print(f"[digest] ✅ Pushed {len(high_score)} papers to Feishu")
    else:
        print("[digest] ❌ Failed to push to Feishu")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Arxiv Daily Digest")
    parser.add_argument("--categories", type=str, help="Comma-separated categories, e.g. cs.CL,cs.AI")
    parser.add_argument("--keywords", type=str, help="Comma-separated keywords, e.g. agent,RAG")
    parser.add_argument("--threshold", type=float, help="Relevance threshold (0-1)")
    parser.add_argument("--max-papers", type=int, help="Max papers to push")
    parser.add_argument("--lookback", type=int, help="Lookback days")
    parser.add_argument("--dry-run", action="store_true", help="Print results without pushing")
    parser.add_argument("--config", type=str, help="Path to keywords.yaml")

    args = parser.parse_args()

    # Load config
    config = load_config_abs(args.config) if args.config else load_config()

    categories = args.categories.split(",") if args.categories else config.get("categories", ["cs.CL", "cs.AI"])
    keywords = args.keywords.split(",") if args.keywords else config.get("keywords", [])
    threshold = args.threshold if args.threshold is not None else config.get("threshold", 0.75)
    max_papers = args.max_papers if args.max_papers is not None else config.get("max_papers", 10)
    lookback = args.lookback if args.lookback is not None else config.get("lookback_days", 1)

    scoring_model = config.get("scoring_model") or os.environ.get("LLM_SCORING_MODEL")
    digest_model = config.get("digest_model") or os.environ.get("LLM_MODEL")

    run_digest(
        categories=categories,
        keywords=keywords,
        threshold=threshold,
        max_papers=max_papers,
        lookback_days=lookback,
        dry_run=args.dry_run,
        scoring_model=scoring_model,
        digest_model=digest_model,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
