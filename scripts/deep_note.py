"""
paper-deep-note: 结构化精读卡
产出 Obsidian 格式的精读卡，长期可复用。

硬约束：
- 不编造实验数字、消融结论、数据集细节
- 不确定的字段显式标"未知"
- 显式标注"仅基于摘要判断"（如果确实只拿到了摘要）

Usage:
    python deep_note.py --url https://arxiv.org/abs/2210.03629
    python deep_note.py --from-reading reading_output.json
    python deep_note.py --url https://arxiv.org/abs/2210.03629 --use-ocr  # 用 OCR 读全文
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    llm_chat, get_llm_model,
    ocr_arxiv_pdf, extract_experiment_section,
)
from paper_context import load_prompt


DEEP_NOTE_SYSTEM_PROMPT = load_prompt("deep_note")


def generate_deep_note(paper_info: dict, model: str = None) -> dict:
    """Generate a structured deep-reading note."""
    if not model:
        model = get_llm_model()

    # Build input from whatever info we have
    title = paper_info.get("title", paper_info.get("paper_title", ""))
    summary = paper_info.get("summary", "")
    authors = paper_info.get("authors", [])

    user_prompt = f"""论文信息：

标题：{title}
作者：{', '.join(authors[:5]) if authors else '未知'}
摘要：{summary}

已有分析（来自 paper-reading）：
{json.dumps({k: v for k, v in paper_info.items() if k not in ['summary', 'authors']}, ensure_ascii=False, indent=2)}

请生成精读卡。"""

    try:
        result = llm_chat(
            messages=[
                {"role": "system", "content": DEEP_NOTE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.1,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        note = json.loads(result)
        note["generated_at"] = datetime.now().isoformat()
        return note
    except Exception as e:
        print(f"[deep_note error] {e}")
        return {"error": str(e), "paper_title": title}


def format_obsidian(note: dict) -> str:
    """Format deep note as Obsidian markdown."""
    tags = " ".join(f"#{t}" for t in note.get("tags", []))
    priority = note.get("reading_priority", "")
    priority_emoji = {"值得精读": "⭐", "值得速读": "📖", "可暂缓": "📎"}.get(priority, "")

    lines = [
        f"---",
        f"tags: [{', '.join(note.get('tags', []))}]",
        f"priority: {priority}",
        f"arxiv: {note.get('arxiv_id', '')}",
        f"generated: {note.get('generated_at', '')}",
        f"---",
        f"",
        f"# {priority_emoji} {note.get('paper_title', 'Unknown')}",
        f"",
        f"**arxiv**: [{note.get('arxiv_id', '')}](https://arxiv.org/abs/{note.get('arxiv_id', '')})",
        f"**输入覆盖**: {note.get('input_coverage', '未知')}",
        f"**阅读优先级**: {priority}",
        f"",
        f"## 研究问题",
        f"{note.get('research_question', '未知')}",
        f"",
        f"## 方法",
        f"- **方法名**: {note.get('method', {}).get('name', '未知')}",
        f"- **类型**: {note.get('method', {}).get('category', '未知')}",
        f"- **概述**: {note.get('method', {}).get('summary', '未知')}",
        f"",
        f"## 核心发现",
    ]

    for f_ in note.get("key_findings", []):
        lines.append(f"- {f_}")

    exp = note.get("experiments", {})
    lines += [
        f"",
        f"## 实验",
        f"- **数据集**: {', '.join(exp.get('datasets', ['未知']))}",
        f"- **Baselines**: {', '.join(exp.get('baselines', ['未知']))}",
        f"- **指标**: {', '.join(exp.get('metrics', ['未知']))}",
        f"- **主要结果**: {exp.get('main_results', '未知')}",
        f"- **消融**: {exp.get('ablation', '未知')}",
        f"",
        f"## 局限",
    ]
    for l in note.get("limitations", []):
        lines.append(f"- {l}")

    lines += [
        f"",
        f"## 复现关注点",
    ]
    for r in note.get("reproducibility_concerns", []):
        lines.append(f"- {r}")

    lines += [
        f"",
        f"## 启发",
    ]
    for i in note.get("inspirations", []):
        lines.append(f"- {i}")

    lines += [
        f"",
        f"---",
        f"*{tags}*",
    ]

    return "\n".join(lines)


def run_deep_note(
    arxiv_url: str = None,
    from_reading: str = None,
    output_dir: str = None,
    dry_run: bool = False,
    model: str = None,
    use_ocr: bool = False,
):
    """Generate a deep reading note."""
    paper_info = {}
    ocr_result = None

    if from_reading:
        # Load from reading.py output
        try:
            with open(from_reading, "r", encoding="utf-8") as f:
                text = f.read()
            # Find JSON in output (may have log lines before it)
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                paper_info = json.loads(text[start:end])
            arxiv_url = paper_info.get("abstract_url", "")
        except Exception as e:
            print(f"[deep_note] Failed to load reading output: {e}")
            return None
    elif arxiv_url:
        # Fetch basic info
        from reading import fetch_single_paper
        paper_info = fetch_single_paper(arxiv_url)
    else:
        print("[deep_note] No input provided")
        return None

    print(f"[deep_note] Generating note for: {paper_info.get('title', paper_info.get('paper_title', 'Unknown'))}")

    # 如果开启 OCR，下载 PDF 全文提取实验部分
    if use_ocr and arxiv_url:
        print("[deep_note] Enabling OCR for full-text extraction...")
        try:
            ocr_result = ocr_arxiv_pdf(arxiv_url, output_dir="output/ocr")
            if ocr_result:
                exp_section = extract_experiment_section(ocr_result)
                paper_info["full_text_experiment"] = exp_section
                paper_info["input_coverage"] = "全文（OCR）"
                print(f"[deep_note] OCR extracted {len(ocr_result['markdown'])} chars, experiment section: {len(exp_section)} chars")
        except Exception as e:
            print(f"[deep_note] OCR failed, falling back to abstract: {e}")

    if not paper_info.get("input_coverage"):
        paper_info["input_coverage"] = "仅摘要"

    note = generate_deep_note(paper_info, model=model)

    if note.get("error"):
        print(f"[deep_note] Error: {note['error']}")
        return note

    # Format as Obsidian markdown
    md = format_obsidian(note)

    if dry_run:
        print("\n[DRY RUN] Deep note:")
        print(md)
        return note

    # Save to file
    output_dir = output_dir or os.environ.get("DEEP_NOTE_DIR", "output")
    os.makedirs(output_dir, exist_ok=True)

    arxiv_id = note.get("arxiv_id", "unknown")
    safe_id = arxiv_id.replace("/", "_")
    filename = f"{safe_id}_deepnote.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[deep_note] ✅ Saved to {filepath}")
    return note


def main():
    parser = argparse.ArgumentParser(description="Generate deep reading notes")
    parser.add_argument("--url", type=str, help="Arxiv URL or ID")
    parser.add_argument("--from-reading", type=str, help="Path to reading.py output JSON")
    parser.add_argument("--output-dir", type=str, help="Output directory for markdown files")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-ocr", action="store_true", help="Use PaddleOCR to extract full text from PDF")
    args = parser.parse_args()

    run_deep_note(
        arxiv_url=args.url,
        from_reading=args.from_reading,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        use_ocr=args.use_ocr,
    )


if __name__ == "__main__":
    main()
