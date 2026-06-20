# CLAUDE.md

Arxiv Digest — LLM 驱动的 arxiv 论文筛选 + 飞书速递 + 精读系统。

## Project Layout

```
.github/workflows/
  daily-digest.yml      # schedule: daily UTC 22:00
  paper-analysis.yml    # GitHub Issue / manual trigger, OCR + reading + app push
scripts/
  digest.py             # abstract prefilter → candidate OCR → decision card
  reading.py            # OCR-grounded quick understanding
  deep_note.py          # structured deep-reading card (Obsidian format)
  benchmark.py          # multi-paper experiment table extraction
  utils.py              # shared utilities (LLM client, Feishu API, arxiv helpers)
  paper_context.py      # OCR evidence, institution aliases, grounded links
config/
  keywords.yaml         # keywords, categories, threshold
  institutions.yaml     # fixed CSRankings/company reference set
prompts/                # editable prompts for every reading stage
```

## Key Design Decisions

1. **Conservative scoring**: LLM prompt is designed to UNDER-score rather than OVER-score. 
   Default threshold is 0.75. The philosophy is "missing a relevant paper is better than flooding with noise."
   
2. **Three-tier assertion classification**: paper-reading explicitly labels each claim as:
   - "论文明确说" (explicitly stated by paper)
   - "合理推断" (reasonable inference)
   - "未支撑" (unsupported)

3. **Hard constraints on fabrication**: deep-note and reading scripts must never invent experimental numbers, 
   ablation conclusions, dataset details, or open-source status. When uncertain, output "未知".

4. **Feishu interaction flow**: 
   - Daily digest → custom bot webhook (simple push)
   - Digest card button → prefilled GitHub Issue → GitHub Actions
   - Reading result → Feishu app API (target chat/user configured by receive ID)

5. **OCR for reading**: the remote paper-analysis workflow enables PaddleOCR and passes the
   extracted experiment section to the reading model. OCR failure falls back to abstract-only analysis.

6. **Institution signal, never a pass**: digest content relevance is scored independently. A configured
   institution can add at most 0.08 only when content relevance is at least 0.5; bonuses never stack.

## LLM Configuration

- Uses OpenAI-compatible API (`/v1/chat/completions`)
- Configurable via secrets: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`
- `digest.py` uses two separate calls: scoring (cheap, parallel) and summarization (one per high-score paper)
- `reading.py` uses a single high-quality call with structured output

## Python Version

Python 3.11+. Dependencies in `requirements.txt`.
