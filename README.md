# Arxiv Digest

LLM 驱动的 arxiv 论文筛选 + 速递 + 精读系统。推送到飞书。

## 功能

| Skill | 说明 | 触发方式 |
|-------|------|---------|
| **paper-feishu-digest** | 摘要预筛 → 候选 OCR → 机构/代码/实验决策卡 | 每日定时 (GitHub Actions) |
| **paper-reading** | OCR 首页与实验提取 + 快速理解式精读 | 飞书按钮 → GitHub Issue → Actions |
| **paper-deep-note** | 结构化精读卡（Obsidian 格式） | paper-reading 判定「值得精读」自动触发 |
| **benchmark-extractor** | 多篇论文实验表抽取 | workflow_dispatch 手动触发 |

## 快速开始

1. Fork 此仓库
2. 在 Settings → Secrets 中添加必要的 secrets（见下方）
3. 修改 `config/keywords.yaml` 配置你的关键词和类别
4. 启用 GitHub Actions

## 需要的 Secrets

| Secret | 用途 |
|--------|------|
| `LLM_API_KEY` | LLM API key |
| `LLM_BASE_URL` | LLM API base URL |
| `LLM_MODEL` | 模型名称 |
| `FEISHU_WEBHOOK` | 每日速递使用的群自定义机器人 Webhook |
| `FEISHU_APP_ID` | 发送精读结果的飞书应用 App ID |
| `FEISHU_APP_SECRET` | 发送精读结果的飞书应用 App Secret |
| `FEISHU_RECEIVE_ID` | 精读结果目标群的 `chat_id`（或用户 ID） |
| `FEISHU_RECEIVE_ID_TYPE` | 可选，默认 `chat_id`；私聊可设为 `open_id` |
| `OCR_API_TOKEN` | PaddleOCR API Token |

飞书应用需要开通发送消息权限。私聊时设置 `FEISHU_RECEIVE_ID_TYPE=open_id`，无需加入群。仓库 workflow 已声明 `issues: write`，用于回写并关闭精读 Issue。

## 调整阅读方式

Prompt 均为可直接编辑的 Markdown：

| 文件 | 用途 |
|------|------|
| `prompts/digest_scoring.md` | 摘要相关性预筛 |
| `prompts/digest_decision.md` | OCR 后的粗读决策字段 |
| `prompts/reading.md` | 精读卡内容与 JSON 结构 |
| `prompts/deep_note.md` | 长期研究笔记 |

`config/institutions.yaml` 是固定机构参考集，包含 CSRankings AI 领域院校快照和生成模型厂商别名。`config/keywords.yaml` 控制预筛阈值、OCR 候选上限和机构加分。

每次 Actions 运行会上传 `digest-analysis` 或 `paper-reading-evidence` artifact。查看 JSON 中的 `ocr_status`、`input_coverage`、`content_score` 和 `institution_bonus`，即可确认 OCR 是否参与分析以及机构如何影响最终分数。

## 本地测试

```bash
pip install -r requirements.txt
python scripts/digest.py --categories cs.CL --keywords "agent,RAG" --dry-run
```

## 架构

```
arxiv API → 摘要预筛 → 候选 OCR → 决策卡 → 飞书推送
                  ↓
        用户点击「精读」
                  ↓
       GitHub Issue → GitHub Actions → OCR → 快速理解式精读
                  ↓
        飞书应用 → 精读结果卡片
```
