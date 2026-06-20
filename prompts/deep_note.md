将已有论文分析与 OCR 实验证据整理成长期可复用的中文研究笔记。禁止编造数字、机构、代码、数据集或消融结论；缺失内容写“未知”。

返回 JSON：
{
  "paper_title":"标题", "arxiv_id":"编号", "input_coverage":"输入覆盖",
  "research_question":"研究问题",
  "method":{"name":"方法名", "category":"方法类型", "summary":"核心直觉与步骤"},
  "key_findings":["关键发现"],
  "experiments":{"datasets":["数据集"], "baselines":["基线"], "metrics":["指标"], "main_results":"主要结果", "ablation":"消融"},
  "limitations":["局限"], "reproducibility_concerns":["复现关注点"],
  "inspirations":["可迁移启发"], "reading_priority":"值得精读|值得速读|可暂缓",
  "tags":["标签"]
}
