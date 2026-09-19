# 示例：金融市场状态变化

本例使用冻结的论文元数据与摘要，演示从检索规划到补检、主题组织和综述初稿的完整流程。使用 `heuristic` 模式，不调用模型 API。

在项目根目录运行：

```bash
python run_agent.py --topic "金融市场 regime shift：经济状态、统计状态、实时识别、样本外预测与配置价值" --language zh --year-from 1980 --year-to 2026 --llm-mode heuristic --offline-corpus examples/regime-shift/corpus.json --no-full-text --min-candidates 30 --min-selected 15 --target-selected 24 --relevance-threshold 3.2 --recent-years 5 --min-recent-selected 5 --min-chinese-selected 2 --theme-min 3 --theme-max 6 --max-reflection-rounds 1 --supplemental-query-count 6 --out runs/regime-shift
```

这次运行从 90 篇初始候选中筛选出 24 篇，补检后最终入选 27 篇，形成 3 个主题。可查看[综述初稿](publication/review.md)、[参考文献](publication/references.bib)和[质量报告](quality-report.json)。质量报告显示论证一致性检查未通过；这些文件是流程示例，仍需人工核验和修改。
