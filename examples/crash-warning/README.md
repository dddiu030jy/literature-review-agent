# 示例：市场大跌预警

本例使用冻结的论文元数据与摘要，展示证据覆盖与补充检索。使用 `heuristic` 模式，不调用模型 API。

在项目根目录运行：

```bash
python run_agent.py --topic "金融市场大跌预警：哪些指标具有预测能力、可预测性的边界与样本外稳定性（不限ETF）" --language zh --year-from 1980 --year-to 2026 --llm-mode heuristic --offline-corpus examples/crash-warning/corpus.json --no-full-text --min-candidates 30 --min-selected 15 --target-selected 24 --relevance-threshold 2.0 --recent-years 5 --min-recent-selected 5 --min-chinese-selected 1 --theme-min 3 --theme-max 4 --max-reflection-rounds 1 --supplemental-query-count 6 --out runs/crash-warning
```

这次运行从 48 篇初始候选中筛选出 24 篇，补检后最终入选 30 篇，形成 4 个主题。可查看[综述初稿](publication/review.md)、[参考文献](publication/references.bib)和[质量报告](quality-report.json)。质量报告显示论证一致性检查未通过；这些文件是流程示例，仍需人工核验和修改。
