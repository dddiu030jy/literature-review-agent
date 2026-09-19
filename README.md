# LitReview Agent

**把研究主题整理成有来源、可复核的文献综述初稿。**

LitReview Agent 面向需要快速梳理研究领域的个人与团队。它从主题出发，完成中英文检索、文献筛选、证据抽取、主题组织和带引用的初稿写作；每一步的记录单独保存，便于研究者追查结论从哪里来、哪些问题还需要补证据。

## 核心能力

- 从 OpenAlex、Semantic Scholar、Crossref 和 arXiv 检索文献，并合并重复记录。
- 根据相关性和证据角色筛选论文，发现覆盖缺口后补充检索。
- 按研究问题组织多篇论文，生成 Markdown 综述与 BibTeX 参考文献。
- 输出检索、抽取和质量检查记录，支持人工复核。

## 运行

需要 Python 3.10 或更新版本。在线运行还需要学术数据源网络访问和 OpenAI 兼容的 Chat Completions API。

```bash
python -m pip install -e ".[fulltext]"
export OPENAI_API_KEY="你的 API Key"
export OPENAI_MODEL="你使用的模型"
python run_agent.py --topic "你的研究主题" --llm-mode openai --out runs/my-review
```

Windows PowerShell 使用 `$env:OPENAI_API_KEY="你的 API Key"` 和 `$env:OPENAI_MODEL="你使用的模型"` 设置变量。其他兼容服务可另设 `OPENAI_BASE_URL`；[`.env.example`](.env.example)列出了可配置项，但程序不会自动读取 `.env` 文件。

运行结果中的 `publication/review.md` 是综述初稿，`publication/references.bib` 是正文引用文献；`audit/` 保存检索、筛选、抽取与质量记录。正式使用前仍需核对原论文和关键判断。

## 两个可复现示例

示例使用冻结语料和启发式模式，无需 API Key，适合查看完整流程与输出格式。

| 示例                                          | 内容                     |
| ------------------------------------------- | ---------------------- |
| [金融市场状态变化](examples/regime-shift/README.md) | 语料、运行命令、综述初稿、参考文献和质量报告 |
| [市场大跌预警](examples/crash-warning/README.md)  | 语料、运行命令、综述初稿、参考文献和质量报告 |

系统模块和两个质量检查闭环见[架构与原理](docs/architecture.md)。核心代码在 `src/litreview_agent/`，自动测试在 `tests/`。
