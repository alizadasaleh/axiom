# nDCG@10 retrieval eval report

Generated 2026-07-04T12:01:35.267104+00:00 · mode=`hybrid` · collection=`axiom_eval_v1` · corpus=1500 papers · 15 queries

> **Caveat:** relevance judgments in `eval/ndcg_queries.json` are single-pass, AI-drafted (see its `_provenance` field and `docs/DECISIONS.md` OD11) -- not the backlog's '2 annotators' acceptance criterion. Treat this as a working demo signal, not a validated quality claim. A low score can also mean the judgment set didn't happen to cover a paper the retriever correctly surfaced (judgments were built via keyword-pattern search over the sample, not an exhaustive relevance pass) -- inspect low scorers before treating them as retrieval failures.

**Mean nDCG@10: 0.453**

| Query | nDCG@10 |
|---|---|
| retrieval-augmented generation for question answering | 0.495 |
| parameter-efficient fine-tuning with low-rank adapters | 0.719 |
| detecting and mitigating hallucination in large language models | 0.344 |
| chain-of-thought prompting for reasoning in large language models | 0.867 |
| named entity recognition with limited labeled data | 0.501 |
| low-resource and multilingual machine translation | 0.000 |
| abstractive text summarization | 0.083 |
| open-domain question answering with dense retrieval | 0.471 |
| instruction tuning for large language models | 0.532 |
| bias and fairness evaluation in language models | 0.686 |
| task-oriented dialogue state tracking | 0.576 |
| knowledge distillation for compact and efficient models | 0.131 |
| aspect-based sentiment analysis | 0.829 |
| few-shot text classification | 0.555 |
| cross-lingual and multilingual representation learning | 0.000 |
