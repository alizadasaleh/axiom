"""Load a synthetic 30-paper NLP corpus into SQLite + Qdrant.

Lets the UI run end-to-end before real OpenAlex ingestion exists.

IDEMPOTENT: re-running wipes & recreates the Qdrant collection (axiom_v1) and
rewrites the SQLite rows, so you can run it repeatedly and get a clean state.

Run from the repo root:
    python scripts/bootstrap_synthetic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom import config, db
from axiom.indexer import reindex_qdrant

# ---------------------------------------------------------------------------
# Synthetic corpus: 30 plausible ACL-style NLP papers, 2021–2025.
# Fields mirror what OpenAlex will eventually provide so neighbors look sane.
# Each: id, title, abstract, year, venue, cited_by_count, concepts, references
# ---------------------------------------------------------------------------
PAPERS = [
    # --- Retrieval-Augmented Generation -----------------------------------
    {
        "id": "S0001", "year": 2021, "venue": "NeurIPS", "cited_by_count": 2100,
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "abstract": "We introduce a general architecture that combines a pretrained "
        "sequence-to-sequence generator with a dense retriever over an external "
        "passage index. The model attends over retrieved documents while decoding, "
        "letting it ground generations in non-parametric memory. On open-domain "
        "question answering and fact verification the approach outperforms purely "
        "parametric baselines and produces more specific, factual outputs.",
        "concepts": [("Retrieval-augmented generation", 2), ("Question answering", 2),
                     ("Information retrieval", 1)],
        "references": [],
    },
    {
        "id": "S0002", "year": 2023, "venue": "ACL", "cited_by_count": 410,
        "title": "Self-Reflective Retrieval Augmentation Reduces Hallucination in Long-Form QA",
        "abstract": "Retrieval-augmented language models still hallucinate when retrieved "
        "passages are irrelevant. We propose a self-reflective decoding loop in which the "
        "model critiques whether each retrieved passage supports its draft answer and "
        "selectively re-retrieves. On long-form question answering this lowers "
        "unsupported claims by a wide margin while preserving fluency.",
        "concepts": [("Retrieval-augmented generation", 2), ("Hallucination", 2),
                     ("Question answering", 2)],
        "references": ["S0001"],
    },
    {
        "id": "S0003", "year": 2024, "venue": "EMNLP", "cited_by_count": 95,
        "title": "Adaptive Chunking Strategies for Retrieval-Augmented Generation Pipelines",
        "abstract": "The granularity at which documents are split into passages strongly "
        "affects retrieval quality, yet most pipelines use fixed-size chunks. We study "
        "semantic and discourse-aware chunking and show that adapting chunk boundaries to "
        "topical shifts improves answer accuracy in retrieval-augmented generation across "
        "several knowledge-intensive benchmarks.",
        "concepts": [("Retrieval-augmented generation", 2), ("Information retrieval", 1),
                     ("Text segmentation", 2)],
        "references": ["S0001", "S0002"],
    },
    {
        "id": "S0004", "year": 2025, "venue": "ACL", "cited_by_count": 12,
        "title": "Graph-Conditioned Retrieval for Multi-Hop Question Answering",
        "abstract": "Multi-hop questions require composing evidence scattered across "
        "documents. We condition retrieval on an entity graph induced from the corpus so "
        "that each retrieval step expands along plausible reasoning paths. Coupling this "
        "with a retrieval-augmented generator yields stronger multi-hop question answering "
        "than flat dense retrieval, especially on questions needing three or more hops.",
        "concepts": [("Retrieval-augmented generation", 2), ("Question answering", 2),
                     ("Knowledge graph", 2)],
        "references": ["S0001", "S0003"],
    },
    # --- LoRA / PEFT -------------------------------------------------------
    {
        "id": "S0005", "year": 2021, "venue": "ICLR", "cited_by_count": 5200,
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "abstract": "Fine-tuning all parameters of large language models is increasingly "
        "impractical. We propose freezing the pretrained weights and injecting trainable "
        "low-rank decomposition matrices into each layer, drastically cutting the number of "
        "trainable parameters and optimizer memory. The adapted models match full "
        "fine-tuning quality on a range of tasks with no added inference latency.",
        "concepts": [("Parameter-efficient fine-tuning", 2), ("Low-rank adaptation", 3),
                     ("Transfer learning", 1)],
        "references": [],
    },
    {
        "id": "S0006", "year": 2023, "venue": "NeurIPS", "cited_by_count": 1800,
        "title": "QLoRA: Efficient Finetuning of Quantized Language Models",
        "abstract": "We show that 4-bit quantized language models can be finetuned through "
        "low-rank adapters without loss of quality. By back-propagating gradients through a "
        "frozen quantized backbone into the adapters, we finetune very large models on a "
        "single consumer GPU. The method makes parameter-efficient fine-tuning accessible "
        "under tight memory budgets.",
        "concepts": [("Parameter-efficient fine-tuning", 2), ("Quantization", 2),
                     ("Low-rank adaptation", 3)],
        "references": ["S0005"],
    },
    {
        "id": "S0007", "year": 2024, "venue": "ICLR", "cited_by_count": 220,
        "title": "Rank-Adaptive PEFT: Allocating Capacity Across Layers Dynamically",
        "abstract": "Fixed-rank adapters waste capacity on layers that need little adaptation "
        "and starve those that need more. We introduce a parameter-efficient fine-tuning "
        "scheme that learns per-layer ranks under a global budget via differentiable mask "
        "pruning. The adaptive allocation consistently beats uniform low-rank adaptation at "
        "the same parameter count.",
        "concepts": [("Parameter-efficient fine-tuning", 2), ("Low-rank adaptation", 3),
                     ("Model compression", 2)],
        "references": ["S0005", "S0006"],
    },
    {
        "id": "S0008", "year": 2025, "venue": "EMNLP", "cited_by_count": 8,
        "title": "Composable Adapters for Continual Instruction Tuning",
        "abstract": "Sequentially fine-tuning a model on new instruction datasets causes it "
        "to forget earlier skills. We store each skill as an independent low-rank adapter "
        "and learn a lightweight router that composes them at inference. This preserves "
        "old capabilities while adding new ones, giving strong continual instruction "
        "tuning without rehearsal data.",
        "concepts": [("Parameter-efficient fine-tuning", 2), ("Continual learning", 2),
                     ("Instruction tuning", 2)],
        "references": ["S0005", "S0007", "S0015"],
    },
    # --- Low-resource & multilingual NLP -----------------------------------
    {
        "id": "S0009", "year": 2021, "venue": "EMNLP", "cited_by_count": 640,
        "title": "Cross-Lingual Transfer for Low-Resource Named Entity Recognition",
        "abstract": "Annotated data for named entity recognition is scarce in most of the "
        "world's languages. We study zero-shot cross-lingual transfer from a multilingual "
        "encoder and find that aligning subword representations across scripts substantially "
        "improves low-resource performance. Adding a few hundred target examples closes most "
        "of the remaining gap.",
        "concepts": [("Low-resource NLP", 2), ("Named entity recognition", 2),
                     ("Cross-lingual transfer", 2)],
        "references": [],
    },
    {
        "id": "S0010", "year": 2022, "venue": "ACL", "cited_by_count": 380,
        "title": "Data Augmentation via Back-Translation for Low-Resource Machine Translation",
        "abstract": "We revisit back-translation as a way to exploit monolingual data for "
        "low-resource machine translation. By iteratively generating synthetic source "
        "sentences and filtering them with a quality estimator, we improve translation for "
        "language pairs with fewer than fifty thousand parallel sentences, narrowing the gap "
        "to high-resource settings.",
        "concepts": [("Low-resource NLP", 2), ("Machine translation", 2),
                     ("Data augmentation", 2)],
        "references": ["S0009"],
    },
    {
        "id": "S0011", "year": 2024, "venue": "NAACL", "cited_by_count": 70,
        "title": "Tokenizer Fairness Across Scripts in Multilingual Language Models",
        "abstract": "Multilingual tokenizers fragment low-resource scripts into far more "
        "subwords than Latin scripts, inflating cost and hurting accuracy. We quantify this "
        "disparity and propose a script-balanced vocabulary objective. The resulting "
        "tokenizer reduces over-segmentation for low-resource languages and improves "
        "downstream multilingual understanding.",
        "concepts": [("Low-resource NLP", 2), ("Tokenization", 2),
                     ("Multilingual models", 2)],
        "references": ["S0009"],
    },
    {
        "id": "S0012", "year": 2025, "venue": "ACL", "cited_by_count": 5,
        "title": "Synthetic Dialogue Generation for Under-Resourced Conversational Agents",
        "abstract": "Building conversational agents for under-resourced languages is hard "
        "because dialogue corpora barely exist. We prompt a multilingual model to generate "
        "grounded synthetic dialogues, filter them with native-speaker preference models, "
        "and fine-tune on the result. The pipeline produces usable low-resource dialogue "
        "systems with minimal human annotation.",
        "concepts": [("Low-resource NLP", 2), ("Dialogue systems", 2),
                     ("Data augmentation", 2)],
        "references": ["S0010", "S0011"],
    },
    # --- Evaluation --------------------------------------------------------
    {
        "id": "S0013", "year": 2022, "venue": "NeurIPS", "cited_by_count": 900,
        "title": "Beyond Accuracy: Holistic Evaluation of Language Models",
        "abstract": "Single-number leaderboards hide important failure modes. We propose a "
        "holistic evaluation framework that measures accuracy, calibration, robustness, "
        "bias, and efficiency across many scenarios under standardized conditions. The "
        "framework exposes trade-offs invisible to accuracy-only evaluation and makes model "
        "comparisons reproducible.",
        "concepts": [("Model evaluation", 2), ("Benchmarking", 2),
                     ("Robustness", 2)],
        "references": [],
    },
    {
        "id": "S0014", "year": 2024, "venue": "EMNLP", "cited_by_count": 150,
        "title": "LLM-as-a-Judge: Reliability and Bias of Model-Based Evaluation",
        "abstract": "Using a strong language model to score the outputs of other models is "
        "fast and cheap, but its reliability is unclear. We audit position bias, verbosity "
        "bias, and self-preference in model-based evaluation and propose calibration "
        "procedures. With these corrections, automatic judgments agree with human "
        "preferences closely enough for routine evaluation.",
        "concepts": [("Model evaluation", 2), ("Automatic evaluation", 2),
                     ("Bias", 2)],
        "references": ["S0013"],
    },
    # --- Instruction tuning / alignment ------------------------------------
    {
        "id": "S0015", "year": 2022, "venue": "NeurIPS", "cited_by_count": 3400,
        "title": "Training Language Models to Follow Instructions with Human Feedback",
        "abstract": "We align a pretrained language model to user intent by collecting "
        "human demonstrations and preference comparisons, then optimizing the model with "
        "reinforcement learning from human feedback. The aligned model is preferred over a "
        "much larger unaligned baseline and is more truthful and less toxic, showing that "
        "alignment can matter more than scale.",
        "concepts": [("Instruction tuning", 2), ("Reinforcement learning from human feedback", 3),
                     ("Alignment", 2)],
        "references": [],
    },
    {
        "id": "S0016", "year": 2023, "venue": "ACL", "cited_by_count": 720,
        "title": "Self-Instruct: Bootstrapping Instruction Data from Language Models",
        "abstract": "High-quality instruction data is expensive to annotate. We bootstrap it "
        "by prompting a language model to generate new instructions, inputs, and outputs, "
        "then filtering for validity and diversity. Fine-tuning on this self-generated data "
        "substantially improves instruction-following, rivaling models trained on "
        "human-written instructions.",
        "concepts": [("Instruction tuning", 2), ("Data augmentation", 2),
                     ("Self-training", 2)],
        "references": ["S0015"],
    },
    {
        "id": "S0017", "year": 2024, "venue": "ICLR", "cited_by_count": 1100,
        "title": "Direct Preference Optimization: Aligning Models Without Reward Models",
        "abstract": "Reinforcement learning from human feedback is complex and unstable. We "
        "show that the same preference objective can be optimized directly with a simple "
        "classification loss, removing the explicit reward model and online sampling. This "
        "direct preference optimization is stable, lightweight, and matches or exceeds "
        "reinforcement-learning-based alignment.",
        "concepts": [("Alignment", 2), ("Preference optimization", 3),
                     ("Instruction tuning", 2)],
        "references": ["S0015"],
    },
    # --- Hallucination & factuality ----------------------------------------
    {
        "id": "S0018", "year": 2023, "venue": "EMNLP", "cited_by_count": 540,
        "title": "Surveying Hallucination in Natural Language Generation",
        "abstract": "Hallucination — generating content unsupported by the input — undermines "
        "trust in language generation. We taxonomize intrinsic and extrinsic hallucination "
        "across summarization, translation, and dialogue, survey detection metrics, and "
        "catalog mitigation strategies. We argue that faithfulness must be evaluated "
        "separately from fluency.",
        "concepts": [("Hallucination", 2), ("Natural language generation", 1),
                     ("Faithfulness", 2)],
        "references": [],
    },
    {
        "id": "S0019", "year": 2025, "venue": "NAACL", "cited_by_count": 18,
        "title": "Uncertainty-Aware Decoding for Factual Text Generation",
        "abstract": "Models often state unsupported facts with high confidence. We estimate "
        "token-level epistemic uncertainty during decoding and abstain or hedge when it is "
        "high. On knowledge-intensive generation this reduces confidently wrong statements "
        "while keeping most correct content, improving factuality without external "
        "retrieval.",
        "concepts": [("Hallucination", 2), ("Uncertainty estimation", 2),
                     ("Decoding", 2)],
        "references": ["S0018", "S0002"],
    },
    # --- Long context ------------------------------------------------------
    {
        "id": "S0020", "year": 2023, "venue": "NeurIPS", "cited_by_count": 460,
        "title": "Efficient Attention for Long-Context Language Modeling",
        "abstract": "Standard self-attention scales quadratically with sequence length, "
        "limiting context windows. We propose a sparse-plus-low-rank attention that "
        "approximates the full attention matrix in near-linear time. Models trained with it "
        "handle much longer contexts at a fraction of the memory while preserving "
        "downstream quality.",
        "concepts": [("Long-context modeling", 2), ("Attention mechanism", 2),
                     ("Efficient transformers", 2)],
        "references": [],
    },
    {
        "id": "S0021", "year": 2024, "venue": "ACL", "cited_by_count": 130,
        "title": "Lost in the Middle: How Position Affects Long-Context Retrieval",
        "abstract": "We probe how language models use information placed at different "
        "positions in a long input. Models reliably use evidence at the beginning and end "
        "of the context but frequently miss relevant content in the middle. We connect this "
        "U-shaped curve to training data layout and propose position-balanced fine-tuning "
        "as a partial remedy.",
        "concepts": [("Long-context modeling", 2), ("Information retrieval", 1),
                     ("Probing", 2)],
        "references": ["S0020"],
    },
    # --- Prompting & reasoning ---------------------------------------------
    {
        "id": "S0022", "year": 2022, "venue": "NeurIPS", "cited_by_count": 6100,
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "abstract": "We show that prompting a large language model to produce intermediate "
        "reasoning steps before its final answer markedly improves performance on "
        "arithmetic, commonsense, and symbolic reasoning. This chain-of-thought ability "
        "emerges only at sufficient scale and requires no additional training, just a few "
        "worked examples in the prompt.",
        "concepts": [("Prompting", 2), ("Reasoning", 2),
                     ("In-context learning", 2)],
        "references": [],
    },
    {
        "id": "S0023", "year": 2023, "venue": "ICLR", "cited_by_count": 1500,
        "title": "Self-Consistency Improves Chain-of-Thought Reasoning",
        "abstract": "Greedy decoding of a single reasoning chain is brittle. We sample many "
        "diverse chains of thought and marginalize over them by majority vote on the final "
        "answer. This self-consistency decoding substantially boosts reasoning accuracy "
        "across arithmetic and commonsense benchmarks without any fine-tuning.",
        "concepts": [("Prompting", 2), ("Reasoning", 2),
                     ("Decoding", 2)],
        "references": ["S0022"],
    },
    {
        "id": "S0024", "year": 2025, "venue": "ICLR", "cited_by_count": 9,
        "title": "Verifier-Guided Search for Reliable Multi-Step Reasoning",
        "abstract": "Sampling many reasoning chains is wasteful and still error-prone. We "
        "train a stepwise verifier that scores partial reasoning and guide a search over "
        "chains toward high-scoring states. This verifier-guided search reaches higher "
        "reasoning accuracy than self-consistency at a fraction of the sampling budget.",
        "concepts": [("Reasoning", 2), ("Search", 2),
                     ("Verification", 2)],
        "references": ["S0022", "S0023"],
    },
    # --- Distillation & efficiency -----------------------------------------
    {
        "id": "S0025", "year": 2021, "venue": "ACL", "cited_by_count": 340,
        "title": "Task-Aware Knowledge Distillation for Compact Encoders",
        "abstract": "Deploying large encoders is costly. We distill a large teacher into a "
        "compact student using a task-aware objective that weights tokens by their "
        "downstream importance. The distilled encoder retains most of the teacher's "
        "accuracy on classification and retrieval while running several times faster.",
        "concepts": [("Knowledge distillation", 2), ("Model compression", 2),
                     ("Efficient transformers", 2)],
        "references": [],
    },
    {
        "id": "S0026", "year": 2024, "venue": "EMNLP", "cited_by_count": 60,
        "title": "Speculative Decoding with Self-Distilled Draft Models",
        "abstract": "Speculative decoding speeds up generation by drafting tokens with a "
        "small model and verifying them with a large one. We self-distill the draft model "
        "from the target so their distributions align, raising the acceptance rate. This "
        "yields larger speedups than off-the-shelf draft models with no quality loss.",
        "concepts": [("Knowledge distillation", 2), ("Decoding", 2),
                     ("Inference efficiency", 2)],
        "references": ["S0025"],
    },
    # --- Embeddings & retrieval models -------------------------------------
    {
        "id": "S0027", "year": 2021, "venue": "NAACL", "cited_by_count": 880,
        "title": "Dense Passage Retrieval with Hard Negative Mining",
        "abstract": "We learn dense representations for passage retrieval by contrasting "
        "questions against relevant and carefully mined hard-negative passages. Trained "
        "this way, a dual encoder retrieves far more accurately than sparse term-matching "
        "baselines and serves as a strong first stage for open-domain question answering.",
        "concepts": [("Information retrieval", 1), ("Dense retrieval", 2),
                     ("Contrastive learning", 2)],
        "references": [],
    },
    {
        "id": "S0028", "year": 2023, "venue": "ACL", "cited_by_count": 300,
        "title": "Instruction-Tuned Text Embeddings for Heterogeneous Retrieval",
        "abstract": "A single embedding space rarely serves every retrieval task. We train a "
        "text embedding model on diverse tasks with natural-language task instructions, so "
        "the same encoder adapts its representation to the requested notion of similarity. "
        "The model generalizes to unseen retrieval tasks better than task-specific "
        "encoders.",
        "concepts": [("Dense retrieval", 2), ("Text embeddings", 2),
                     ("Instruction tuning", 2)],
        "references": ["S0027"],
    },
    {
        "id": "S0029", "year": 2022, "venue": "EMNLP", "cited_by_count": 250,
        "title": "Scientific Document Representations from Citation-Aware Pretraining",
        "abstract": "We pretrain a document encoder for scientific papers using citation "
        "links as a weak signal of relatedness, pulling citing and cited papers together in "
        "embedding space. The resulting representations excel at finding related work, "
        "recommending citations, and clustering scientific literature by topic.",
        "concepts": [("Text embeddings", 2), ("Scientific NLP", 2),
                     ("Citation analysis", 2)],
        "references": ["S0027"],
    },
    {
        "id": "S0030", "year": 2025, "venue": "NAACL", "cited_by_count": 4,
        "title": "Contrastive Reranking for Scientific Literature Discovery",
        "abstract": "First-stage dense retrieval surfaces broadly related papers but blurs "
        "fine distinctions. We add a contrastively trained cross-encoder reranker tailored "
        "to scientific text that sharpens relevance among near-duplicates and survey "
        "papers. On literature-discovery benchmarks the reranker markedly improves the "
        "ordering of related work.",
        "concepts": [("Dense retrieval", 2), ("Reranking", 2),
                     ("Scientific NLP", 2)],
        "references": ["S0027", "S0029"],
    },
]


def load_sqlite() -> None:
    conn = db.connect()
    try:
        db.init_db(conn)
        # Idempotent: clear prior synthetic rows before reinserting.
        db.reset_corpus(conn)

        for p in PAPERS:
            db.insert_paper(
                conn,
                openalex_id=p["id"],
                title=p["title"],
                abstract=p["abstract"],
                publication_year=p["year"],
                venue_id=None,
                venue=p["venue"],
                cited_by_count=p["cited_by_count"],
                doi=None,
            )
            db.insert_concepts(conn, p["id"], [(c, lvl) for c, lvl in p["concepts"]])
            db.insert_provenance(
                conn, paper_id=p["id"], source="synthetic",
                abstract=p["abstract"], has_fulltext=False,
            )
            # Citation edges from synthetic referenced_works[].
            db.insert_citation_edges(
                conn, [(p["id"], dst, p["year"]) for dst in p["references"]]
            )
        conn.commit()
        print(f"[sqlite] wrote {len(PAPERS)} papers to {config.DB_PATH}")
    finally:
        conn.close()


def main() -> None:
    load_sqlite()
    reindex_qdrant()
    print("\nBootstrap complete. Run:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
