import time
from pathlib import Path

import pandas as pd

from src.train import (
    answer_question,
    answer_question_rerank,
    answer_question_local,
)

from src.metrics import (
    context_relevance,
    answer_relevance,
    query_resolved,
)

BASE_DIR = Path(__file__).parent.parent

queries = [
    "Explain self attention.",
    "What is masked language modeling?",
    "What is in-context learning?",
    "How does Retrieval-Augmented Generation work?",
    "What is Low Rank Adaptation?",
    "Why was Sentence-BERT introduced?",
    "Compare BERT and GPT-3.",
    "How did BERT build upon the Transformer architecture?",
    "Compare GPT-3 and Llama 2.",
    "How does Retrieval-Augmented Generation address limitations of GPT-3?",
    "Why is LoRA useful for fine tuning Llama models?",
    "How does Sentence-BERT differ from BERT?",
    "How are attention mechanisms used in Retrieval-Augmented Generation systems?",
    "Compare BERT, GPT-3 and Llama 2 in terms of training objectives and applications.",
    "Which techniques were introduced to improve parameter efficiency in large language models?",
    "How have large language models evolved from Transformers to GPT-3 and Llama 2?",
]
queries = queries[:2]
pipelines = {
    "API LLM": answer_question,
    "API + Rerank": answer_question_rerank,
    "Local LLM": answer_question_local,
}

# Set to a small number while debugging (e.g. queries[:4]) to iterate fast.
# Restore the full list before producing the final report numbers.
# queries = queries[:4]

all_results = []
run_start = time.time()

for pipeline_name, pipeline_fn in pipelines.items():

    print("\n" + "=" * 60)
    print(pipeline_name)
    print("=" * 60)

    total_cr = 0.0
    total_ar = 0.0
    total_qr = 0
    total_retrieval_time = 0.0
    total_generation_time = 0.0

    pipeline_start = time.time()

    for i, query in enumerate(queries, start=1):

        query_start = time.time()

        # Single call per query. Each pipeline now returns the docs it
        # actually retrieved/used, so we no longer re-run FAISS search
        # a second time just to compute Context Relevance. This also
        # means CR for the rerank pipeline is measured against the
        # reranked top-k, not the raw FAISS top-k, which is the
        # correct comparison to make.
        result = pipeline_fn(query)

        docs = result.get("retrieved_docs", [])

        cr = context_relevance(query, docs) if docs else 0.0
        ar = answer_relevance(query, result.get("answer", ""))
        qr = query_resolved(result.get("answer"))

        total_cr += cr
        total_ar += ar
        total_qr += qr

        retrieval_time = result.get("retrieval_time", 0.0)
        generation_time = result.get("generation_time", 0.0)

        total_retrieval_time += retrieval_time
        total_generation_time += generation_time

        all_results.append({
            "Pipeline": pipeline_name,
            "Query": query,
            "CR": cr,
            "AR": ar,
            "QR": qr,
            "Retrieval_Time": retrieval_time,
            "Generation_Time": generation_time,
        })

        query_elapsed = time.time() - query_start

        # Progress logging so you can see live which pipeline/query is
        # slow instead of waiting for the whole run to finish blind.
        print(
            f"[{i}/{len(queries)}] {query_elapsed:.2f}s total "
            f"(retrieval {retrieval_time:.2f}s, generation {generation_time:.2f}s) "
            f"CR={cr:.3f} AR={ar:.3f} QR={qr} :: {query[:60]}"
        )

    n = len(queries)
    pipeline_elapsed = time.time() - pipeline_start

    print("\n" + "-" * 60)
    print(f"{pipeline_name} SUMMARY (took {pipeline_elapsed:.1f}s total)")
    print("-" * 60)
    print("Average Context Relevance:", round(total_cr / n, 4))
    print("Average Answer Relevance:", round(total_ar / n, 4))
    print("Query Resolution Rate:", round(total_qr / n, 4))
    print("Average Retrieval Time:", round(total_retrieval_time / n, 4))
    print("Average Generation Time:", round(total_generation_time / n, 4))
    print(
        "Average Total Latency:",
        round((total_retrieval_time + total_generation_time) / n, 4),
    )

total_elapsed = time.time() - run_start
print(f"\nFull evaluation run took {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

df = pd.DataFrame(all_results)
output_path = BASE_DIR / "evaluation.csv"
df.to_csv(output_path, index=False)

print(f"\nEvaluation saved successfully to {output_path}")