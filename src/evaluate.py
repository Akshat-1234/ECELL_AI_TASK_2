import pandas as pd

from train import (
    answer_question,
    answer_question_rerank,
    answer_question_local,
    vectorstore
)

from metrics import (
    context_relevance,
    answer_relevance,
    query_resolved
)

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
    "How have large language models evolved from Transformers to GPT-3 and Llama 2?"
]

pipelines = {
    "API LLM": answer_question,
    "API + Rerank": answer_question_rerank,
    "Local LLM": answer_question_local
}

all_results = []

for name, pipeline in pipelines.items():

    print("\n" + "="*60)
    print(name)
    print("="*60)

    total_cr = 0
    total_ar = 0
    total_qr = 0
    total_retrieval_time = 0
    total_generation_time = 0

    for query in queries:

        result = pipeline(query)

        docs_and_scores = vectorstore.similarity_search_with_score(query, k=5)
        docs = [doc for doc, _ in docs_and_scores]

        cr = context_relevance(query, docs)
        ar = answer_relevance(query, result["answer"])
        qr = query_resolved(result["answer"])

        total_cr += cr
        total_ar += ar
        total_qr += qr

        total_retrieval_time += result["retrieval_time"]
        total_generation_time += result["generation_time"]

        all_results.append({
            "Pipeline": name,
            "Query": query,
            "CR": cr,
            "AR": ar,
            "QR": qr,
            "Retrieval_Time": result["retrieval_time"],
            "Generation_Time": result["generation_time"]
        })

        print("\nQuery:", query)
        print("CR:", round(cr, 4))
        print("AR:", round(ar, 4))
        print("QR:", qr)
        print("Retrieval Time:", round(result["retrieval_time"], 4))
        print("Generation Time:", round(result["generation_time"], 4))

    n = len(queries)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("Average Context Relevance:", round(total_cr / n, 4))
    print("Average Answer Relevance:", round(total_ar / n, 4))
    print("Query Resolution Rate:", round(total_qr / n, 4))
    print("Average Retrieval Time:", round(total_retrieval_time / n, 4))
    print("Average Generation Time:", round(total_generation_time / n, 4))
    print("Average Total Latency:", round((total_retrieval_time + total_generation_time) / n, 4))

df = pd.DataFrame(all_results)
df.to_csv("../evaluation.csv", index=False)

print("\nEvaluation saved successfully")