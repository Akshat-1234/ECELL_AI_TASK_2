import time

from src.train import answer_question

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

# Different values of k to compare
k_values = [3,5,10,20]

# Loop over all k values
for k in k_values:

    total_retrieval_time = 0
    total_generation_time = 0

    print(f"\nTesting k = {k}")

    for query in queries:

        # Asking the question using current k
        result = answer_question(
            query,
            k
        )

        # Adding times to compute average later
        total_retrieval_time += result["retrieval_time"]
        total_generation_time += result["generation_time"]

        print("\nQuery:")
        print(query)

        print("\nConfidence:")
        print(result["confidence"])

        print("\nRetrieval Time:")
        print(round(result["retrieval_time"],4),"seconds")

        print("\nGeneration Time:")
        print(round(result["generation_time"],4),"seconds")

        print("\nSources Used:")

        for source in result["sources"]:

            print(
                source["file"],
                "Page",
                source["page"]
            )

        print("-"*50)

    avg_retrieval_time = total_retrieval_time / len(queries)
    avg_generation_time = total_generation_time / len(queries)

    print("\nAverage Retrieval Time:")
    print(round(avg_retrieval_time,4),"seconds")

    print("\nAverage Generation Time:")
    print(round(avg_generation_time,4),"seconds")

    print("\nAverage Total Latency:")
    print(round(
        avg_retrieval_time + avg_generation_time,
        4
    ),"seconds")

    print("\n")