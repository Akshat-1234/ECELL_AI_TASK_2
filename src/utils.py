# Used to calculate confidence score from similarity scores
def calculate_confidence(scores):

    # Lower distance means better similarity
    avg_score = sum(scores) / len(scores)

    confidence = float(
    round(
        max(
            0.0,
            1 - avg_score / 10
        ),
        4
    )
)
    return confidence

# Used to remove duplicate sources
def get_sources(docs):

    sources = []

    for doc in docs:

        sources.append(
            {
                "file": doc.metadata["source_file"],
                "page": doc.metadata["page_number"]
            }
        )

    unique_sources = []

    seen = set()

    for source in sources:

        key = (
            source["file"],
            source["page"]
        )

        if key not in seen:

            seen.add(key)

            unique_sources.append(source)

    return unique_sources