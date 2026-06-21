from src.localmodel import generate_local_answer

context = """
Transformers use self attention mechanisms.
"""

question = "What do transformers use?"

answer = generate_local_answer(
    context,
    question
)

print(answer)