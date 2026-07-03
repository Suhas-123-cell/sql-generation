"""Single source of truth for the prompt format.

Training data and inference must use the exact same message structure —
a template mismatch between the two is the most common fine-tuning bug.
"""

SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. Given a database schema and a question, "
    "reply with only the SQL query that answers the question. No explanations."
)


def build_messages(question, context):
    user = f"Schema:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
