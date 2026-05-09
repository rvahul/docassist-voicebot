def build_response_prompt(
    query: str,
    retrieved_chunks: list[str],
    intent: str,
    emotion: str,
    tone_instruction: str
) -> str:
    """
    Build the final LLM prompt using retrieved context,
    detected intent and emotion.
    """
    context = "\n\n".join(retrieved_chunks) if retrieved_chunks else "No relevant document found."

    intent_instruction = {
        "Question":  "Answer the user's question clearly and accurately using only the context provided.",
        "Complaint": "Acknowledge the user's complaint, apologize if appropriate, and provide a resolution or next steps based on the context.",
        "Request":   "Understand what the user wants done and provide clear action steps or confirmation based on the context.",
        "Feedback":  "Thank the user for their feedback and acknowledge their input professionally."
    }.get(intent, "Respond helpfully based on the context.")

    prompt = f"""
You are DocAssist+, a helpful AI assistant that answers questions based on provided documents.

TONE INSTRUCTION:
{tone_instruction}

INTENT INSTRUCTION:
{intent_instruction}

DOCUMENT CONTEXT:
{context}

USER QUERY:
{query}

RULES:
- Only use information from the DOCUMENT CONTEXT above.
- If the context does not contain relevant information, say: "I don't have enough information in the provided documents to answer this."
- Do NOT make up information.
- Keep the response concise and relevant.

YOUR RESPONSE:
"""
    return prompt


def build_no_context_prompt(query: str, intent: str, tone_instruction: str) -> str:
    """Fallback prompt when no relevant chunks are retrieved."""
    return f"""
You are DocAssist+, a helpful AI assistant.

{tone_instruction}

The user asked: "{query}"

Unfortunately, no relevant information was found in the uploaded documents.
Politely inform the user and suggest they check if the correct document was uploaded.
"""
