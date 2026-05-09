import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

VALID_INTENTS = ["check_status", "raise_complaint", "general_query", "context_switch", "unknown"]
CONFIDENCE_THRESHOLD = 0.7


def detect_intent(user_message: str, conversation_history: list) -> dict:
    """
    Detect intent from user message considering conversation history.
    Returns intent, confidence, and whether user is switching context.
    """

    # Build history string for context
    history_text = ""
    for msg in conversation_history[-4:]:  # last 4 messages for context
        history_text += f"{msg['role'].upper()}: {msg['content']}\n"

    prompt = f"""
You are an intent classifier for a customer support voicebot.

Conversation so far:
{history_text}

New user message: "{user_message}"

Classify the intent into exactly one of:
- check_status: User wants to check status of loan, complaint, order, etc.
- raise_complaint: User wants to raise or register a complaint
- general_query: User has a general question
- context_switch: User is changing topic mid-conversation (e.g., "actually I want to...")
- unknown: Cannot determine intent

Respond ONLY in this JSON format, no extra text:
{{
  "intent": "<one of the above>",
  "confidence": <float 0 to 1>,
  "is_context_switch": <true or false>,
  "reason": "<one line>"
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    result = json.loads(response.choices[0].message.content)

    if result["confidence"] < CONFIDENCE_THRESHOLD:
        result["intent"] = "unknown"
        result["fallback"] = True
    else:
        result["fallback"] = False

    return result
