import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CONFIDENCE_THRESHOLD = 0.7


def detect_intent(query: str) -> dict:
    """Detect user intent from query using Groq LLM."""
    prompt = f"""
You are an intent classifier. Classify the user query into exactly one of these intents:
- Question: User is asking for information
- Complaint: User is expressing dissatisfaction or frustration
- Request: User wants an action to be performed
- Feedback: User is providing opinion or feedback

User query: "{query}"

Respond in this exact JSON format only, no extra text:
{{
  "intent": "<one of: Question, Complaint, Request, Feedback>",
  "confidence": <float between 0 and 1>,
  "reason": "<one line explanation>"
}}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    result = json.loads(response.choices[0].message.content)

    if result["confidence"] < CONFIDENCE_THRESHOLD:
        result["intent"] = "Question"
        result["fallback"] = True
    else:
        result["fallback"] = False

    return result