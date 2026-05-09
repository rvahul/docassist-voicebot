import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CONFIDENCE_THRESHOLD = 0.7


def detect_emotion(query: str) -> dict:
    """Detect user emotion from query using Groq LLM."""
    prompt = f"""
You are an emotion classifier. Classify the emotion in the user query into exactly one of these:
- Happy: User is satisfied, positive, or appreciative
- Neutral: User has no strong emotion, just asking normally
- Frustrated: User is mildly upset or annoyed
- Angry: User is very upset or expressing strong dissatisfaction

User query: "{query}"

Respond in this exact JSON format only, no extra text:
{{
  "emotion": "<one of: Happy, Neutral, Frustrated, Angry>",
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
        result["emotion"] = "Neutral"
        result["fallback"] = True
    else:
        result["fallback"] = False

    return result


def get_tone_instruction(emotion: str) -> str:
    """Return tone instruction based on detected emotion."""
    tone_map = {
        "Happy":      "The user seems happy. Respond in a warm, positive, and friendly tone.",
        "Neutral":    "The user is neutral. Respond in a clear, professional, and informative tone.",
        "Frustrated": "The user seems frustrated. Respond with empathy, acknowledge their concern, and be calm and helpful.",
        "Angry":      "The user is angry. Respond with strong empathy, apologize sincerely, stay very calm, and offer a clear resolution."
    }
    return tone_map.get(emotion, tone_map["Neutral"])