import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

from services.embeddings import get_embedding
from services.vector_store import retrieve_top_k
from services.intent import detect_intent
from services.emotion import detect_emotion, get_tone_instruction
from prompts.templates import build_response_prompt, build_no_context_prompt

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

router = APIRouter()

SIMILARITY_THRESHOLD = 0.75


class QueryRequest(BaseModel):
    user_id: str
    document_id: str
    query: str


@router.post("/query")
async def query_document(payload: QueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Step 1: Detect intent and emotion
    intent_result = detect_intent(payload.query)
    emotion_result = detect_emotion(payload.query)

    intent = intent_result["intent"]
    emotion = emotion_result["emotion"]
    tone_instruction = get_tone_instruction(emotion)

    # Step 2: Generate embedding for the query
    query_embedding = get_embedding(payload.query)

    # Step 3: Retrieve top-K relevant chunks from FAISS
    top_chunks = retrieve_top_k(
        user_id=payload.user_id,
        document_id=payload.document_id,
        query_embedding=query_embedding,
        k=3
    )

    # Step 4: Filter by similarity threshold
    relevant_chunks = [c["chunk"] for c in top_chunks if c["score"] >= SIMILARITY_THRESHOLD]

    # Step 5: Build prompt
    if relevant_chunks:
        prompt = build_response_prompt(
            query=payload.query,
            retrieved_chunks=relevant_chunks,
            intent=intent,
            emotion=emotion,
            tone_instruction=tone_instruction
        )
    else:
        prompt = build_no_context_prompt(
            query=payload.query,
            intent=intent,
            tone_instruction=tone_instruction
        )

    # Step 6: Generate response using Groq
    llm_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    response_text = llm_response.choices[0].message.content.strip()

    return {
        "status": "success",
        "query": payload.query,
        "intent": intent_result,
        "emotion": emotion_result,
        "retrieved_chunks": top_chunks,
        "response": response_text
    }