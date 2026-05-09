import uuid
from fastapi import APIRouter
from pydantic import BaseModel
from voicebot.chat import process_message
from voicebot.context import clear_session

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str = None   # optional — auto-generated if not provided
    message: str


class VoiceRequest(BaseModel):
    session_id: str = None
    audio_text: str          # mocked — in real system this comes from STT


@router.post("/chat")
async def chat(payload: ChatRequest):
    """
    POST /chat

    Main text-based conversation endpoint.
    Handles multi-turn conversation with context retention.

    Request JSON:
    {
        "session_id": "abc123",   ← optional, omit for new conversation
        "message": "Check my loan status"
    }

    Response JSON:
    {
        "session_id": "abc123",
        "intent": "check_status",
        "slots": { "loan_id": "12345" },
        "ticket_id": null,
        "awaiting": null,
        "response": "Your loan 12345 is approved.",
        "history": [...]
    }
    """

    # Auto-generate session ID for new conversations
    session_id = payload.session_id or str(uuid.uuid4())

    result = process_message(session_id, payload.message)
    return result


@router.post("/voice")
async def voice(payload: VoiceRequest):
    """
    POST /voice

    Simulated voice endpoint.
    In production: audio → STT → text → same pipeline → TTS → audio
    Here we mock STT by accepting text directly.

    Request JSON:
    {
        "session_id": "abc123",
        "audio_text": "I want to raise a complaint"
    }
    """

    session_id = payload.session_id or str(uuid.uuid4())

    # Same pipeline as /chat — STT is mocked
    result = process_message(session_id, payload.audio_text)

    # In production: pass result["response"] through TTS here
    result["note"] = "Voice simulated. In production: response would be converted to audio via TTS."

    return result


@router.delete("/session/{session_id}")
async def end_session(session_id: str):
    """
    DELETE /session/{session_id}
    Clears a user session completely.
    """
    clear_session(session_id)
    return {"status": "success", "message": f"Session {session_id} cleared."}
