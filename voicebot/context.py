from typing import Optional
import uuid

# In-memory session store
# Structure: { session_id: { "intent": str, "slots": {}, "history": [], "ticket_id": str } }
sessions: dict = {}


def get_session(session_id: str) -> dict:
    """Get existing session or create new one."""
    if session_id not in sessions:
        sessions[session_id] = {
            "intent": None,          # current active intent
            "slots": {},             # collected slot values e.g. { "loan_id": "12345" }
            "history": [],           # full conversation history
            "ticket_id": None,       # complaint ticket ID if raised
            "awaiting": None         # what bot is currently waiting for from user
        }
    return sessions[session_id]


def update_session(session_id: str, updates: dict):
    """Update specific fields in a session."""
    session = get_session(session_id)
    session.update(updates)


def reset_intent(session_id: str):
    """
    Reset intent and slots but keep history.
    Used when user switches context mid-conversation.
    """
    session = get_session(session_id)
    session["intent"] = None
    session["slots"] = {}
    session["awaiting"] = None
    session["ticket_id"] = None


def add_to_history(session_id: str, role: str, message: str):
    """Add a message to conversation history."""
    session = get_session(session_id)
    session["history"].append({
        "role": role,      # "user" or "assistant"
        "content": message
    })


def get_history(session_id: str) -> list:
    """Get full conversation history for a session."""
    session = get_session(session_id)
    return session["history"]


def generate_ticket_id() -> str:
    """Generate a unique complaint ticket ID."""
    return str(uuid.uuid4())[:8].upper()


def clear_session(session_id: str):
    """Completely clear a session."""
    if session_id in sessions:
        del sessions[session_id]
