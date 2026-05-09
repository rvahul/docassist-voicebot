import os
from groq import Groq
from dotenv import load_dotenv

from voicebot.context import (
    get_session, update_session, reset_intent,
    add_to_history, get_history, generate_ticket_id
)
from voicebot.intent import detect_intent
from voicebot.prompts import build_chat_prompt, build_slot_collection_prompt

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def process_message(session_id: str, user_message: str) -> dict:
    """
    Main function that processes a user message and returns bot response.
    Handles all 3 scenarios: slot filling, context continuity, context switch.
    """

    # Step 1: Get current session state
    session = get_session(session_id)
    history = get_history(session_id)

    # Step 2: Add user message to history
    add_to_history(session_id, "user", user_message)

    # Step 3: Detect intent (with conversation history for context)
    intent_result = detect_intent(user_message, history)
    detected_intent = intent_result["intent"]
    is_context_switch = intent_result.get("is_context_switch", False)

    # ─────────────────────────────────────────────
    # SCENARIO 3: Context Switch
    # User was mid-conversation but changed topic
    # ─────────────────────────────────────────────
    if is_context_switch or (
        session["intent"] is not None and
        detected_intent != session["intent"] and
        detected_intent not in ["unknown"]
    ):
        reset_intent(session_id)
        session = get_session(session_id)
        update_session(session_id, {"intent": detected_intent})

    # ─────────────────────────────────────────────
    # Set intent if not already set
    # ─────────────────────────────────────────────
    elif session["intent"] is None:
        update_session(session_id, {"intent": detected_intent})

    # Refresh session after updates
    session = get_session(session_id)
    active_intent = session["intent"] or detected_intent

    # ─────────────────────────────────────────────
    # SCENARIO 1: Slot Filling — Check Status
    # Collect loan_id if missing
    # ─────────────────────────────────────────────
    if active_intent == "check_status":
        if "loan_id" not in session["slots"]:
            # Check if current message contains a loan ID (simple number check)
            if session["awaiting"] == "loan_id" and user_message.strip().isdigit():
                # User just provided the loan ID
                update_session(session_id, {
                    "slots": {"loan_id": user_message.strip()},
                    "awaiting": None
                })
                session = get_session(session_id)
            else:
                # Ask for loan ID
                update_session(session_id, {"awaiting": "loan_id"})
                bot_response = build_slot_collection_prompt("loan_id")
                add_to_history(session_id, "assistant", bot_response)
                return _build_response(session_id, bot_response, active_intent, session)

    # ─────────────────────────────────────────────
    # SCENARIO 2: Context Continuity — Complaint
    # Collect complaint description, assign ticket
    # ─────────────────────────────────────────────
    if active_intent == "raise_complaint":
        if "complaint_desc" not in session["slots"]:
            if session["awaiting"] == "complaint_desc" and len(user_message.strip()) > 3:
                # User just described their complaint
                ticket_id = generate_ticket_id()
                update_session(session_id, {
                    "slots": {"complaint_desc": user_message.strip()},
                    "awaiting": None,
                    "ticket_id": ticket_id
                })
                session = get_session(session_id)
            else:
                # Ask for complaint description
                update_session(session_id, {"awaiting": "complaint_desc"})
                bot_response = build_slot_collection_prompt("complaint_desc")
                add_to_history(session_id, "assistant", bot_response)
                return _build_response(session_id, bot_response, active_intent, session)

    # ─────────────────────────────────────────────
    # Generate LLM Response
    # ─────────────────────────────────────────────
    session = get_session(session_id)
    prompt = build_chat_prompt(
        user_message=user_message,
        intent=active_intent,
        session=session,
        conversation_history=get_history(session_id)
    )

    llm_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    bot_response = llm_response.choices[0].message.content.strip()

    # Save bot response to history
    add_to_history(session_id, "assistant", bot_response)

    return _build_response(session_id, bot_response, active_intent, session)


def _build_response(session_id: str, bot_response: str, intent: str, session: dict) -> dict:
    """Build the final API response dict."""
    return {
        "session_id": session_id,
        "intent": intent,
        "slots": session.get("slots", {}),
        "ticket_id": session.get("ticket_id"),
        "awaiting": session.get("awaiting"),
        "response": bot_response,
        "history": get_history(session_id)
    }
