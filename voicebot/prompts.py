def build_chat_prompt(
    user_message: str,
    intent: str,
    session: dict,
    conversation_history: list
) -> str:
    """
    Build the LLM prompt based on current intent, session state,
    and conversation history.
    """

    # Format conversation history
    history_text = ""
    for msg in conversation_history[-6:]:  # last 6 turns
        history_text += f"{msg['role'].upper()}: {msg['content']}\n"

    # Slot info
    slots = session.get("slots", {})
    awaiting = session.get("awaiting")
    ticket_id = session.get("ticket_id")

    # Intent-specific instructions
    intent_instructions = {
        "check_status": f"""
The user wants to check a status (loan, complaint, order, etc.).
Currently collected slots: {slots}
Currently awaiting from user: {awaiting if awaiting else "nothing yet"}

Rules:
- If loan_id is not collected yet, ask for it politely
- If loan_id is collected (available in slots), provide status: "Your loan [ID] is approved and being processed."
- Do NOT make up real data. Use placeholder responses.
- Keep response short and clear.
""",
        "raise_complaint": f"""
The user wants to raise a complaint.
Currently collected slots: {slots}
Ticket ID assigned: {ticket_id if ticket_id else "not yet assigned"}
Currently awaiting from user: {awaiting if awaiting else "nothing yet"}

Rules:
- If complaint description is not collected yet, ask the user to describe their issue
- If description is collected, confirm: "Complaint registered. Ticket ID: [ticket_id]. We will resolve within 48 hours."
- If user asks status of existing complaint, reply: "Your complaint (Ticket {ticket_id}) is currently in progress."
- Keep response empathetic and clear.
""",
        "general_query": """
The user has a general question.
Rules:
- Answer helpfully and concisely
- If you don't know the answer, say: "I don't have that information. Would you like me to connect you to an agent?"
- Never make up facts
""",
        "context_switch": """
The user is switching to a different topic mid-conversation.
Rules:
- Acknowledge the switch politely: "Sure, let me help you with that instead."
- Then respond to their new request
- Do NOT continue the previous topic
""",
        "unknown": """
The intent is unclear.
Rules:
- Politely ask for clarification
- Offer options: "I can help you check a status, raise a complaint, or answer general questions. What would you like?"
"""
    }

    instruction = intent_instructions.get(intent, intent_instructions["unknown"])

    prompt = f"""
You are a helpful customer support voicebot. You are professional, empathetic, and concise.

CONVERSATION HISTORY:
{history_text}

CURRENT INTENT: {intent}

INSTRUCTIONS:
{instruction}

RULES (always follow):
- Never hallucinate or make up data
- If unsure, ask for clarification or offer to connect to an agent: "Connecting you to a live agent."
- Keep responses under 3 sentences unless absolutely necessary
- Be warm and professional

USER SAID: "{user_message}"

YOUR RESPONSE:
"""
    return prompt


def build_slot_collection_prompt(slot_name: str) -> str:
    """Simple prompts to collect missing slots."""
    slot_prompts = {
        "loan_id":        "Could you please provide your Loan ID so I can check the status?",
        "complaint_desc": "Please describe your issue and I'll register a complaint right away.",
        "order_id":       "Could you please share your Order ID?",
        "complaint_id":   "Please provide your Complaint ID or Ticket number."
    }
    return slot_prompts.get(slot_name, f"Could you please provide your {slot_name.replace('_', ' ')}?")
