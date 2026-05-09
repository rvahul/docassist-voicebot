# DocAssist+ & VoiceBot

An AI-powered document assistant and customer support voicebot built with FastAPI, Groq LLM, FAISS, and Sentence Transformers.

---

## Problem Statement

### DocAssist+
A RAG-based document assistant that:
- Accepts document uploads and indexes them
- Detects user intent (Question, Complaint, Request, Feedback)
- Detects user emotion (Happy, Neutral, Frustrated, Angry)
- Retrieves relevant chunks and generates tone-adjusted responses

### VoiceBot
A multi-turn customer support voicebot that:
- Handles slot filling (collects missing information)
- Maintains context across conversation turns
- Detects and handles context switches
- Supports both text (chat) and simulated voice input

---

## Tech Stack

| Component | Tool |
|---|---|
| API Framework | FastAPI |
| LLM | Groq (llama-3.3-70b-versatile) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS (faiss-cpu) |
| Session Store | In-memory Python dict |
| Environment | Python 3.11, M1 Mac |

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/rvahul/docassist-voicebot.git
cd docassist-voicebot
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_key_here
```
Get a free key at: https://console.groq.com

### 5. Run the server
```bash
uvicorn main:app --reload
```

### 6. Open Swagger UI
```
http://127.0.0.1:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | /documents/upload | Upload and index a document |
| POST | /query | Query a document with intent and emotion detection |
| POST | /voicebot/chat | Multi-turn text conversation |
| POST | /voicebot/voice | Simulated voice conversation |
| DELETE | /voicebot/session/{id} | Clear a session |

---

## Sample Inputs & Outputs

### DocAssist+ — Upload
```json
// Request (multipart/form-data)
{ "user_id": "user123", "file": "test.txt" }

// Response
{
  "status": "success",
  "document_id": "f3a1c-uuid",
  "user_id": "user123",
  "chunks_stored": 4,
  "message": "Document uploaded and indexed successfully"
}
```

### DocAssist+ — Query (Neutral Question)
```json
// Request
{
  "user_id": "user123",
  "document_id": "f3a1c-uuid",
  "query": "What is the return policy?"
}

// Response
{
  "intent": { "intent": "Question", "confidence": 0.95 },
  "emotion": { "emotion": "Neutral", "confidence": 0.90 },
  "response": "Returns are accepted within 7 days of delivery."
}
```

### DocAssist+ — Query (Angry Complaint)
```json
// Request
{
  "user_id": "user123",
  "document_id": "f3a1c-uuid",
  "query": "This is ridiculous, my refund is still not processed!"
}

// Response
{
  "intent": { "intent": "Complaint", "confidence": 0.97 },
  "emotion": { "emotion": "Angry", "confidence": 0.93 },
  "response": "I sincerely apologize for the inconvenience. Refunds are processed within 5 business days..."
}
```

---

## VoiceBot — 3 Scenarios

### Scenario 1: Slot Filling (Loan Status Check)
Bot collects missing information (loan ID) before responding.

```
User:  "Check my loan status"
Bot:   "Could you please provide your Loan ID so I can check the status?"
User:  "12345"
Bot:   "Your loan 12345 is approved and being processed."
```

### Scenario 2: Context Continuity (Complaint Flow)
Bot remembers context and ticket ID across multiple turns.

```
User:  "I want to raise a complaint"
Bot:   "Please describe your issue and I'll register a complaint right away."
User:  "My payment failed but money was deducted from my account"
Bot:   "Complaint registered. Ticket ID: 16E45FAB. We will resolve within 48 hours."
User:  "What is the status of my complaint?"
Bot:   "Your complaint (Ticket 16E45FAB) is currently in progress."
```

### Scenario 3: Context Switch
Bot detects when user changes topic and resets to new intent.

```
User:  "Check my loan status"
Bot:   "Could you please provide your Loan ID?"
User:  "Actually I want to raise a complaint instead"
Bot:   "Sure, let me help you with that instead. Please describe your issue."
```

---

## Architecture

### DocAssist+ Flow
```
Upload:
User uploads file
→ Extract text
→ Chunk into 500-char pieces with 50-char overlap
→ Generate embeddings (sentence-transformers)
→ Store in FAISS with userId + documentId

Query:
User sends query
→ Detect intent (Question/Complaint/Request/Feedback)
→ Detect emotion (Happy/Neutral/Frustrated/Angry)
→ Embed query → Search FAISS top-3 chunks
→ Build prompt with context + intent + emotion
→ Generate response via Groq LLM
→ Return tone-adjusted response
```

### VoiceBot Flow
```
User sends message
→ Detect intent + context switch
→ Check session for existing slots
→ Collect missing slots if needed
→ Generate response via Groq LLM
→ Update session history
→ Return response + session state
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Groq over OpenAI | Free tier, fast inference, no billing needed |
| sentence-transformers for embeddings | Free, runs locally, no API cost |
| FAISS for vector store | Lightweight, runs in-memory, no external service needed |
| K=3 chunks retrieval | Enough context without overloading LLM prompt |
| Cosine similarity (normalized IP) | More accurate for semantic search than L2 |
| Confidence threshold 0.7 | Prevents wrong classifications from causing bad responses |
| Separate intent and emotion detection | They serve different purposes — what vs how |

---

## Edge Cases Handled

| Edge Case | Handling |
|---|---|
| Low similarity score (< 0.75) | Returns "no relevant info found" instead of wrong answer |
| Low confidence intent (< 0.7) | Defaults to Question (safest fallback) |
| Low confidence emotion (< 0.7) | Defaults to Neutral (safest fallback) |
| Empty query | Returns 400 error with clear message |
| Empty document | Returns 400 error with clear message |
| Context switch mid-conversation | Resets slots and intent, starts fresh flow |
| Missing slots | Bot asks for them politely before proceeding |

---

## Production Considerations

| Concern | Solution |
|---|---|
| Multiple users | Each user has own FAISS index tagged with userId |
| Slow response | Run intent+emotion detection in parallel using asyncio |
| High cost | Use smaller model for classification, larger only for response |
| Session persistence | Replace in-memory dict with Redis |
| Vector persistence | Replace FAISS with Pinecone or Weaviate |
| Voice in production | Add Whisper (STT) + ElevenLabs or gTTS (TTS) |

---

## Project Structure

```
docassist-voicebot/
├── main.py                  # FastAPI entry point — registers all routes
├── requirements.txt         # All dependencies
├── test.txt                 # Sample document for testing
├── routes/
│   ├── __init__.py
│   ├── upload.py            # POST /documents/upload
│   └── query.py             # POST /query
├── services/
│   ├── __init__.py
│   ├── embeddings.py        # Chunking + sentence-transformer embeddings
│   ├── vector_store.py      # FAISS storage and retrieval
│   ├── intent.py            # DocAssist intent detection via Groq
│   └── emotion.py           # Emotion detection + tone mapping
├── prompts/
│   ├── __init__.py
│   └── templates.py         # LLM prompt templates for DocAssist
└── voicebot/
    ├── __init__.py
    ├── routes.py            # POST /voicebot/chat, /voicebot/voice
    ├── chat.py              # Core conversation logic + slot filling
    ├── context.py           # Session management and history
    ├── intent.py            # VoiceBot intent detection via Groq
    └── prompts.py           # VoiceBot prompt templates
```

---

## Assumptions

- FAISS runs in-memory (resets on server restart — use Pinecone for persistence)
- Session data is in-memory (use Redis for production)
- Voice endpoint mocks STT/TTS — accepts text directly
- Loan/complaint status responses are simulated (no real database)
- Complaint ticket IDs are randomly generated 8-character strings

---

## Author
Rahul Verma
