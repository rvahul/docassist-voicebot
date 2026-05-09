import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from services.embeddings import chunk_text, get_embeddings_batch
from services.vector_store import store_embeddings

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    POST /documents/upload

    Accepts a text/pdf file and a userId.
    Extracts text → chunks → embeds → stores in FAISS.

    Request (multipart/form-data):
    - file: the document file (.txt or .pdf)
    - user_id: string identifier for the user

    Response JSON:
    {
        "status": "success",
        "document_id": "<uuid>",
        "user_id": "<user_id>",
        "chunks_stored": <int>,
        "message": "Document uploaded and indexed successfully"
    }
    """

    # Step 1: Read file content
    content = await file.read()

    # Step 2: Extract text (handle .txt and basic .pdf)
    try:
        if file.filename.endswith(".pdf"):
            import io
            from pypdf import PdfReader
            pdf = PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        else:
            text = content.decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Document appears to be empty.")

    # Step 3: Chunk the text
    chunks = chunk_text(text)

    # Step 4: Generate embeddings for all chunks
    embeddings = get_embeddings_batch(chunks)

    # Step 5: Store in FAISS with userId and documentId
    document_id = str(uuid.uuid4())
    store_embeddings(user_id, document_id, chunks, embeddings)

    return {
        "status": "success",
        "document_id": document_id,
        "user_id": user_id,
        "chunks_stored": len(chunks),
        "message": "Document uploaded and indexed successfully"
    }
