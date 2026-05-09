from fastapi import FastAPI
from routes.upload import router as upload_router
from routes.query import router as query_router
from voicebot.routes import router as voicebot_router

app = FastAPI(title="DocAssist+ & VoiceBot", version="1.0.0")

# DocAssist+ routes
app.include_router(upload_router, prefix="/documents", tags=["DocAssist - Upload"])
app.include_router(query_router, tags=["DocAssist - Query"])

# VoiceBot routes
app.include_router(voicebot_router, prefix="/voicebot", tags=["VoiceBot"])


@app.get("/")
def root():
    return {"message": "DocAssist+ and VoiceBot APIs are running"}
