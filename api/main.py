from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import chat, tts, video
from core.rag.chain import build_rag_chain
from core.tts.synthesizer import VoiceSynthesizer
from core.video.generator import TalkingHeadGenerator
from dotenv import load_dotenv
import os

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    name = os.getenv("CLONE_NAME", "Will")
    app.state.chain, _ = build_rag_chain(name=name)
    ref_audio = os.getenv("TTS_REF_AUDIO", "data/media/voice_ref/reference.wav")
    app.state.synthesizer = VoiceSynthesizer(ref_audio_path=ref_audio)
    try:
        app.state.generator = TalkingHeadGenerator()
    except (RuntimeError, FileNotFoundError) as e:
        app.state.generator = None
        print(f"[video] Generator unavailable: {e}")
    yield


app = FastAPI(title="CloneMe API", version="1.0.0", docs_url="/docs", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(tts.router, prefix="/tts", tags=["TTS"])
app.include_router(video.router, prefix="/video", tags=["Video"])


@app.get("/health")
def health():
    return {"status": "ok"}
