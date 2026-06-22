import os
import time
import tempfile
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()


class VideoRequest(BaseModel):
    message: str
    session_id: str = "default"


@router.post("/generate")
async def generate_video(request: VideoRequest, req: Request, background_tasks: BackgroundTasks):
    """Full pipeline: RAG answer → Chatterbox voice → SadTalker talking-head video."""
    if req.app.state.generator is None:
        raise HTTPException(status_code=503, detail="Video generator unavailable. Download SadTalker models first.")
    try:
        uid = uuid.uuid4().hex[:8]
        t_total = time.perf_counter()

        print(f"[api/video] [{uid}] Step 1/3 — RAG inference")
        t0 = time.perf_counter()
        text = req.app.state.chain.invoke(
            {"question": request.message},
            config={"configurable": {"session_id": request.session_id}},
        )
        print(f"[api/video] [{uid}] RAG done in {time.perf_counter() - t0:.1f}s — {len(text)} chars")

        print(f"[api/video] [{uid}] Step 2/3 — TTS synthesis")
        t0 = time.perf_counter()
        _, audio_path = tempfile.mkstemp(suffix=".wav", prefix=f"clone_{uid}_")
        try:
            req.app.state.synthesizer.synthesize(text, audio_path)
            print(f"[api/video] [{uid}] TTS done in {time.perf_counter() - t0:.1f}s")

            print(f"[api/video] [{uid}] Step 3/3 — SadTalker video generation")
            t0 = time.perf_counter()
            video_path = req.app.state.generator.generate(audio_path, f"response_{uid}.mp4")
            print(f"[api/video] [{uid}] SadTalker done in {time.perf_counter() - t0:.1f}s")
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

        print(f"[api/video] [{uid}] Total pipeline: {time.perf_counter() - t_total:.1f}s → {video_path}")
        response = FileResponse(video_path, media_type="video/mp4", filename="response.mp4")
        response.headers["X-Response-Text"] = text
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
