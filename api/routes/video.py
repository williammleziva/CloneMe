import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()


class VideoRequest(BaseModel):
    message: str
    session_id: str = "default"


@router.post("/generate")
async def generate_video(request: VideoRequest, req: Request):
    """Full pipeline: RAG answer → Chatterbox voice → SadTalker talking-head video."""
    if req.app.state.generator is None:
        raise HTTPException(status_code=503, detail="Video generator unavailable. Download SadTalker models first.")
    try:
        uid = uuid.uuid4().hex[:8]

        text = req.app.state.chain.invoke(
            {"question": request.message},
            config={"configurable": {"session_id": request.session_id}},
        )

        audio_path = f"/tmp/clone_{uid}.wav"
        req.app.state.synthesizer.synthesize(text, audio_path)

        video_path = req.app.state.generator.generate(audio_path, f"response_{uid}.mp4")

        return FileResponse(video_path, media_type="video/mp4", filename="response.mp4")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
