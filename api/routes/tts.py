import os
import time
import tempfile

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()


class TTSRequest(BaseModel):
    text: str


@router.post("/synthesize", response_class=FileResponse)
async def synthesize(request: TTSRequest, req: Request, background_tasks: BackgroundTasks):
    print(f"[api/tts] /synthesize — {len(request.text)} chars")
    t0 = time.perf_counter()
    try:
        _, path = tempfile.mkstemp(suffix=".wav")
        path = req.app.state.synthesizer.synthesize(request.text, path)
        background_tasks.add_task(os.unlink, path)
        print(f"[api/tts] Done in {time.perf_counter() - t0:.1f}s")
        return FileResponse(path, media_type="audio/wav", filename="response.wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
