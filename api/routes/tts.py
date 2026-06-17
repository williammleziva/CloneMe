import tempfile

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()


class TTSRequest(BaseModel):
    text: str


@router.post("/synthesize", response_class=FileResponse)
async def synthesize(request: TTSRequest, req: Request):
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        path = req.app.state.synthesizer.synthesize(request.text, tmp.name)
        return FileResponse(path, media_type="audio/wav", filename="response.wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
