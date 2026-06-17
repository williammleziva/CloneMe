from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from core.rag.chain import _session_store

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    try:
        response = req.app.state.chain.invoke(
            {"question": request.message},
            config={"configurable": {"session_id": request.session_id}},
        )
        return ChatResponse(response=response, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    _session_store.pop(session_id, None)
    return {"cleared": session_id}
