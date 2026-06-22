import time
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
    print(f"[api/chat] User: {request.message[:60]}{'...' if len(request.message) > 60 else ''}")
    t0 = time.perf_counter()
    try:
        response = req.app.state.chain.invoke(
            {"question": request.message},
            config={"configurable": {"session_id": request.session_id}},
        )
        elapsed = time.perf_counter() - t0
        print(f"[api/chat] Clone: {response[:60]}{'...' if len(response) > 60 else ''}")
        print(f"[api/chat] Done in {elapsed:.1f}s")
        return ChatResponse(response=response, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    _session_store.pop(session_id, None)
    return {"cleared": session_id}
