from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.chatbot_service import ChatbotService

router = APIRouter()
_service = ChatbotService()


class ChatRequest(BaseModel):
    conversation_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    reply: str
    sources: list
    conversation_id: str


class HistoryResponse(BaseModel):
    conversation_id: str
    history: list


@router.post("/", response_model=ChatResponse, summary="Multi-turn chatbot")
def chat(body: ChatRequest):
    """
    Send a message in a conversation. Pass the same conversation_id across turns
    to maintain history. Sources are book records retrieved via RAG.
    """
    try:
        result = _service.chat(
            conversation_id=body.conversation_id,
            message=body.message,
        )
    except Exception as e:
        msg = str(e)
        if "rate limit" in msg.lower():
            raise HTTPException(status_code=429, detail=msg)
        raise HTTPException(status_code=503, detail=f"AI provider error: {msg}")

    return ChatResponse(
        reply=result["reply"],
        sources=result.get("sources", []),
        conversation_id=body.conversation_id,
    )


@router.get("/{conversation_id}/history", response_model=HistoryResponse, summary="Get conversation history")
def get_history(conversation_id: str):
    """
    Retrieve the full message history for a conversation.
    Returns an empty history list if the conversation_id is unknown.
    """
    history = _service.get_conversation_history(conversation_id)
    return HistoryResponse(conversation_id=conversation_id, history=history)


@router.delete("/{conversation_id}", status_code=204, summary="Clear conversation history")
def clear_conversation(conversation_id: str):
    """
    Delete a conversation's history from memory.
    Subsequent messages using the same conversation_id will start fresh.
    """
    _service.conversation.pop(conversation_id, None)
