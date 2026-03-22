"""
Chat route - Handles incoming chat requests.

Integrates with IntentAgent for intent classification.
"""

from fastapi import APIRouter, HTTPException
from schemas.chat import ChatRequest, ChatResponse
from services.ChatService import get_chat_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint that processes user messages.

    Uses IntentAgent to classify the intent of the user message.

    Args:
        request: ChatRequest containing message, tenant_id, chatbot_id, and optional session_id

    Returns:
        ChatResponse with the agent's response
    """
    try:
        # Get chat service and process request
        chat_service = await get_chat_service()
        print(f"Received chat request: {request}")
        response =  await chat_service.process_chat_request(request)
        print(f"Generated chat response: {response}")
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )