from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging

from services.external_llm import external_llm_service
from database.database import get_db
from sqlalchemy.orm import Session

# Setup logging
logger = logging.getLogger("ai_assistant.llm")

router = APIRouter(
    prefix="/llm",
    tags=["llm"],
    responses={404: {"description": "Not found"}},
)

class LLMGenerateRequest(BaseModel):
    prompt: str
    provider: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.7
    system_message: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

class LLMGenerateResponse(BaseModel):
    text: str
    provider: Optional[str] = None
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
@router.get("/providers")
async def list_providers():
    """List available LLM providers"""
    providers = external_llm_service.get_available_providers()
    return {
        "providers": providers,
        "default": external_llm_service.default_provider,
        "status": "available" if providers else "no_providers_available"
    }

@router.get("/models")
async def list_models(provider: Optional[str] = None):
    """List available models for a provider or all providers"""
    models = external_llm_service.list_available_models(provider)
    return {
        "models": models,
        "provider": provider or "all",
        "count": len(models)
    }

@router.post("/generate", response_model=LLMGenerateResponse)
async def generate_text(request: LLMGenerateRequest):
    """Generate text using an external LLM"""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
        
    # Prepare kwargs based on provider
    kwargs = request.options or {}
    if request.system_message:
        if request.provider == "anthropic":
            kwargs["system"] = request.system_message
        elif request.provider == "openai":
            kwargs["messages"] = [
                {"role": "system", "content": request.system_message},
                {"role": "user", "content": request.prompt}
            ]
    
    try:
        result = await external_llm_service.generate_text(
            prompt=request.prompt,
            provider=request.provider,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            **kwargs
        )
        return result
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat(
    user_id: int,
    message: str,
    conversation_id: Optional[str] = None,
    system_message: Optional[str] = None,
    db: Session = Depends(get_db),
    provider: Optional[str] = None,
    model: Optional[str] = None
):
    """Chat with an AI assistant"""
    # Check if provider and model are available
    if provider and provider not in external_llm_service.get_available_providers():
        providers = external_llm_service.get_available_providers()
        if providers:
            provider = providers[0]
        else:
            raise HTTPException(status_code=400, detail="No LLM providers available")
    
    try:
        # Here we would normally store the conversation history and retrieve it for context
        # For simplicity, we'll just use the single message
        result = await external_llm_service.generate_text(
            prompt=message,
            provider=provider,
            model=model,
            system_message=system_message or "You are a helpful assistant."
        )
        
        return {
            "message": result["text"],
            "conversation_id": conversation_id or "new_conversation",
            "provider": result.get("provider"),
            "model": result.get("model")
        }
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
