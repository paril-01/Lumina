from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from database.database import get_db
from models.user_model import UserProfile
from services.communication_clone import CommunicationClone
import os

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/communication",
    tags=["communication"],
    responses={404: {"description": "Not found"}},
)

# In-memory store for communication clone instances
# In a production system, this would be handled by a more robust solution
communication_clones = {}

def get_communication_clone(user_id: int) -> CommunicationClone:
    """Get or create a communication clone for a user"""
    if user_id not in communication_clones:
        communication_clones[user_id] = CommunicationClone(
            user_id=user_id,
            api_key=OPENAI_API_KEY
        )
    return communication_clones[user_id]

@router.get("/settings/{user_id}")
async def get_communication_settings(user_id: int, db: Session = Depends(get_db)):
    """Get communication clone settings for a user"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get communication clone
    clone = get_communication_clone(user_id)
    return clone.get_settings()

@router.put("/settings/{user_id}")
async def update_communication_settings(
    user_id: int, 
    settings: Dict[str, Any], 
    db: Session = Depends(get_db)
):
    """Update communication clone settings"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get and update communication clone
    clone = get_communication_clone(user_id)
    success = clone.update_settings(settings)
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to update settings")
    
    return {"status": "success", "settings": clone.get_settings()}

@router.get("/style_profile/{user_id}")
async def get_style_profile(user_id: int, db: Session = Depends(get_db)):
    """Get the user's communication style profile"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get communication clone and analyze style
    clone = get_communication_clone(user_id)
    return clone.analyze_style()

@router.post("/add_sample/{user_id}")
async def add_communication_sample(
    user_id: int,
    sample: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Add a communication sample for learning"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get communication clone and add sample
    clone = get_communication_clone(user_id)
    success = clone.add_communication_sample(sample)
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add sample")
    
    return {"status": "success", "message": "Communication sample added successfully"}

@router.post("/generate/{user_id}")
async def generate_text(
    user_id: int,
    request: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Generate text in the user's style"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get communication clone and generate text
    clone = get_communication_clone(user_id)
    
    prompt = request.get("prompt", "")
    context = request.get("context", "email_draft")
    max_length = request.get("max_length", 500)
    
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    # Generate text
    result = clone.generate_text(prompt, context, max_length)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result
