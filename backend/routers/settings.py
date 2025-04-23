from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any

from database.database import get_db
from models.user_model import User, UserSettings, UserSettingsUpdate, UserSettingsResponse

router = APIRouter(
    prefix="/settings",
    tags=["settings"],
    responses={404: {"description": "Not found"}},
)

@router.get("/{user_id}", response_model=UserSettingsResponse)
async def get_settings(user_id: int, db: Session = Depends(get_db)):
    """Get settings for a user"""
    # Check if user exists
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get or create settings
    if not db_user.settings:
        # Create default settings
        db_settings = UserSettings(
            user_id=user_id,
            monitoring_enabled=True,
            learning_enabled=True,
            automation_enabled=False,
            privacy_level=2,
            data_retention_days=90,
            preferences={}
        )
        db.add(db_settings)
        db.commit()
        db.refresh(db_settings)
    else:
        db_settings = db_user.settings
    
    return db_settings

@router.patch("/{user_id}", response_model=UserSettingsResponse)
async def update_settings(user_id: int, settings_update: UserSettingsUpdate, db: Session = Depends(get_db)):
    """Update settings for a user"""
    # Check if user exists
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get or create settings
    if not db_user.settings:
        # Create settings with defaults first
        db_settings = UserSettings(
            user_id=user_id,
            monitoring_enabled=True,
            learning_enabled=True,
            automation_enabled=False,
            privacy_level=2,
            data_retention_days=90,
            preferences={}
        )
        db.add(db_settings)
        db.commit()
        db.refresh(db_settings)
    else:
        db_settings = db_user.settings
    
    # Update fields
    update_data = settings_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_settings, key, value)
    
    db.commit()
    db.refresh(db_settings)
    
    # Update service configurations if needed
    # This would update settings in the ActivityMonitor, LearningEngine, and TaskAutomator
    # services for this user, but we'll implement that later
    
    return db_settings

@router.get("/{user_id}/privacy", response_model=Dict[str, Any])
async def get_privacy_settings(user_id: int, db: Session = Depends(get_db)):
    """Get privacy-specific settings for a user"""
    # Check if user exists
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get settings
    settings = db_user.settings
    if not settings:
        raise HTTPException(status_code=404, detail="Settings not found")
    
    # Map privacy level to detailed settings
    privacy_settings = {
        "privacy_level": settings.privacy_level,
        "data_retention_days": settings.data_retention_days,
        "detailed_settings": {}
    }
    
    # Level 1: Minimal privacy (maximum data collection)
    if settings.privacy_level == 1:
        privacy_settings["detailed_settings"] = {
            "collect_keyboard_input": True,
            "collect_mouse_movements": True,
            "collect_application_usage": True,
            "store_text_content": True,
            "anonymize_sensitive_data": False,
            "share_data_for_improvement": True
        }
    # Level 2: Balanced privacy (default)
    elif settings.privacy_level == 2:
        privacy_settings["detailed_settings"] = {
            "collect_keyboard_input": True,
            "collect_mouse_movements": True,
            "collect_application_usage": True,
            "store_text_content": False,
            "anonymize_sensitive_data": True,
            "share_data_for_improvement": False
        }
    # Level 3: Maximum privacy
    else:
        privacy_settings["detailed_settings"] = {
            "collect_keyboard_input": False,
            "collect_mouse_movements": True,
            "collect_application_usage": True,
            "store_text_content": False,
            "anonymize_sensitive_data": True,
            "share_data_for_improvement": False
        }
    
    return privacy_settings

@router.post("/{user_id}/export-data", response_model=Dict[str, Any])
async def export_user_data(user_id: int, db: Session = Depends(get_db)):
    """Request data export for a user (GDPR compliance)"""
    # Check if user exists
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # This would generate a data export job and return a job ID
    # In a real implementation, this would be a background task
    
    return {
        "status": "pending",
        "message": "Data export request received",
        "job_id": f"export_{user_id}_{int(datetime.now().timestamp())}",
        "estimated_completion_time": "10 minutes"
    }

@router.post("/{user_id}/delete-data", status_code=status.HTTP_202_ACCEPTED)
async def delete_user_data(user_id: int, db: Session = Depends(get_db)):
    """Request deletion of all user data (GDPR compliance)"""
    # Check if user exists
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # This would start a data deletion process
    # In a real implementation, this would be a background task
    
    return {
        "status": "pending",
        "message": "Data deletion request received and will be processed"
    }

# Missing import that was needed
from datetime import datetime
