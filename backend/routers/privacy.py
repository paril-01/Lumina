from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta

from database.database import get_db
from models.user_model import UserProfile
from models.privacy_model import PrivacySettings, DataRetentionPolicy, DataAccessLog

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/privacy",
    tags=["privacy"],
    responses={404: {"description": "Not found"}},
)

@router.get("/settings/{user_id}")
async def get_privacy_settings(user_id: int, db: Session = Depends(get_db)):
    """Get privacy settings for a user"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get or create privacy settings
    settings = db.query(PrivacySettings).filter(PrivacySettings.user_id == user_id).first()
    if not settings:
        # Create default settings
        settings = PrivacySettings(
            user_id=user_id,
            data_collection_enabled=True,
            keyboard_monitoring=True,
            mouse_monitoring=True,
            screen_monitoring=False,
            application_monitoring=True,
            browser_history_monitoring=False,
            email_monitoring=False,
            calendar_monitoring=True,
            file_monitoring=False,
            retention_days=30,
            anonymize_sensitive_data=True,
            share_data_with_third_parties=False
        )
        db.add(settings)
        db.commit()
        db.refresh(settings)
    
    return settings

@router.put("/settings/{user_id}")
async def update_privacy_settings(
    user_id: int, 
    settings: Dict[str, Any], 
    db: Session = Depends(get_db)
):
    """Update privacy settings for a user"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get existing settings
    db_settings = db.query(PrivacySettings).filter(PrivacySettings.user_id == user_id).first()
    if not db_settings:
        # Create with provided settings
        db_settings = PrivacySettings(user_id=user_id, **settings)
        db.add(db_settings)
    else:
        # Update existing settings
        for key, value in settings.items():
            if hasattr(db_settings, key):
                setattr(db_settings, key, value)
    
    # Log the update
    log_entry = DataAccessLog(
        user_id=user_id,
        operation="update_privacy_settings",
        accessed_by=user_id,  # Self-update
        timestamp=datetime.now(),
        details=f"Updated privacy settings: {', '.join(settings.keys())}"
    )
    db.add(log_entry)
    
    db.commit()
    db.refresh(db_settings)
    
    return {"status": "success", "settings": db_settings}

@router.get("/retention_policies/{user_id}")
async def get_retention_policies(user_id: int, db: Session = Depends(get_db)):
    """Get data retention policies for a user"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get retention policies
    policies = db.query(DataRetentionPolicy).filter(DataRetentionPolicy.user_id == user_id).all()
    
    # If no policies exist, create defaults
    if not policies:
        default_policies = [
            DataRetentionPolicy(
                user_id=user_id,
                data_type="activity_logs",
                retention_days=30,
                anonymize_after_days=60,
                delete_after_days=90
            ),
            DataRetentionPolicy(
                user_id=user_id,
                data_type="keyboard_data",
                retention_days=7,
                anonymize_after_days=15,
                delete_after_days=30
            ),
            DataRetentionPolicy(
                user_id=user_id,
                data_type="application_usage",
                retention_days=60,
                anonymize_after_days=90,
                delete_after_days=180
            ),
            DataRetentionPolicy(
                user_id=user_id,
                data_type="behavior_models",
                retention_days=365,
                anonymize_after_days=None,
                delete_after_days=None
            ),
        ]
        
        db.add_all(default_policies)
        db.commit()
        
        for policy in default_policies:
            db.refresh(policy)
        
        policies = default_policies
    
    return policies

@router.put("/retention_policies/{user_id}")
async def update_retention_policy(
    user_id: int, 
    policy_id: int, 
    policy_data: Dict[str, Any], 
    db: Session = Depends(get_db)
):
    """Update a specific data retention policy"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get policy
    policy = db.query(DataRetentionPolicy).filter(
        DataRetentionPolicy.id == policy_id,
        DataRetentionPolicy.user_id == user_id
    ).first()
    
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    
    # Update policy
    for key, value in policy_data.items():
        if hasattr(policy, key):
            setattr(policy, key, value)
    
    # Log the update
    log_entry = DataAccessLog(
        user_id=user_id,
        operation="update_retention_policy",
        accessed_by=user_id,  # Self-update
        timestamp=datetime.now(),
        details=f"Updated retention policy for {policy.data_type}"
    )
    db.add(log_entry)
    
    db.commit()
    db.refresh(policy)
    
    return {"status": "success", "policy": policy}

@router.post("/request_data_export/{user_id}")
async def request_data_export(
    user_id: int, 
    data_types: List[str], 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Request an export of user data"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Log the request
    log_entry = DataAccessLog(
        user_id=user_id,
        operation="request_data_export",
        accessed_by=user_id,  # Self-request
        timestamp=datetime.now(),
        details=f"Requested data export for types: {', '.join(data_types)}"
    )
    db.add(log_entry)
    db.commit()
    
    # In a real system, you would add this to a background task queue
    # background_tasks.add_task(export_user_data, user_id, data_types)
    
    return {
        "status": "success", 
        "message": "Data export requested",
        "estimated_completion": (datetime.now() + timedelta(hours=1)).isoformat()
    }

@router.post("/request_data_deletion/{user_id}")
async def request_data_deletion(
    user_id: int, 
    data_types: List[str], 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Request deletion of user data"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Log the request
    log_entry = DataAccessLog(
        user_id=user_id,
        operation="request_data_deletion",
        accessed_by=user_id,  # Self-request
        timestamp=datetime.now(),
        details=f"Requested data deletion for types: {', '.join(data_types)}"
    )
    db.add(log_entry)
    db.commit()
    
    # In a real system, you would add this to a background task queue
    # background_tasks.add_task(delete_user_data, user_id, data_types)
    
    return {
        "status": "success", 
        "message": "Data deletion requested",
        "estimated_completion": (datetime.now() + timedelta(hours=2)).isoformat()
    }

@router.get("/access_logs/{user_id}")
async def get_access_logs(
    user_id: int, 
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get logs of when user data was accessed"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get access logs
    logs = db.query(DataAccessLog).filter(
        DataAccessLog.user_id == user_id
    ).order_by(
        DataAccessLog.timestamp.desc()
    ).limit(limit).all()
    
    return logs
