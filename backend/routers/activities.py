from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime, timedelta

from database.database import get_db
from models.activity_model import UserActivity, ActivityCreate, ActivityResponse
from models.user_model import User
from services.learning_engine import LearningEngine

router = APIRouter(
    prefix="/activities",
    tags=["activities"],
    responses={404: {"description": "Not found"}},
)

# Learning engines cache by user_id
learning_engines = {}

def get_learning_engine(user_id: int, db: Session = None):
    """Get or create a learning engine for a user"""
    if user_id not in learning_engines:
        # Get user settings
        if db:
            user = db.query(User).filter(User.id == user_id).first()
            if user and user.settings:
                settings = {
                    "learning_rate": 0.01,
                    "min_confidence_threshold": 0.7,
                    "min_pattern_occurrences": 3
                }
                # Could add more settings based on user preferences
                learning_engines[user_id] = LearningEngine(user_id, settings=settings)
            else:
                # Use default settings
                learning_engines[user_id] = LearningEngine(user_id)
        else:
            # Use default settings
            learning_engines[user_id] = LearningEngine(user_id)
    
    return learning_engines[user_id]

# Background task to process activities
def process_activity_in_background(activity_data: Dict[str, Any], user_id: int):
    """Process an activity in the background"""
    learning_engine = get_learning_engine(user_id)
    learning_engine.process_activity(activity_data)

@router.post("/", response_model=ActivityResponse, status_code=status.HTTP_201_CREATED)
async def create_activity(
    activity: ActivityCreate, 
    user_id: int, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new activity record"""
    # Check if user exists
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Create activity record
    db_activity = UserActivity(
        user_id=user_id,
        application=activity.application,
        activity_type=activity.activity_type,
        action=activity.action,
        content=activity.content,
        activity_metadata=activity.activity_metadata
    )
    
    db.add(db_activity)
    db.commit()
    db.refresh(db_activity)
    
    # Add background task to process this activity for learning
    activity_data = {
        "application": activity.application,
        "activity_type": activity.activity_type,
        "action": activity.action,
        "content": activity.content,
        "activity_metadata": activity.activity_metadata,
        "timestamp": datetime.now().isoformat()
    }
    background_tasks.add_task(process_activity_in_background, activity_data, user_id)
    
    return db_activity

@router.get("/", response_model=List[ActivityResponse])
async def read_activities(
    user_id: int, 
    skip: int = 0, 
    limit: int = 100, 
    activity_type: str = None,
    application: str = None,
    start_date: datetime = None,
    end_date: datetime = None,
    db: Session = Depends(get_db)
):
    """Get list of activities with optional filters"""
    # Check if user exists
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Build query with filters
    query = db.query(UserActivity).filter(UserActivity.user_id == user_id)
    
    if activity_type:
        query = query.filter(UserActivity.activity_type == activity_type)
    
    if application:
        query = query.filter(UserActivity.application == application)
    
    if start_date:
        query = query.filter(UserActivity.timestamp >= start_date)
    
    if end_date:
        query = query.filter(UserActivity.timestamp <= end_date)
    
    # Get results
    activities = query.order_by(UserActivity.timestamp.desc()).offset(skip).limit(limit).all()
    
    return activities

@router.get("/stats", response_model=Dict[str, Any])
async def get_activity_stats(user_id: int, days: int = 7, db: Session = Depends(get_db)):
    """Get activity statistics for a user"""
    # Check if user exists
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Build query for this time range
    query = db.query(UserActivity).filter(
        UserActivity.user_id == user_id,
        UserActivity.timestamp >= start_date,
        UserActivity.timestamp <= end_date
    )
    
    # Get count by activity type
    activity_type_counts = {}
    for activity_type in ["keyboard", "mouse", "system"]:
        count = query.filter(UserActivity.activity_type == activity_type).count()
        activity_type_counts[activity_type] = count
    
    # Get count by application
    application_counts = {}
    applications = db.query(UserActivity.application).filter(
        UserActivity.user_id == user_id,
        UserActivity.timestamp >= start_date,
        UserActivity.timestamp <= end_date
    ).distinct().all()
    
    for app in applications:
        app_name = app[0]
        count = query.filter(UserActivity.application == app_name).count()
        application_counts[app_name] = count
    
    # Get total count
    total_count = query.count()
    
    # Get learning progress
    learning_engine = get_learning_engine(user_id, db)
    learning_info = {
        "stage": learning_engine.learning_stage,
        "progress": learning_engine.learning_progress,
        "last_updated": learning_engine.last_updated.isoformat()
    }
    
    # Return stats
    return {
        "total_activities": total_count,
        "by_activity_type": activity_type_counts,
        "by_application": application_counts,
        "learning_status": learning_info,
        "time_range": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days": days
        }
    }

@router.get("/patterns", response_model=List[Dict[str, Any]])
async def get_detected_patterns(user_id: int, confidence_threshold: float = 0.7, db: Session = Depends(get_db)):
    """Get detected patterns for a user"""
    # Check if user exists
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get learning engine and detected patterns
    learning_engine = get_learning_engine(user_id, db)
    patterns = learning_engine.detect_patterns()
    
    # Filter by confidence threshold
    filtered_patterns = [
        pattern for pattern in patterns 
        if pattern.get("confidence", 0) >= confidence_threshold
    ]
    
    return filtered_patterns

@router.delete("/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_activities(
    user_id: int, 
    activity_type: str = None,
    application: str = None,
    start_date: datetime = None,
    end_date: datetime = None,
    db: Session = Depends(get_db)
):
    """Delete activities with optional filters"""
    # Check if user exists
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Build query with filters
    query = db.query(UserActivity).filter(UserActivity.user_id == user_id)
    
    if activity_type:
        query = query.filter(UserActivity.activity_type == activity_type)
    
    if application:
        query = query.filter(UserActivity.application == application)
    
    if start_date:
        query = query.filter(UserActivity.timestamp >= start_date)
    
    if end_date:
        query = query.filter(UserActivity.timestamp <= end_date)
    
    # Delete matching activities
    query.delete(synchronize_session=False)
    db.commit()
    
    return None
