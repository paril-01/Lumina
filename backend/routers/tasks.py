from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime

from database.database import get_db
from models.activity_model import AutomatedTask, TaskExecutionLog, TaskBase, TaskCreate, TaskUpdate, TaskResponse, TaskExecutionLogResponse
from models.user_model import User
from services.task_automator import TaskAutomator

router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
    responses={404: {"description": "Not found"}},
)

# Task automators cache by user_id
task_automators = {}

def get_task_automator(user_id: int, db: Session = None):
    """Get or create a task automator for a user"""
    if user_id not in task_automators:
        # Get user settings
        if db:
            user = db.query(User).filter(User.id == user_id).first()
            if user and user.settings:
                settings = {
                    "automation_enabled": user.settings.automation_enabled,
                    "required_confidence": 0.85,
                    "confirmation_required": True,
                    "max_consecutive_actions": 5
                }
                task_automators[user_id] = TaskAutomator(user_id, settings=settings)
            else:
                # Use default settings
                task_automators[user_id] = TaskAutomator(user_id)
        else:
            # Use default settings
            task_automators[user_id] = TaskAutomator(user_id)
    
    return task_automators[user_id]

@router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(task: TaskCreate, user_id: int, db: Session = Depends(get_db)):
    """Create a new automated task"""
    # Check if user exists
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Create task in database
    db_task = AutomatedTask(
        user_id=user_id,
        task_name=task.task_name,
        task_type=task.task_type,
        trigger_conditions=task.trigger_conditions,
        actions=task.actions,
        is_active=task.is_active,
        confidence_level=70  # Default starting confidence
    )
    
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    
    # Add to task automator
    automator = get_task_automator(user_id, db)
    
    task_data = {
        "task_name": task.task_name,
        "task_type": task.task_type,
        "trigger_conditions": task.trigger_conditions,
        "actions": task.actions,
        "is_active": task.is_active == 1  # Convert to boolean
    }
    
    automator.add_task(task_data)
    
    return db_task

@router.get("/", response_model=List[TaskResponse])
async def read_tasks(
    user_id: int, 
    skip: int = 0, 
    limit: int = 100, 
    is_active: Optional[int] = None,
    task_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get list of tasks with optional filters"""
    # Check if user exists
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Build query with filters
    query = db.query(AutomatedTask).filter(AutomatedTask.user_id == user_id)
    
    if is_active is not None:
        query = query.filter(AutomatedTask.is_active == is_active)
    
    if task_type:
        query = query.filter(AutomatedTask.task_type == task_type)
    
    # Get results
    tasks = query.offset(skip).limit(limit).all()
    
    return tasks

@router.get("/{task_id}", response_model=TaskResponse)
async def read_task(task_id: int, db: Session = Depends(get_db)):
    """Get task by ID"""
    db_task = db.query(AutomatedTask).filter(AutomatedTask.id == task_id).first()
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return db_task

@router.patch("/{task_id}", response_model=TaskResponse)
async def update_task(task_id: int, task_update: TaskUpdate, db: Session = Depends(get_db)):
    """Update task details"""
    db_task = db.query(AutomatedTask).filter(AutomatedTask.id == task_id).first()
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Update task in database
    update_data = task_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_task, key, value)
    
    db_task.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_task)
    
    # Update in task automator
    automator = get_task_automator(db_task.user_id, db)
    
    task_data = {}
    if task_update.task_name is not None:
        task_data["task_name"] = task_update.task_name
    if task_update.trigger_conditions is not None:
        task_data["trigger_conditions"] = task_update.trigger_conditions
    if task_update.actions is not None:
        task_data["actions"] = task_update.actions
    if task_update.is_active is not None:
        task_data["is_active"] = task_update.is_active == 1  # Convert to boolean
    
    if task_data:
        automator.update_task(str(task_id), task_data)
    
    return db_task

@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(task_id: int, db: Session = Depends(get_db)):
    """Delete a task"""
    db_task = db.query(AutomatedTask).filter(AutomatedTask.id == task_id).first()
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Get user ID before deleting
    user_id = db_task.user_id
    
    # Delete from database
    db.delete(db_task)
    db.commit()
    
    # Delete from task automator
    automator = get_task_automator(user_id, db)
    automator.delete_task(str(task_id))
    
    return None

@router.post("/{task_id}/execute", response_model=Dict[str, Any])
async def execute_task(
    task_id: int, 
    confirmation: bool = True, 
    db: Session = Depends(get_db)
):
    """Execute a specific task"""
    db_task = db.query(AutomatedTask).filter(AutomatedTask.id == task_id).first()
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Get task automator
    automator = get_task_automator(db_task.user_id, db)
    
    # Execute task
    result = automator.execute_task(str(task_id), confirmation=confirmation)
    
    # If successful, log execution
    if result.get("success", False):
        # Update task activity_metadata
        db_task.execution_count += 1
        db_task.last_executed = datetime.utcnow()
        
        # Create execution log
        log = TaskExecutionLog(
            task_id=task_id,
            execution_time=datetime.utcnow(),
            status="success" if result.get("success", False) else "failure",
            execution_details=result.get("details"),
            error_message=result.get("error")
        )
        
        db.add(log)
        db.commit()
    
    return result

@router.get("/{task_id}/logs", response_model=List[TaskExecutionLogResponse])
async def get_task_logs(
    task_id: int, 
    skip: int = 0, 
    limit: int = 20, 
    db: Session = Depends(get_db)
):
    """Get execution logs for a task"""
    db_task = db.query(AutomatedTask).filter(AutomatedTask.id == task_id).first()
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Get logs
    logs = (
        db.query(TaskExecutionLog)
        .filter(TaskExecutionLog.task_id == task_id)
        .order_by(TaskExecutionLog.execution_time.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    
    return logs

@router.post("/start-monitoring", status_code=status.HTTP_200_OK)
async def start_task_monitoring(user_id: int, db: Session = Depends(get_db)):
    """Start monitoring for task triggers"""
    # Check if user exists
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if automation is enabled
    if db_user.settings and not db_user.settings.automation_enabled:
        raise HTTPException(status_code=400, detail="Automation is disabled in user settings")
    
    # Start monitoring
    automator = get_task_automator(user_id, db)
    automator.start_monitoring()
    
    return {"status": "Monitoring started"}

@router.post("/stop-monitoring", status_code=status.HTTP_200_OK)
async def stop_task_monitoring(user_id: int, db: Session = Depends(get_db)):
    """Stop monitoring for task triggers"""
    # Check if user exists
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Stop monitoring
    if user_id in task_automators:
        task_automators[user_id].stop_monitoring()
    
    return {"status": "Monitoring stopped"}
