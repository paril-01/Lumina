from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel
from database.database import Base

# SQLAlchemy Models
class UserActivity(Base):
    __tablename__ = "user_activities"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    application = Column(String, index=True)  # Which application/program was used
    activity_type = Column(String, index=True)  # Type of activity (e.g., "keyboard", "mouse", "system")
    action = Column(String)  # What action was performed
    content = Column(Text, nullable=True)  # The content of the action if applicable
    activity_metadata = Column(JSON, nullable=True)  # Additional metadata about the activity
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", back_populates="activities")

class AutomatedTask(Base):
    __tablename__ = "automated_tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    task_name = Column(String, index=True)
    task_type = Column(String, index=True)  # Classification of the task
    trigger_conditions = Column(JSON)  # Conditions that trigger the task
    actions = Column(JSON)  # Actions to perform when triggered
    is_active = Column(Integer, default=1)  # 0: Disabled, 1: Enabled
    confidence_level = Column(Integer)  # 0-100 confidence in the automation
    execution_count = Column(Integer, default=0)  # How many times the task was executed
    last_executed = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="tasks")
    logs = relationship("TaskExecutionLog", back_populates="task")

class TaskExecutionLog(Base):
    __tablename__ = "task_execution_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("automated_tasks.id"))
    execution_time = Column(DateTime, default=datetime.utcnow)
    status = Column(String)  # "success", "failure", "partial"
    execution_details = Column(JSON, nullable=True)  # Details of what was done
    error_message = Column(String, nullable=True)
    
    # Relationships
    task = relationship("AutomatedTask", back_populates="logs")

# Pydantic Models (for API requests/responses)
class ActivityBase(BaseModel):
    application: str
    activity_type: str
    action: str
    content: Optional[str] = None
    activity_metadata: Optional[Dict[str, Any]] = None

class ActivityCreate(ActivityBase):
    pass

class ActivityResponse(ActivityBase):
    id: int
    user_id: int
    timestamp: datetime
    
    class Config:
        orm_mode = True

class TaskBase(BaseModel):
    task_name: str
    task_type: str
    trigger_conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    is_active: int = 1

class TaskCreate(TaskBase):
    pass

class TaskUpdate(BaseModel):
    task_name: Optional[str] = None
    trigger_conditions: Optional[Dict[str, Any]] = None
    actions: Optional[List[Dict[str, Any]]] = None
    is_active: Optional[int] = None

class TaskResponse(TaskBase):
    id: int
    user_id: int
    confidence_level: int
    execution_count: int
    last_executed: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class TaskExecutionLogResponse(BaseModel):
    id: int
    task_id: int
    execution_time: datetime
    status: str
    execution_details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    class Config:
        orm_mode = True
