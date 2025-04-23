from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from database.database import Base

# SQLAlchemy Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    profiles = relationship("UserProfile", back_populates="user")
    activities = relationship("UserActivity", back_populates="user")
    tasks = relationship("AutomatedTask", back_populates="user")
    settings = relationship("UserSettings", back_populates="user", uselist=False)

class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    profile_name = Column(String, index=True)
    behavioral_data = Column(JSON)  # Store learned behavior patterns
    communication_style = Column(JSON)  # Store communication style parameters
    usage_patterns = Column(JSON)  # Store app usage patterns
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="profiles")

class UserSettings(Base):
    __tablename__ = "user_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    monitoring_enabled = Column(Boolean, default=True)
    learning_enabled = Column(Boolean, default=True)
    automation_enabled = Column(Boolean, default=False)
    privacy_level = Column(Integer, default=2)  # 1: Minimal, 2: Balanced, 3: Maximum
    data_retention_days = Column(Integer, default=90)
    preferences = Column(JSON)  # Store user-specific preferences
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="settings")

# Pydantic Models (for API requests/responses)
class UserBase(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        orm_mode = True

class UserProfileResponse(BaseModel):
    id: int
    profile_name: str
    behavioral_data: Dict
    communication_style: Dict
    usage_patterns: Dict
    
    class Config:
        orm_mode = True

class UserSettingsUpdate(BaseModel):
    monitoring_enabled: Optional[bool] = None
    learning_enabled: Optional[bool] = None
    automation_enabled: Optional[bool] = None
    privacy_level: Optional[int] = Field(None, ge=1, le=3)
    data_retention_days: Optional[int] = Field(None, ge=1)
    preferences: Optional[Dict] = None

class UserSettingsResponse(BaseModel):
    id: int
    monitoring_enabled: bool
    learning_enabled: bool
    automation_enabled: bool
    privacy_level: int
    data_retention_days: int
    preferences: Dict
    
    class Config:
        orm_mode = True
