from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Float, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime

from database.database import Base

class PrivacySettings(Base):
    """Privacy settings for a user"""
    __tablename__ = "privacy_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user_profiles.id"), unique=True)
    
    # Data collection controls
    data_collection_enabled = Column(Boolean, default=True)
    keyboard_monitoring = Column(Boolean, default=True)
    mouse_monitoring = Column(Boolean, default=True)
    screen_monitoring = Column(Boolean, default=False)
    application_monitoring = Column(Boolean, default=True)
    browser_history_monitoring = Column(Boolean, default=False)
    email_monitoring = Column(Boolean, default=False)
    calendar_monitoring = Column(Boolean, default=True)
    file_monitoring = Column(Boolean, default=False)
    
    # Data usage controls
    anonymize_sensitive_data = Column(Boolean, default=True)
    share_data_with_third_parties = Column(Boolean, default=False)
    
    # Retention settings
    retention_days = Column(Integer, default=30)
    
    # Relationships
    user = relationship("UserProfile", back_populates="privacy_settings")
    retention_policies = relationship("DataRetentionPolicy", back_populates="privacy_settings")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DataRetentionPolicy(Base):
    """Data retention policy for a specific data type"""
    __tablename__ = "data_retention_policies"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user_profiles.id"))
    privacy_settings_id = Column(Integer, ForeignKey("privacy_settings.id"))
    
    # Data type this policy applies to
    data_type = Column(String(50), index=True)
    
    # Retention periods (in days)
    retention_days = Column(Integer)  # How long to keep full data
    anonymize_after_days = Column(Integer, nullable=True)  # When to anonymize data (null = never)
    delete_after_days = Column(Integer, nullable=True)  # When to delete data completely (null = never)
    
    # Relationships
    privacy_settings = relationship("PrivacySettings", back_populates="retention_policies")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DataAccessLog(Base):
    """Log of data access events for audit and transparency"""
    __tablename__ = "data_access_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user_profiles.id"))
    
    # Access metadata
    operation = Column(String(50), index=True)  # e.g., "view", "update", "delete", "export"
    accessed_by = Column(Integer, index=True)  # User ID who performed the access
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Additional details
    details = Column(Text, nullable=True)
    
    # Relationships
    user = relationship("UserProfile", back_populates="access_logs")

# Update UserProfile model relationships (in user_model.py)
# The following is provided as a reference for how to update the UserProfile model
"""
class UserProfile(Base):
    # ... existing code ...
    
    # Add these relationships
    privacy_settings = relationship("PrivacySettings", back_populates="user", uselist=False)
    access_logs = relationship("DataAccessLog", back_populates="user")
"""
