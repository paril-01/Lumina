import os
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from sqlalchemy.orm import Session
from sqlalchemy import delete, and_

from database.database import SessionLocal
from models.activity_model import UserActivity
from models.user_model import User, UserSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_retention")

class DataRetentionService:
    """
    Service responsible for enforcing data retention policies.
    Automatically cleans up old raw activity data after ML processing.
    """
    
    def __init__(self, 
                 default_retention_hours: int = 24, 
                 check_interval_minutes: int = 60):
        self.default_retention_hours = default_retention_hours
        self.check_interval_minutes = check_interval_minutes
        self.running = False
        self.thread = None
        logger.info(f"Data retention service initialized (default: {default_retention_hours}h)")

    def start(self):
        """Start the data retention service background thread"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._retention_loop, daemon=True)
        self.thread.start()
        logger.info("Data retention service started")
    
    def stop(self):
        """Stop the data retention service"""
        self.running = False
        if self.thread:
            self.thread = None
        logger.info("Data retention service stopped")
    
    def _retention_loop(self):
        """Background thread that periodically cleans up old data"""
        while self.running:
            try:
                self._perform_cleanup()
            except Exception as e:
                logger.error(f"Error in data retention service: {str(e)}")
            
            # Sleep until next check interval
            for _ in range(self.check_interval_minutes * 60):
                if not self.running:
                    break
                time.sleep(1)
    
    def _perform_cleanup(self):
        """Perform the actual data cleanup"""
        logger.info("Running data retention cleanup")
        
        # Create a new database session
        db = SessionLocal()
        try:
            # Get all users with their retention settings
            users = db.query(User).filter(User.is_active == True).all()
            
            for user in users:
                # Get retention period for this user (default to 24 hours if not specified)
                retention_hours = self.default_retention_hours
                
                if user.settings and user.settings.data_retention_days:
                    # Convert days to hours, but use a minimum of 24 hours
                    retention_hours = max(user.settings.data_retention_days * 24, 24)
                
                # Calculate cutoff time
                cutoff_time = datetime.utcnow() - timedelta(hours=retention_hours)
                
                # Delete old activities
                result = db.query(UserActivity).filter(
                    and_(
                        UserActivity.user_id == user.id,
                        UserActivity.timestamp < cutoff_time
                    )
                ).delete(synchronize_session=False)
                
                # Log results
                if result > 0:
                    logger.info(f"Deleted {result} old activities for user {user.id}")
                
            # Commit the transaction
            db.commit()
            
        except Exception as e:
            logger.error(f"Error during data cleanup: {str(e)}")
        finally:
            db.close()
    
    def force_cleanup(self, user_id: int = None, retention_hours: int = None):
        """
        Force an immediate cleanup for testing or manual operation
        
        Args:
            user_id: Optional user ID to clean up for a specific user
            retention_hours: Override default retention period
        """
        logger.info(f"Forcing cleanup for user_id={user_id or 'all'}, retention={retention_hours or self.default_retention_hours}h")
        
        # Create a new database session
        db = SessionLocal()
        try:
            if user_id is not None:
                # Clean up for a specific user
                hours = retention_hours or self.default_retention_hours
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
                
                result = db.query(UserActivity).filter(
                    and_(
                        UserActivity.user_id == user_id,
                        UserActivity.timestamp < cutoff_time
                    )
                ).delete(synchronize_session=False)
                
                logger.info(f"Deleted {result} old activities for user {user_id}")
            else:
                # Clean up for all users
                self._perform_cleanup()
            
            # Commit the transaction
            db.commit()
            
        except Exception as e:
            logger.error(f"Error during forced cleanup: {str(e)}")
        finally:
            db.close()

# Global instance that can be imported and used across the application
retention_service = DataRetentionService()

# For manual testing
if __name__ == "__main__":
    # Start the service with a shorter interval for testing
    test_service = DataRetentionService(default_retention_hours=24, check_interval_minutes=1)
    test_service.start()
    
    try:
        print("Data retention service running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        test_service.stop()
        print("Data retention service stopped.")
