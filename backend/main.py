from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import json
import os
import platform
import logging
from datetime import datetime

# Import local modules
from models.user_model import UserProfile
from services.activity_monitor import ActivityMonitor
from services.learning_engine import LearningEngine
from services.task_automator import TaskAutomator
from services.recipe_engine import RecipeEngine
from services.communication_clone import CommunicationClone
from services.data_retention import retention_service
from database.database import get_db, SessionLocal, Base, engine
from routers import users, activities, tasks, settings, privacy, communication, recipes, websocket, llm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ai_assistant")

# Initialize FastAPI app
app = FastAPI(
    title="Personal AI Assistant API",
    description="Backend API for a learning AI assistant that automates tasks",
    version="0.1.0"
)

# Configure CORS
origins = [
    "http://localhost:3000",
    "https://ai-assistant-frontend.vercel.app",  # Add your Vercel deployment URL here
    "https://lumina-ai.vercel.app"  # Add any additional deployment domains here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(users.router)
app.include_router(activities.router)
app.include_router(tasks.router)
app.include_router(settings.router)
app.include_router(privacy.router)
app.include_router(communication.router)
app.include_router(recipes.router)
app.include_router(websocket.router)
app.include_router(llm.router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Personal AI Assistant API",
        "status": "running",
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat(),
        "system": platform.system(),
        "platform": platform.platform()
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    # Check if we have any LLM providers available
    from services.external_llm import external_llm_service
    llm_providers = external_llm_service.get_available_providers()
    
    return {
        "status": "healthy",
        "services": {
            "data_retention": retention_service.running,
            "database": "connected",
            "llm": {
                "status": "available" if llm_providers else "unavailable",
                "providers": llm_providers
            },
            "websocket": "available"
        },
        "timestamp": datetime.now().isoformat()
    }

# Initialize services
@app.on_event("startup")
async def startup_event():
    # Create database tables if they don't exist
    logger.info("Creating database tables if they don't exist")
    Base.metadata.create_all(bind=engine)
    
    # Start data retention service (automatically cleans up old activity data)
    logger.info("Starting data retention service")
    retention_service.start()
    
    logger.info("AI Assistant backend started successfully")

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    # Stop data retention service
    logger.info("Stopping data retention service")
    retention_service.stop()
    
    logger.info("AI Assistant backend shutdown completed")

# Manual cleanup endpoint (for testing and maintenance)
@app.post("/admin/cleanup")
async def force_cleanup(user_id: Optional[int] = None, retention_hours: Optional[int] = None):
    logger.info(f"Manual cleanup requested for user_id={user_id}, retention_hours={retention_hours}")
    retention_service.force_cleanup(user_id, retention_hours)
    return {"status": "cleanup triggered"}

if __name__ == "__main__":
    # Run the API server
    logger.info("Starting AI Assistant backend server")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
