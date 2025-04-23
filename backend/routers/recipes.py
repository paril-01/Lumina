from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from database.database import get_db
from models.user_model import UserProfile
from services.recipe_engine import RecipeEngine
from services.task_automator import TaskAutomator
from services.recipe_task_connector import RecipeTaskConnector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/recipes",
    tags=["recipes"],
    responses={404: {"description": "Not found"}},
)

# In-memory store for recipe engines and connectors
# In a production system, this would be handled by a more robust solution
recipe_engines = {}
task_automators = {}
recipe_task_connectors = {}

def get_recipe_engine(user_id: int) -> RecipeEngine:
    """Get or create a recipe engine for a user"""
    if user_id not in recipe_engines:
        recipe_engines[user_id] = RecipeEngine(user_id=user_id)
    return recipe_engines[user_id]

def get_task_automator(user_id: int) -> TaskAutomator:
    """Get or create a task automator for a user"""
    if user_id not in task_automators:
        task_automators[user_id] = TaskAutomator(user_id=user_id)
    return task_automators[user_id]

def get_recipe_task_connector(user_id: int) -> RecipeTaskConnector:
    """Get or create a recipe-task connector for a user"""
    if user_id not in recipe_task_connectors:
        recipe_task_connectors[user_id] = RecipeTaskConnector(
            user_id=user_id,
            task_automator=get_task_automator(user_id),
            recipe_engine=get_recipe_engine(user_id)
        )
    return recipe_task_connectors[user_id]

@router.get("/templates/{user_id}")
async def get_recipe_templates(user_id: int, db: Session = Depends(get_db)):
    """Get available recipe templates"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get recipe engine and templates
    recipe_engine = get_recipe_engine(user_id)
    templates = recipe_engine.get_all_templates()
    
    return templates

@router.get("/template/{user_id}/{template_id}")
async def get_recipe_template(
    user_id: int,
    template_id: str,
    db: Session = Depends(get_db)
):
    """Get a specific recipe template"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get recipe engine and template
    recipe_engine = get_recipe_engine(user_id)
    template = recipe_engine.get_template(template_id)
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return template

@router.get("/all/{user_id}")
async def get_user_recipes(user_id: int, db: Session = Depends(get_db)):
    """Get all recipes for a user"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get recipe engine and recipes
    recipe_engine = get_recipe_engine(user_id)
    recipes = recipe_engine.get_all_recipes()
    
    return recipes

@router.get("/{user_id}/{recipe_id}")
async def get_recipe(
    user_id: int,
    recipe_id: str,
    db: Session = Depends(get_db)
):
    """Get a specific recipe"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get recipe engine and recipe
    recipe_engine = get_recipe_engine(user_id)
    recipe = recipe_engine.get_recipe(recipe_id)
    
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")
    
    return recipe

@router.post("/create/{user_id}")
async def create_recipe(
    user_id: int,
    recipe_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Create a new recipe"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get recipe engine and create recipe
    recipe_engine = get_recipe_engine(user_id)
    
    try:
        recipe_id = recipe_engine.create_recipe(recipe_data)
        
        # Create a task from the recipe
        connector = get_recipe_task_connector(user_id)
        task_id = connector.create_task_from_recipe(recipe_id)
        
        return {
            "status": "success",
            "recipe_id": recipe_id,
            "task_id": task_id
        }
    except Exception as e:
        logger.error(f"Error creating recipe: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/{user_id}/{recipe_id}")
async def update_recipe(
    user_id: int,
    recipe_id: str,
    recipe_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Update an existing recipe"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get recipe engine and update recipe
    recipe_engine = get_recipe_engine(user_id)
    
    try:
        success = recipe_engine.update_recipe(recipe_id, recipe_data)
        
        if not success:
            raise HTTPException(status_code=404, detail="Recipe not found or update failed")
        
        # Sync with tasks
        connector = get_recipe_task_connector(user_id)
        sync_result = connector.sync_recipes_and_tasks()
        
        return {
            "status": "success",
            "sync_result": sync_result
        }
    except Exception as e:
        logger.error(f"Error updating recipe: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{user_id}/{recipe_id}")
async def delete_recipe(
    user_id: int,
    recipe_id: str,
    db: Session = Depends(get_db)
):
    """Delete a recipe"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get recipe engine and delete recipe
    recipe_engine = get_recipe_engine(user_id)
    
    try:
        success = recipe_engine.delete_recipe(recipe_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Recipe not found or delete failed")
        
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error deleting recipe: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/execute/{user_id}/{recipe_id}")
async def execute_recipe(
    user_id: int,
    recipe_id: str,
    db: Session = Depends(get_db)
):
    """Execute a recipe"""
    # Check if user exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get connector and execute recipe
    connector = get_recipe_task_connector(user_id)
    
    try:
        result = connector.execute_recipe_as_task(recipe_id)
        return result
    except Exception as e:
        logger.error(f"Error executing recipe: {e}")
        raise HTTPException(status_code=400, detail=str(e))
