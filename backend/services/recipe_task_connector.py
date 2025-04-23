import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .recipe_engine import RecipeEngine
from .task_automator import TaskAutomator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecipeTaskConnector:
    """
    Connector class that bridges RecipeEngine and TaskAutomator,
    allowing recipes to be executed as tasks and task definitions
    to be created from recipes.
    """
    
    def __init__(self, user_id: int, task_automator: TaskAutomator, recipe_engine: RecipeEngine):
        """Initialize the connector"""
        self.user_id = user_id
        self.task_automator = task_automator
        self.recipe_engine = recipe_engine
    
    def create_task_from_recipe(self, recipe_id: str) -> Optional[str]:
        """
        Create a task in the TaskAutomator based on a recipe
        
        Args:
            recipe_id: ID of the recipe to convert to a task
            
        Returns:
            task_id: ID of the created task, or None if failed
        """
        # Get recipe
        recipe = self.recipe_engine.get_recipe(recipe_id)
        if not recipe:
            logger.error(f"Recipe {recipe_id} not found")
            return None
        
        # Convert recipe to task format
        task_data = self._convert_recipe_to_task(recipe)
        
        # Create task
        try:
            task_id = self.task_automator.add_task(task_data)
            logger.info(f"Created task {task_id} from recipe {recipe_id}")
            return task_id
        except Exception as e:
            logger.error(f"Error creating task from recipe {recipe_id}: {e}")
            return None
    
    def execute_recipe_as_task(self, recipe_id: str) -> Dict[str, Any]:
        """
        Execute a recipe through the TaskAutomator
        
        Args:
            recipe_id: ID of the recipe to execute
            
        Returns:
            result: Dictionary with execution status and details
        """
        # Get recipe
        recipe = self.recipe_engine.get_recipe(recipe_id)
        if not recipe:
            return {"status": "error", "message": f"Recipe {recipe_id} not found"}
        
        # Create temporary task
        task_data = self._convert_recipe_to_task(recipe)
        task_data["is_temporary"] = True  # Mark as temporary so it can be removed after execution
        
        try:
            # Add task
            task_id = self.task_automator.add_task(task_data)
            
            # Execute task
            result = self.task_automator.execute_task(task_id)
            
            # Remove temporary task
            if task_data.get("is_temporary", False):
                self.task_automator.delete_task(task_id)
            
            return result
        except Exception as e:
            logger.error(f"Error executing recipe {recipe_id} as task: {e}")
            return {"status": "error", "message": str(e)}
    
    def _convert_recipe_to_task(self, recipe: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a recipe to a task format for TaskAutomator
        
        Args:
            recipe: Recipe data dictionary
            
        Returns:
            task_data: Task data dictionary
        """
        # Extract recipe information
        recipe_id = recipe.get("id", "")
        recipe_name = recipe.get("name", "Unnamed Recipe")
        trigger = recipe.get("trigger", {})
        actions = recipe.get("actions", [])
        
        # Basic task information
        task_data = {
            "task_name": f"Recipe: {recipe_name}",
            "task_type": "recipe",
            "description": recipe.get("description", ""),
            "is_active": recipe.get("is_active", True),
            "confirmation_required": recipe.get("confirmation_required", True),
            "recipe_id": recipe_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        # Convert trigger
        trigger_type = trigger.get("type", "manual")
        if trigger_type == "schedule":
            task_data["trigger_conditions"] = {
                "type": "schedule",
                "schedule_type": trigger.get("schedule_type", "once"),
                "time": trigger.get("time", ""),
                "day": trigger.get("day", ""),
                "date": trigger.get("date", "")
            }
        elif trigger_type == "event":
            task_data["trigger_conditions"] = {
                "type": "event",
                "event_type": trigger.get("event_type", ""),
                "conditions": trigger.get("conditions", {})
            }
        else:  # Manual trigger
            task_data["trigger_conditions"] = {
                "type": "manual"
            }
        
        # Store actions
        task_data["actions"] = actions
        
        return task_data
    
    def sync_recipes_and_tasks(self) -> Dict[str, Any]:
        """
        Synchronize recipes and tasks to keep them in sync
        
        Returns:
            result: Dictionary with synchronization results
        """
        created_count = 0
        updated_count = 0
        errors = []
        
        # Get all recipes
        recipes = self.recipe_engine.get_all_recipes()
        
        # Get all tasks
        tasks = self.task_automator.get_all_tasks()
        
        # Find tasks that reference recipes
        recipe_tasks = {}
        for task in tasks:
            recipe_id = task.get("recipe_id")
            if recipe_id:
                recipe_tasks[recipe_id] = task
        
        # Update existing tasks and create new ones
        for recipe in recipes:
            recipe_id = recipe.get("id")
            
            if recipe_id in recipe_tasks:
                # Recipe has a corresponding task, update it
                task = recipe_tasks[recipe_id]
                task_id = task.get("id")
                
                # Convert recipe to task format
                updated_task_data = self._convert_recipe_to_task(recipe)
                
                # Update task
                try:
                    success = self.task_automator.update_task(task_id, updated_task_data)
                    if success:
                        updated_count += 1
                    else:
                        errors.append(f"Failed to update task {task_id} for recipe {recipe_id}")
                except Exception as e:
                    errors.append(f"Error updating task for recipe {recipe_id}: {e}")
            else:
                # Recipe doesn't have a task, create one
                try:
                    task_id = self.create_task_from_recipe(recipe_id)
                    if task_id:
                        created_count += 1
                    else:
                        errors.append(f"Failed to create task for recipe {recipe_id}")
                except Exception as e:
                    errors.append(f"Error creating task for recipe {recipe_id}: {e}")
        
        return {
            "synchronized": True,
            "created": created_count,
            "updated": updated_count,
            "errors": errors
        }
