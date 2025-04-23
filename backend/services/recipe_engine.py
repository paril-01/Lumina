import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecipeEngine:
    """
    RecipeEngine handles the execution of automation recipes and provides
    a building block system for creating custom automations.
    """
    
    def __init__(self, user_id: int):
        """Initialize the recipe engine"""
        self.user_id = user_id
        self.recipes = {}  # Stored recipes
        self.recipe_templates = self._init_templates()
        self.execution_history = []
        self.handlers = {}  # Action handlers for different action types
        
        # Register built-in action handlers
        self._register_default_handlers()
    
    def _init_templates(self) -> Dict[str, Any]:
        """Initialize built-in recipe templates"""
        return {
            "daily_reminder": {
                "name": "Daily Reminder",
                "description": "Send a reminder at a specified time every day",
                "template_id": "daily_reminder",
                "trigger": {
                    "type": "schedule",
                    "schedule_type": "daily",
                    "time": "09:00"
                },
                "actions": [
                    {
                        "type": "notification",
                        "title": "Daily Reminder",
                        "message": "Your reminder message here"
                    }
                ],
                "parameters": [
                    {
                        "name": "reminder_time",
                        "display_name": "Reminder Time",
                        "type": "time",
                        "default": "09:00"
                    },
                    {
                        "name": "reminder_message",
                        "display_name": "Message",
                        "type": "text",
                        "default": "Remember to do this task"
                    }
                ]
            },
            "application_launcher": {
                "name": "Application Launcher",
                "description": "Launch an application at a specific time or event",
                "template_id": "application_launcher",
                "trigger": {
                    "type": "schedule",
                    "schedule_type": "once",
                    "time": "09:00"
                },
                "actions": [
                    {
                        "type": "application",
                        "action": "launch",
                        "application_path": "",
                        "arguments": ""
                    }
                ],
                "parameters": [
                    {
                        "name": "application_path",
                        "display_name": "Application Path",
                        "type": "file_path",
                        "default": ""
                    },
                    {
                        "name": "trigger_type",
                        "display_name": "Trigger Type",
                        "type": "select",
                        "options": ["schedule", "event", "manual"],
                        "default": "manual"
                    }
                ]
            },
            "file_backup": {
                "name": "File Backup",
                "description": "Create a backup copy of specified files or directories",
                "template_id": "file_backup",
                "trigger": {
                    "type": "schedule",
                    "schedule_type": "weekly",
                    "day": "monday",
                    "time": "23:00"
                },
                "actions": [
                    {
                        "type": "file",
                        "action": "copy",
                        "source_path": "",
                        "destination_path": ""
                    }
                ],
                "parameters": [
                    {
                        "name": "source_path",
                        "display_name": "Source Path",
                        "type": "directory_path",
                        "default": ""
                    },
                    {
                        "name": "destination_path",
                        "display_name": "Backup Location",
                        "type": "directory_path",
                        "default": ""
                    },
                    {
                        "name": "schedule_frequency",
                        "display_name": "Backup Frequency",
                        "type": "select",
                        "options": ["daily", "weekly", "monthly"],
                        "default": "weekly"
                    }
                ]
            },
            "email_forwarder": {
                "name": "Email Forwarder",
                "description": "Forward emails matching criteria to another address",
                "template_id": "email_forwarder",
                "trigger": {
                    "type": "event",
                    "event_type": "email_received",
                    "conditions": {
                        "from_contains": "",
                        "subject_contains": ""
                    }
                },
                "actions": [
                    {
                        "type": "email",
                        "action": "forward",
                        "to_address": "",
                        "add_prefix": "FWD: "
                    }
                ],
                "parameters": [
                    {
                        "name": "from_filter",
                        "display_name": "From Contains",
                        "type": "text",
                        "default": ""
                    },
                    {
                        "name": "subject_filter",
                        "display_name": "Subject Contains",
                        "type": "text",
                        "default": ""
                    },
                    {
                        "name": "forward_to",
                        "display_name": "Forward To",
                        "type": "email",
                        "default": ""
                    }
                ]
            },
            "text_expander": {
                "name": "Text Expander",
                "description": "Expand text shortcuts into full phrases",
                "template_id": "text_expander",
                "trigger": {
                    "type": "event",
                    "event_type": "text_pattern",
                    "pattern": ""
                },
                "actions": [
                    {
                        "type": "keyboard",
                        "action": "replace_text",
                        "replacement_text": ""
                    }
                ],
                "parameters": [
                    {
                        "name": "shortcut",
                        "display_name": "Text Shortcut",
                        "type": "text",
                        "default": ""
                    },
                    {
                        "name": "expansion",
                        "display_name": "Expanded Text",
                        "type": "text",
                        "default": ""
                    }
                ]
            }
        }
    
    def _register_default_handlers(self):
        """Register default action handlers"""
        # Notification actions
        self.register_handler("notification", self._handle_notification_action)
        
        # Application actions
        self.register_handler("application", self._handle_application_action)
        
        # File actions
        self.register_handler("file", self._handle_file_action)
        
        # Email actions
        self.register_handler("email", self._handle_email_action)
        
        # Keyboard actions
        self.register_handler("keyboard", self._handle_keyboard_action)
        
        # Web actions
        self.register_handler("web", self._handle_web_action)
        
        # API actions
        self.register_handler("api", self._handle_api_action)
    
    def register_handler(self, action_type: str, handler: Callable):
        """Register a handler for a specific action type"""
        self.handlers[action_type] = handler
        logger.info(f"Registered handler for action type: {action_type}")
    
    def get_template(self, template_id: str) -> Dict[str, Any]:
        """Get a recipe template by ID"""
        return self.recipe_templates.get(template_id, None)
    
    def get_all_templates(self) -> List[Dict[str, Any]]:
        """Get all available recipe templates"""
        return list(self.recipe_templates.values())
    
    def create_recipe(self, recipe_data: Dict[str, Any]) -> str:
        """Create a new recipe from scratch or based on a template"""
        recipe_id = f"recipe_{len(self.recipes) + 1}"
        
        # Check if using a template
        if "template_id" in recipe_data:
            template_id = recipe_data.pop("template_id")
            template = self.get_template(template_id)
            
            if template:
                # Apply template
                recipe = {**template}
                # Override with custom values
                for key, value in recipe_data.items():
                    if key == "parameters":
                        # Apply parameter values to trigger and actions
                        self._apply_parameters(recipe, value)
                    else:
                        recipe[key] = value
            else:
                # Template not found, use recipe_data as is
                recipe = recipe_data
        else:
            # Creating recipe from scratch
            recipe = recipe_data
        
        # Add metadata
        recipe["id"] = recipe_id
        recipe["user_id"] = self.user_id
        recipe["created_at"] = datetime.now().isoformat()
        recipe["updated_at"] = datetime.now().isoformat()
        recipe["is_active"] = recipe.get("is_active", True)
        recipe["execution_count"] = 0
        
        # Store recipe
        self.recipes[recipe_id] = recipe
        
        # If recipe is active, schedule or register it based on trigger type
        if recipe.get("is_active", True):
            self._register_recipe_trigger(recipe)
        
        return recipe_id
    
    def update_recipe(self, recipe_id: str, updated_data: Dict[str, Any]) -> bool:
        """Update an existing recipe"""
        if recipe_id not in self.recipes:
            return False
        
        # Get existing recipe
        recipe = self.recipes[recipe_id]
        
        # Check if active status changed
        was_active = recipe.get("is_active", True)
        will_be_active = updated_data.get("is_active", was_active)
        
        # Update fields
        for key, value in updated_data.items():
            if key in ["id", "user_id", "created_at", "execution_count"]:
                # Don't allow updating these fields
                continue
            elif key == "parameters":
                # Apply parameter values to trigger and actions
                self._apply_parameters(recipe, value)
            else:
                recipe[key] = value
        
        # Update metadata
        recipe["updated_at"] = datetime.now().isoformat()
        
        # Store updated recipe
        self.recipes[recipe_id] = recipe
        
        # If active status changed or trigger changed, update registration
        if was_active != will_be_active or "trigger" in updated_data:
            # Unregister old trigger if it was active
            if was_active:
                self._unregister_recipe_trigger(recipe_id)
            
            # Register new trigger if it will be active
            if will_be_active:
                self._register_recipe_trigger(recipe)
        
        return True
    
    def delete_recipe(self, recipe_id: str) -> bool:
        """Delete an existing recipe"""
        if recipe_id not in self.recipes:
            return False
        
        # Get recipe to check if active
        recipe = self.recipes[recipe_id]
        
        # Unregister trigger if recipe was active
        if recipe.get("is_active", True):
            self._unregister_recipe_trigger(recipe_id)
        
        # Delete recipe
        del self.recipes[recipe_id]
        return True
    
    def get_recipe(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        """Get a recipe by ID"""
        return self.recipes.get(recipe_id)
    
    def get_all_recipes(self) -> List[Dict[str, Any]]:
        """Get all recipes for the current user"""
        return list(self.recipes.values())
    
    def execute_recipe(self, recipe_id: str) -> Dict[str, Any]:
        """Execute a recipe immediately"""
        if recipe_id not in self.recipes:
            return {"status": "error", "message": f"Recipe {recipe_id} not found"}
        
        recipe = self.recipes[recipe_id]
        
        # Execute actions
        result = self._execute_recipe_actions(recipe)
        
        # Record execution
        self._record_execution(recipe_id, result)
        
        return result
    
    def _apply_parameters(self, recipe: Dict[str, Any], parameters: Dict[str, Any]):
        """Apply parameter values to a recipe's trigger and actions"""
        # This is a simplified implementation
        # In a full implementation, you would use a template language or token replacement
        
        # Convert recipe to string
        recipe_str = json.dumps(recipe)
        
        # Replace parameter tokens
        for param_name, param_value in parameters.items():
            recipe_str = recipe_str.replace(f"${{{param_name}}}", str(param_value))
        
        # Convert back to dict and update recipe
        updated_recipe = json.loads(recipe_str)
        recipe.update(updated_recipe)
    
    def _register_recipe_trigger(self, recipe: Dict[str, Any]):
        """Register a recipe's trigger"""
        # This is a simplified implementation
        # In a real system, you would register with a scheduler or event system
        trigger = recipe.get("trigger", {})
        trigger_type = trigger.get("type")
        
        if trigger_type == "schedule":
            # Register with a scheduler (simplified)
            logger.info(f"Scheduled recipe {recipe['id']} with trigger {trigger}")
        elif trigger_type == "event":
            # Register with an event system (simplified)
            logger.info(f"Registered recipe {recipe['id']} for event {trigger}")
        elif trigger_type == "manual":
            # Manual triggers don't need registration
            pass
        else:
            logger.warning(f"Unknown trigger type: {trigger_type}")
    
    def _unregister_recipe_trigger(self, recipe_id: str):
        """Unregister a recipe's trigger"""
        # This is a simplified implementation
        # In a real system, you would unregister from a scheduler or event system
        logger.info(f"Unregistered recipe {recipe_id}")
    
    def _execute_recipe_actions(self, recipe: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all actions in a recipe"""
        actions = recipe.get("actions", [])
        
        # Initialize results
        results = []
        success = True
        error = None
        
        # Execute each action in sequence
        for i, action in enumerate(actions):
            action_type = action.get("type")
            
            # Skip action if no handler or invalid type
            if not action_type or action_type not in self.handlers:
                results.append({
                    "index": i,
                    "type": action_type,
                    "success": False,
                    "error": f"No handler for action type: {action_type}"
                })
                success = False
                continue
            
            try:
                # Execute action with appropriate handler
                handler = self.handlers[action_type]
                result = handler(action)
                
                # Add to results
                results.append({
                    "index": i,
                    "type": action_type,
                    "success": result.get("success", False),
                    "result": result
                })
                
                # If action failed, mark recipe execution as failed
                if not result.get("success", False):
                    success = False
                    error = result.get("error")
                    
                    # Check if should continue on error
                    continue_on_error = recipe.get("continue_on_error", False)
                    if not continue_on_error:
                        break
            
            except Exception as e:
                # Handle exceptions
                success = False
                error = str(e)
                results.append({
                    "index": i,
                    "type": action_type,
                    "success": False,
                    "error": str(e)
                })
                
                # Check if should continue on error
                continue_on_error = recipe.get("continue_on_error", False)
                if not continue_on_error:
                    break
        
        # Update recipe execution count
        recipe["execution_count"] = recipe.get("execution_count", 0) + 1
        
        # Return overall result
        return {
            "recipe_id": recipe["id"],
            "success": success,
            "error": error,
            "action_results": results
        }
    
    def _record_execution(self, recipe_id: str, result: Dict[str, Any]):
        """Record recipe execution for history and analytics"""
        execution_record = {
            "recipe_id": recipe_id,
            "timestamp": datetime.now().isoformat(),
            "success": result.get("success", False),
            "error": result.get("error"),
            "action_results": result.get("action_results", [])
        }
        
        # Add to execution history with limit
        self.execution_history.append(execution_record)
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    # Action handlers
    def _handle_notification_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Handle notification actions"""
        title = action.get("title", "Notification")
        message = action.get("message", "")
        
        # In a real system, you would use system notifications or a notification service
        logger.info(f"NOTIFICATION: {title} - {message}")
        
        return {"success": True, "message": f"Notification sent: {title}"}
    
    def _handle_application_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Handle application actions"""
        app_action = action.get("action", "launch")
        app_path = action.get("application_path", "")
        arguments = action.get("arguments", "")
        
        if not app_path:
            return {"success": False, "error": "No application path specified"}
        
        # In a real system, you would use subprocess or platform-specific methods
        logger.info(f"APPLICATION: {app_action} - {app_path} {arguments}")
        
        return {"success": True, "message": f"Application {app_action}: {app_path}"}
    
    def _handle_file_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file actions"""
        file_action = action.get("action", "")
        source_path = action.get("source_path", "")
        destination_path = action.get("destination_path", "")
        
        if not source_path:
            return {"success": False, "error": "No source path specified"}
        
        # In a real system, you would use file operations like copy, move, etc.
        logger.info(f"FILE: {file_action} - {source_path} to {destination_path}")
        
        return {"success": True, "message": f"File {file_action} completed"}
    
    def _handle_email_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Handle email actions"""
        email_action = action.get("action", "")
        to_address = action.get("to_address", "")
        subject = action.get("subject", "")
        body = action.get("body", "")
        
        if not to_address:
            return {"success": False, "error": "No recipient specified"}
        
        # In a real system, you would use an email library or service
        logger.info(f"EMAIL: {email_action} - To: {to_address}, Subject: {subject}")
        
        return {"success": True, "message": f"Email {email_action} completed"}
    
    def _handle_keyboard_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Handle keyboard actions"""
        kb_action = action.get("action", "")
        text = action.get("text", "")
        keys = action.get("keys", [])
        
        # In a real system, you would use keyboard simulation
        logger.info(f"KEYBOARD: {kb_action} - {'Text: ' + text if text else 'Keys: ' + str(keys)}")
        
        return {"success": True, "message": f"Keyboard {kb_action} completed"}
    
    def _handle_web_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Handle web actions"""
        web_action = action.get("action", "")
        url = action.get("url", "")
        
        if not url:
            return {"success": False, "error": "No URL specified"}
        
        # In a real system, you would use web automation or browser control
        logger.info(f"WEB: {web_action} - {url}")
        
        return {"success": True, "message": f"Web {web_action} completed"}
    
    def _handle_api_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API actions"""
        method = action.get("method", "GET")
        url = action.get("url", "")
        headers = action.get("headers", {})
        body = action.get("body", {})
        
        if not url:
            return {"success": False, "error": "No URL specified"}
        
        # In a real system, you would use HTTP requests
        logger.info(f"API: {method} {url}")
        
        return {"success": True, "message": f"API {method} request completed"}
