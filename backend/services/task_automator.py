import os
import json
import time
import threading
import platform
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union

# Optional: For simulating keyboard/mouse actions
try:
    from pynput import keyboard, mouse
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

# Optional: For more advanced NLP capabilities
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class TaskAutomator:
    """
    TaskAutomator handles the execution of automated tasks based on learned patterns
    and user triggers.
    """
    
    def __init__(self, user_id: int, settings: Dict = None):
        """
        Initialize the task automator
        
        Args:
            user_id: ID of the user whose tasks are being automated
            settings: Configuration settings for automation
        """
        self.user_id = user_id
        self.settings = settings or {
            "automation_enabled": False,  # Default to disabled for safety
            "required_confidence": 0.85,  # Minimum confidence level to auto-execute
            "confirmation_required": True,  # Whether user confirmation is needed
            "max_consecutive_actions": 5,  # Safety limit for number of consecutive actions
            "allowed_applications": [],  # If empty, all non-restricted apps are allowed
            "restricted_applications": ["banking", "password", "finance"],
        }
        
        self.tasks = {}  # Dictionary of automated tasks indexed by ID
        self.active_tasks = set()  # Set of currently active task IDs
        self.execution_log = []  # History of task executions
        
        # For OS interaction
        self.os_type = platform.system()  # 'Windows', 'Linux', 'Darwin' (macOS)
        
        # Communication component for user output
        self.communication_handler = None
        
        # Flag for monitoring thread
        self.monitoring = False
        self.monitoring_thread = None
    
    def register_communication_handler(self, handler: Callable):
        """
        Register a function to handle communication with the user
        
        Args:
            handler: Function that takes messages to send to user
        """
        self.communication_handler = handler
    
    def add_task(self, task_data: Dict[str, Any]) -> str:
        """
        Add a new automated task
        
        Args:
            task_data: Dictionary with task definition
                {
                    "task_name": str,
                    "task_type": str,
                    "trigger_conditions": dict,
                    "actions": list,
                    "is_active": bool
                }
        
        Returns:
            task_id: ID of the created task
        """
        task_id = f"task_{len(self.tasks) + 1}"
        
        # Add activity_metadata
        task = {
            **task_data,
            "id": task_id,
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "execution_count": 0,
            "last_executed": None,
            "success_rate": 1.0  # Initial perfect success rate
        }
        
        # Validate task
        if not self._validate_task(task):
            raise ValueError("Invalid task configuration")
        
        # Store task
        self.tasks[task_id] = task
        
        # Add to active tasks if enabled
        if task.get("is_active"):
            self.active_tasks.add(task_id)
        
        return task_id
    
    def update_task(self, task_id: str, updated_data: Dict[str, Any]) -> bool:
        """
        Update an existing task
        
        Args:
            task_id: ID of the task to update
            updated_data: Dictionary with updated task data
        
        Returns:
            success: True if updated successfully
        """
        if task_id not in self.tasks:
            return False
        
        # Get existing task
        task = self.tasks[task_id]
        
        # Update fields
        for key, value in updated_data.items():
            if key in ["id", "user_id", "created_at", "execution_count", "last_executed"]:
                # Don't allow updating these fields
                continue
            task[key] = value
        
        # Update activity_metadata
        task["updated_at"] = datetime.now().isoformat()
        
        # Update active status
        if task.get("is_active"):
            self.active_tasks.add(task_id)
        else:
            self.active_tasks.discard(task_id)
        
        # Re-validate
        if not self._validate_task(task):
            # Revert to previous state if invalid
            self.tasks[task_id] = task
            return False
        
        # Store updated task
        self.tasks[task_id] = task
        return True
    
    def delete_task(self, task_id: str) -> bool:
        """
        Delete an existing task
        
        Args:
            task_id: ID of the task to delete
        
        Returns:
            success: True if deleted successfully
        """
        if task_id not in self.tasks:
            return False
        
        # Remove from active tasks
        self.active_tasks.discard(task_id)
        
        # Delete task
        del self.tasks[task_id]
        return True
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a task by ID
        
        Args:
            task_id: ID of the task to retrieve
        
        Returns:
            task: Task data dictionary
        """
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        Get all tasks for the current user
        
        Returns:
            tasks: List of task dictionaries
        """
        return list(self.tasks.values())
    
    def execute_task(self, task_id: str, confirmation: bool = None) -> Dict[str, Any]:
        """
        Execute a specific task
        
        Args:
            task_id: ID of the task to execute
            confirmation: Override for confirmation setting
        
        Returns:
            Dictionary with execution result:
                {
                    "success": bool,
                    "task_id": str,
                    "execution_time": str (ISO format),
                    "details": dict,
                    "error": str or None
                }
        """
        if task_id not in self.tasks:
            return {"status": "error", "message": f"Task {task_id} not found"}
        
        task = self.tasks[task_id]
        
        # Check if the task is active
        if not task.get("is_active", False):
            return {"status": "error", "message": f"Task {task_id} is not active"}
        
        # Check if confirmation is required
        confirmation_required = task.get("confirmation_required", self.settings["confirmation_required"])
        if confirmation_required and confirmation is None:
            return {"status": "confirmation_required", "task_id": task_id}
        
        # Execute task based on type
        task_type = task.get("task_type")
        execution_start = datetime.now()
        result = None
        error = None
        
        try:
            if task_type == "notification":
                result = self._execute_notification_task(task)
            elif task_type == "keyboard":
                result = self._execute_keyboard_task(task)
            elif task_type == "application":
                result = self._execute_application_task(task)
            elif task_type == "file":
                result = self._execute_file_task(task)
            elif task_type == "email":
                result = self._execute_email_task(task)
            elif task_type == "web":
                result = self._execute_web_task(task)
            elif task_type == "api":
                result = self._execute_api_task(task)
            elif task_type == "recipe":
                result = self._execute_recipe_task(task)
            else:
                error = f"Unknown task type: {task_type}"
        except Exception as e:
            error = str(e)
        
        # Record execution attempt
        execution_time = (datetime.now() - execution_start).total_seconds()
        execution_record = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "success": error is None,
            "error": error,
            "execution_time": execution_time,
            "result": result
        }
        
        # Update task statistics
        task["execution_count"] = task.get("execution_count", 0) + 1
        task["last_executed"] = datetime.now().isoformat()
        
        if error is None:
            task["success_rate"] = ((task.get("success_rate", 1.0) * (task["execution_count"] - 1)) + 1) / task["execution_count"]
        else:
            task["success_rate"] = ((task.get("success_rate", 1.0) * (task["execution_count"] - 1)) + 0) / task["execution_count"]
        
        # Add to execution log with limit
        self.execution_log.append(execution_record)
        if len(self.execution_log) > 1000:
            self.execution_log = self.execution_log[-1000:]
            
        # Return the execution result
        return {
            "success": error is None,
            "task_id": task_id,
            "execution_time": datetime.now().isoformat(),
            "details": result,
            "error": error
        }
    
    def _execute_task_actions(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the actions for a given task
        
        Args:
            task: Task dictionary
        
        Returns:
            result: Execution result
        """
        task_type = task.get("task_type", "")
        actions = task.get("actions", [])
        
        execution_results = {
            "success": True,
            "task_id": task.get("id", ""),
            "execution_time": datetime.now().isoformat(),
            "details": {
                "action_results": []
            },
            "error": None
        }
        
        # Safety check: limit number of actions
        if len(actions) > self.settings.get("max_consecutive_actions", 5):
            execution_results["success"] = False
            execution_results["error"] = f"Task exceeds maximum allowed actions ({self.settings.get('max_consecutive_actions', 5)})"
            return execution_results
        
        try:
            for action in actions:
                action_type = action.get("type", "")
                action_result = None
                
                if action_type == "keyboard":
                    action_result = self._execute_keyboard_action(action)
                elif action_type == "mouse":
                    action_result = self._execute_mouse_action(action)
                elif action_type == "application":
                    action_result = self._execute_application_action(action)
                elif action_type == "system":
                    action_result = self._execute_system_action(action)
                elif action_type == "text_generation":
                    action_result = self._execute_text_generation_action(action)
                else:
                    action_result = {
                        "success": False,
                        "error": f"Unknown action type: {action_type}"
                    }
                
                execution_results["details"]["action_results"].append(action_result)
                
                # Stop on failure
                if not action_result.get("success", False):
                    execution_results["success"] = False
                    execution_results["error"] = action_result.get("error", "Action failed")
                    break
                
                # Respect delay between actions
                if "delay_after" in action:
                    time.sleep(action["delay_after"])
        
        except Exception as e:
            execution_results["success"] = False
            execution_results["error"] = str(e)
        
        return execution_results
    
    def _execute_keyboard_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a keyboard action"""
        if not PYNPUT_AVAILABLE:
            return {"success": False, "error": "Keyboard automation not available"}
        
        try:
            action_subtype = action.get("subtype", "")
            
            # Check application restrictions
            target_app = action.get("application", "")
            if not self._is_allowed_application(target_app):
                return {"success": False, "error": f"Application not allowed: {target_app}"}
            
            kb = keyboard.Controller()
            
            if action_subtype == "type":
                text = action.get("text", "")
                kb.type(text)
                return {"success": True, "details": f"Typed {len(text)} characters"}
            
            elif action_subtype == "keypress":
                key_name = action.get("key", "")
                # Parse key
                if hasattr(keyboard.Key, key_name):
                    key = getattr(keyboard.Key, key_name)
                else:
                    key = key_name
                
                # Press and release
                kb.press(key)
                kb.release(key)
                return {"success": True, "details": f"Pressed key: {key_name}"}
            
            elif action_subtype == "hotkey":
                keys = action.get("keys", [])
                
                # Press all keys in sequence
                for key_name in keys:
                    if hasattr(keyboard.Key, key_name):
                        kb.press(getattr(keyboard.Key, key_name))
                    else:
                        kb.press(key_name)
                
                # Release all keys in reverse order
                for key_name in reversed(keys):
                    if hasattr(keyboard.Key, key_name):
                        kb.release(getattr(keyboard.Key, key_name))
                    else:
                        kb.release(key_name)
                
                return {"success": True, "details": f"Pressed hotkey: {'+'.join(keys)}"}
            
            else:
                return {"success": False, "error": f"Unknown keyboard action type: {action_subtype}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_mouse_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a mouse action"""
        if not PYNPUT_AVAILABLE:
            return {"success": False, "error": "Mouse automation not available"}
        
        try:
            action_subtype = action.get("subtype", "")
            
            # Check application restrictions
            target_app = action.get("application", "")
            if not self._is_allowed_application(target_app):
                return {"success": False, "error": f"Application not allowed: {target_app}"}
            
            m = mouse.Controller()
            
            if action_subtype == "move":
                x = action.get("x", 0)
                y = action.get("y", 0)
                m.position = (x, y)
                return {"success": True, "details": f"Moved to {x}, {y}"}
            
            elif action_subtype == "click":
                button_name = action.get("button", "left")
                
                if button_name == "left":
                    button = mouse.Button.left
                elif button_name == "right":
                    button = mouse.Button.right
                elif button_name == "middle":
                    button = mouse.Button.middle
                else:
                    return {"success": False, "error": f"Unknown button: {button_name}"}
                
                # Click
                m.click(button)
                return {"success": True, "details": f"Clicked {button_name} button"}
            
            elif action_subtype == "scroll":
                dx = action.get("dx", 0)
                dy = action.get("dy", 0)
                m.scroll(dx, dy)
                return {"success": True, "details": f"Scrolled {dx}, {dy}"}
            
            else:
                return {"success": False, "error": f"Unknown mouse action type: {action_subtype}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_application_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an application-related action"""
        try:
            action_subtype = action.get("subtype", "")
            app_name = action.get("application", "")
            
            # Check application restrictions
            if not self._is_allowed_application(app_name):
                return {"success": False, "error": f"Application not allowed: {app_name}"}
            
            if action_subtype == "launch":
                # Launch the application
                path = action.get("path", app_name)
                
                if self.os_type == "Windows":
                    subprocess.Popen(path, shell=True)
                elif self.os_type == "Darwin":  # macOS
                    subprocess.Popen(["open", "-a", path])
                else:  # Linux
                    subprocess.Popen(path, shell=True)
                
                return {"success": True, "details": f"Launched {app_name}"}
            
            elif action_subtype == "close":
                # Close the application
                if self.os_type == "Windows":
                    subprocess.Popen(f"taskkill /f /im {app_name}.exe", shell=True)
                elif self.os_type == "Darwin":  # macOS
                    subprocess.Popen(["osascript", "-e", f'tell application "{app_name}" to quit'])
                else:  # Linux
                    subprocess.Popen(["pkill", app_name])
                
                return {"success": True, "details": f"Closed {app_name}"}
            
            else:
                return {"success": False, "error": f"Unknown application action type: {action_subtype}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_system_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a system-level action"""
        try:
            action_subtype = action.get("subtype", "")
            
            if action_subtype == "notification":
                # Display a notification
                title = action.get("title", "Notification")
                message = action.get("message", "")
                
                # Send to communication handler
                if self.communication_handler:
                    self.communication_handler({
                        "type": "notification",
                        "title": title,
                        "message": message
                    })
                
                return {"success": True, "details": f"Notification: {title}"}
            
            elif action_subtype == "schedule":
                # Schedule a task
                # This would integrate with the system scheduler or task manager
                # Placeholder for implementation
                return {"success": True, "details": "Task scheduled"}
            
            else:
                return {"success": False, "error": f"Unknown system action type: {action_subtype}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_text_generation_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a text generation action using an AI model"""
        if not OPENAI_AVAILABLE:
            return {"success": False, "error": "Text generation not available"}
        
        try:
            action_subtype = action.get("subtype", "completion")
            prompt = action.get("prompt", "")
            model = action.get("model", "gpt-3.5-turbo")
            max_tokens = action.get("max_tokens", 100)
            temperature = action.get("temperature", 0.7)
            
            if not prompt:
                return {"success": False, "error": "Prompt is required for text generation"}
            
            if action_subtype == "completion":
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                generated_text = response.choices[0].text.strip()
                
            elif action_subtype == "chat":
                messages = action.get("messages", [{"role": "user", "content": prompt}])
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                generated_text = response.choices[0].message.content.strip()
                
            else:
                return {"success": False, "error": f"Unknown text generation subtype: {action_subtype}"}
            
            return {
                "success": True,
                "details": {
                    "generated_text": generated_text,
                    "model": model
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_api_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an API task"""
        try:
            import requests
            
            actions = task.get("actions", [])
            execution_results = {
                "success": True,
                "details": {
                    "action_results": []
                }
            }
            
            for action in actions:
                method = action.get("method", "GET").upper()
                url = action.get("url", "")
                headers = action.get("headers", {})
                params = action.get("params", {})
                data = action.get("data", {})
                json_data = action.get("json", None)
                timeout = action.get("timeout", 30)
                
                if not url:
                    action_result = {"success": False, "error": "URL is required for API action"}
                else:
                    try:
                        response = requests.request(
                            method=method,
                            url=url,
                            headers=headers,
                            params=params,
                            data=data,
                            json=json_data,
                            timeout=timeout
                        )
                        
                        response.raise_for_status()  # Raise exception for 4XX/5XX responses
                        
                        try:
                            response_data = response.json()
                        except:
                            response_data = response.text
                        
                        action_result = {
                            "success": True,
                            "details": {
                                "status_code": response.status_code,
                                "response": response_data
                            }
                        }
                    except Exception as e:
                        action_result = {"success": False, "error": str(e)}
                
                execution_results["details"]["action_results"].append(action_result)
                
                if not action_result["success"]:
                    execution_results["success"] = False
                    execution_results["error"] = action_result.get("error", "API action failed")
                    break
            
            return execution_results
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_recipe_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a recipe task"""
        recipe_id = task.get("recipe_id")
        
        if not recipe_id:
            return {"success": False, "error": "Recipe ID is required"}
        
        try:
            # This would typically involve calling into a RecipeTaskConnector
            # Here we're just simulating successful execution
            return {
                "success": True,
                "details": {
                    "recipe_id": recipe_id,
                    "message": f"Recipe {recipe_id} executed successfully"
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_email_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an email task"""
        try:
            actions = task.get("actions", [])
            execution_results = {
                "success": True,
                "details": {
                    "action_results": []
                }
            }
            
            for action in actions:
                action_subtype = action.get("subtype", "")
                
                if action_subtype == "send":
                    to = action.get("to", [])
                    cc = action.get("cc", [])
                    bcc = action.get("bcc", [])
                    subject = action.get("subject", "")
                    body = action.get("body", "")
                    attachments = action.get("attachments", [])
                    
                    if not to:
                        action_result = {"success": False, "error": "Recipient is required for email"}
                    else:
                        # In a real implementation, this would connect to an email service
                        # Here we're just simulating success
                        action_result = {
                            "success": True,
                            "details": {
                                "to": to,
                                "subject": subject,
                                "message": "Email sending simulated successfully"
                            }
                        }
                else:
                    action_result = {"success": False, "error": f"Unknown email action type: {action_subtype}"}
                
                execution_results["details"]["action_results"].append(action_result)
                
                if not action_result["success"]:
                    execution_results["success"] = False
                    execution_results["error"] = action_result.get("error", "Email action failed")
                    break
            
            return execution_results
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_web_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a web-related task"""
        try:
            actions = task.get("actions", [])
            execution_results = {
                "success": True,
                "details": {
                    "action_results": []
                }
            }
            
            for action in actions:
                action_subtype = action.get("subtype", "")
                
                if action_subtype == "open_url":
                    url = action.get("url", "")
                    
                    if not url:
                        action_result = {"success": False, "error": "URL is required"}
                    else:
                        # In a real implementation, this would open a browser
                        # Here we're just simulating success
                        action_result = {
                            "success": True,
                            "details": {
                                "url": url,
                                "message": f"Opening URL: {url}"
                            }
                        }
                else:
                    action_result = {"success": False, "error": f"Unknown web action type: {action_subtype}"}
                
                execution_results["details"]["action_results"].append(action_result)
                
                if not action_result["success"]:
                    execution_results["success"] = False
                    execution_results["error"] = action_result.get("error", "Web action failed")
                    break
            
            return execution_results
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_file_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a file-related task"""
        try:
            actions = task.get("actions", [])
            execution_results = {
                "success": True,
                "details": {
                    "action_results": []
                }
            }
            
            for action in actions:
                action_subtype = action.get("subtype", "")
                
                if action_subtype == "create":
                    path = action.get("path", "")
                    content = action.get("content", "")
                    
                    if not path:
                        action_result = {"success": False, "error": "File path is required"}
                    else:
                        # Safety check - restrict to user's documents or similar safe location
                        # This would need proper implementation in a real system
                        action_result = {
                            "success": True,
                            "details": {
                                "path": path,
                                "message": f"File creation simulated: {path}"
                            }
                        }
                        
                elif action_subtype == "read":
                    path = action.get("path", "")
                    
                    if not path:
                        action_result = {"success": False, "error": "File path is required"}
                    else:
                        # Safety check - restrict to user's documents or similar safe location
                        action_result = {
                            "success": True,
                            "details": {
                                "path": path,
                                "message": f"File reading simulated: {path}",
                                "content": "Simulated file content"
                            }
                        }
                        
                else:
                    action_result = {"success": False, "error": f"Unknown file action type: {action_subtype}"}
                
                execution_results["details"]["action_results"].append(action_result)
                
                if not action_result["success"]:
                    execution_results["success"] = False
                    execution_results["error"] = action_result.get("error", "File action failed")
                    break
            
            return execution_results
            
        except Exception as e:
            return {"success": False, "error": str(e)}
        
        try:
            action_subtype = action.get("subtype", "")
            
            if action_subtype == "compose":
                # Generate text based on a prompt
                prompt = action.get("prompt", "")
                style = action.get("style", "default")
                
                # Configure OpenAI API
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    return {"success": False, "error": "OpenAI API key not found"}
                
                openai.api_key = api_key
                
                # Adjust the prompt based on the style
                if style != "default" and "style_template" in action:
                    full_prompt = action["style_template"].format(prompt=prompt)
                else:
                    full_prompt = prompt
                
                # Generate text
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": f"You are writing in the style of the user. Use their typical communication style."},
                            {"role": "user", "content": full_prompt}
                        ],
                        max_tokens=action.get("max_tokens", 150),
                        temperature=action.get("temperature", 0.7)
                    )
                    
                    generated_text = response.choices[0].message.content
                    
                    # Send to communication handler
                    if self.communication_handler:
                        self.communication_handler({
                            "type": "generated_text",
                            "text": generated_text,
                            "prompt": prompt
                        })
                    
                    return {
                        "success": True, 
                        "details": "Text generated",
                        "text": generated_text
                    }
                
                except Exception as e:
                    return {"success": False, "error": f"Text generation error: {str(e)}"}
            
            else:
                return {"success": False, "error": f"Unknown text generation action type: {action_subtype}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def start_monitoring(self):
        """Start the background task monitoring thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop the background task monitoring"""
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread = None
    
    def _monitoring_loop(self):
        """Background thread for monitoring triggers and executing tasks"""
        while self.monitoring:
            # Check each active task for triggers
            for task_id in self.active_tasks:
                task = self.tasks.get(task_id)
                
                if not task:
                    continue
                
                # Check if task meets trigger conditions
                if self._check_task_triggers(task):
                    # Execute the task
                    self.execute_task(task_id)
            
            # Sleep to prevent high CPU usage
            time.sleep(1)
    
    def _check_task_triggers(self, task: Dict[str, Any]) -> bool:
        """
        Check if a task's trigger conditions are met
        
        Args:
            task: Task dictionary
        
        Returns:
            triggered: True if triggers are met
        """
        # For demo purposes, this is a placeholder
        # In a real implementation, this would check time, events, etc.
        return False
    
    def _validate_task(self, task: Dict[str, Any]) -> bool:
        """
        Validate a task configuration
        
        Args:
            task: Task dictionary
        
        Returns:
            valid: True if valid
        """
        # Basic validation
        required_fields = ["task_name", "task_type", "trigger_conditions", "actions"]
        for field in required_fields:
            if field not in task or not task[field]:
                return False
        
        # Validate actions
        actions = task.get("actions", [])
        if not isinstance(actions, list) or not actions:
            return False
        
        # Check each action
        for action in actions:
            if "type" not in action:
                return False
            
            action_type = action["type"]
            if action_type not in ["keyboard", "mouse", "application", "system", "text_generation"]:
                return False
        
        return True
    
    def _is_allowed_application(self, app_name: str) -> bool:
        """
        Check if an application is allowed for automation
        
        Args:
            app_name: Name of the application
        
        Returns:
            allowed: True if allowed
        """
        # Check restricted applications
        restricted = self.settings.get("restricted_applications", [])
        for restricted_app in restricted:
            if restricted_app.lower() in app_name.lower():
                return False
        
        # Check allowed applications
        allowed = self.settings.get("allowed_applications", [])
        if not allowed:
            # If no specific apps are allowed, all non-restricted apps are allowed
            return True
        
        for allowed_app in allowed:
            if allowed_app.lower() in app_name.lower():
                return True
        
        return False
    
    def _log_execution(self, task_id: str, result: Dict[str, Any]):
        """
        Log a task execution
        
        Args:
            task_id: ID of the executed task
            result: Execution result
        """
        log_entry = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "success": result.get("success", False),
            "details": result.get("details"),
            "error": result.get("error")
        }
        
        # Add to log
        self.execution_log.append(log_entry)
        
        # Limit log size
        if len(self.execution_log) > 1000:
            self.execution_log = self.execution_log[-1000:]

# Example usage:
if __name__ == "__main__":
    # Simple test
    automator = TaskAutomator(user_id=1)
    
    # Register communication handler
    def print_message(message):
        print(f"Message: {message}")
    
    automator.register_communication_handler(print_message)
    
    # Add a notification task
    task_id = automator.add_task({
        "task_name": "Test Notification",
        "task_type": "notification",
        "trigger_conditions": {
            "type": "manual"
        },
        "actions": [
            {
                "type": "system",
                "subtype": "notification",
                "title": "Test Notification",
                "message": "This is a test notification from the AI Assistant."
            }
        ],
        "is_active": True
    })
    
    # Execute the task
    result = automator.execute_task(task_id, confirmation=False)
    print(f"Task execution result: {result['success']}")
    if not result['success']:
        print(f"Error: {result['error']}")
