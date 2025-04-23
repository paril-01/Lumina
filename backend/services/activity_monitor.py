import os
import time
import json
import platform
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

# Optional: For keyboard and mouse monitoring
try:
    from pynput import keyboard, mouse
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

# Activity storage and processing
class ActivityMonitor:
    def __init__(self, callback: Callable = None, settings: Dict = None):
        """
        Initialize the activity monitor
        
        Args:
            callback: Function to call when activity is detected
            settings: Configuration settings for monitoring
        """
        self.callback = callback
        self.settings = settings or {
            "keyboard_monitoring": True,
            "mouse_monitoring": True,
            "application_monitoring": True,
            "active_window_monitoring": True,
            "sampling_rate": 1.0,  # seconds
            "privacy_filters": {
                "ignore_passwords": True,
                "ignore_private_browsing": True,
                "sensitive_applications": ["password manager", "banking"]
            }
        }
        
        self.running = False
        self.threads = []
        self.os_type = platform.system()  # 'Windows', 'Linux', 'Darwin' (macOS)
        self.current_application = None
        
        # Check if we can use low-level monitoring
        if not PYNPUT_AVAILABLE:
            print("Warning: pynput package not available. Some monitoring features will be limited.")
    
    def start(self):
        """Start the activity monitoring process"""
        if self.running:
            return
            
        self.running = True
        
        # Start monitoring threads based on settings and available packages
        if PYNPUT_AVAILABLE:
            if self.settings.get("keyboard_monitoring"):
                keyboard_thread = threading.Thread(target=self._monitor_keyboard, daemon=True)
                keyboard_thread.start()
                self.threads.append(keyboard_thread)
                
            if self.settings.get("mouse_monitoring"):
                mouse_thread = threading.Thread(target=self._monitor_mouse, daemon=True)
                mouse_thread.start()
                self.threads.append(mouse_thread)
        
        if self.settings.get("application_monitoring") or self.settings.get("active_window_monitoring"):
            app_thread = threading.Thread(target=self._monitor_applications, daemon=True)
            app_thread.start()
            self.threads.append(app_thread)
        
        print("Activity monitoring started")
    
    def stop(self):
        """Stop the activity monitoring process"""
        self.running = False
        # Let threads terminate naturally since they are daemon threads
        self.threads = []
        print("Activity monitoring stopped")
    
    def _monitor_keyboard(self):
        """Monitor keyboard activity"""
        def on_press(key):
            if not self.running:
                return False
                
            # Skip if in a privacy-filtered application
            if self._is_sensitive_application():
                return True
                
            # Process key press
            try:
                # Don't log the actual key content for privacy
                activity = {
                    "application": self.current_application,
                    "activity_type": "keyboard",
                    "action": "key_press",
                    "content": None,  # Don't store actual keystrokes for privacy
                    "activity_metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "is_special_key": not hasattr(key, 'char')
                    }
                }
                
                if self.callback:
                    self.callback(activity)
                    
            except Exception as e:
                print(f"Error processing keyboard event: {e}")
                
            return True
            
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
    
    def _monitor_mouse(self):
        """Monitor mouse activity"""
        def on_move(x, y):
            if not self.running:
                return False
                
            # Skip high-frequency events and only sample occasionally
            if hasattr(self, "_last_move_time") and time.time() - self._last_move_time < 0.5:
                return True
                
            self._last_move_time = time.time()
            
            # Skip if in a privacy-filtered application
            if self._is_sensitive_application():
                return True
                
            activity = {
                "application": self.current_application,
                "activity_type": "mouse",
                "action": "move",
                "content": None,
                "activity_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "position": {"x": x, "y": y}
                }
            }
            
            if self.callback:
                self.callback(activity)
                
            return True
            
        def on_click(x, y, button, pressed):
            if not self.running:
                return False
                
            # Skip if in a privacy-filtered application
            if self._is_sensitive_application():
                return True
                
            activity = {
                "application": self.current_application,
                "activity_type": "mouse",
                "action": "click",
                "content": None,
                "activity_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "position": {"x": x, "y": y},
                    "button": str(button),
                    "pressed": pressed
                }
            }
            
            if self.callback:
                self.callback(activity)
                
            return True
            
        with mouse.Listener(on_move=on_move, on_click=on_click) as listener:
            listener.join()
    
    def _monitor_applications(self):
        """Monitor active applications and windows"""
        # Platform-specific implementations would go here
        # This is a placeholder that simulates application monitoring
        
        while self.running:
            try:
                # In a real implementation, this would get the actual current application
                # For now, we just simulate it
                self.current_application = "Simulated Application"
                
                # Only log application changes
                activity = {
                    "application": self.current_application,
                    "activity_type": "system",
                    "action": "application_focus",
                    "content": None,
                    "activity_metadata": {
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                if self.callback:
                    self.callback(activity)
                    
                # Sleep to reduce CPU usage
                time.sleep(self.settings.get("sampling_rate", 1.0))
                
            except Exception as e:
                print(f"Error monitoring applications: {e}")
                time.sleep(5)  # Longer sleep on error
    
    def _is_sensitive_application(self) -> bool:
        """Check if the current application is in the sensitive list"""
        if not self.current_application:
            return False
            
        sensitive_apps = self.settings.get("privacy_filters", {}).get("sensitive_applications", [])
        return any(app.lower() in self.current_application.lower() for app in sensitive_apps)


# Example usage:
def activity_callback(activity_data):
    """Process detected activity"""
    print(f"Activity detected: {activity_data['activity_type']} in {activity_data['application']}")
    # In a real application, this would send to server or process locally

if __name__ == "__main__":
    # Simple test
    monitor = ActivityMonitor(callback=activity_callback)
    monitor.start()
    
    try:
        print("Activity monitoring is active. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
        print("Activity monitoring stopped.")
