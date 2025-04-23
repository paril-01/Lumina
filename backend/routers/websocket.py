from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from typing import Dict, List, Any
import json
import logging
import asyncio
from datetime import datetime
from pydantic import BaseModel

# Setup logging
logger = logging.getLogger("ai_assistant.websocket")

# Authentication scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Router
router = APIRouter(
    prefix="/ws",
    tags=["websocket"],
)

# Connection Manager
class ConnectionManager:
    def __init__(self):
        # client_id -> WebSocket connection
        self.active_connections: Dict[str, WebSocket] = {}
        # user_id -> set of client_ids
        self.user_connections: Dict[str, set] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str, user_id: str = None):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(client_id)
            logger.info(f"Client {client_id} connected for user {user_id}")
        else:
            logger.info(f"Anonymous client {client_id} connected")
    
    def disconnect(self, client_id: str, user_id: str = None):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        if user_id and user_id in self.user_connections:
            if client_id in self.user_connections[user_id]:
                self.user_connections[user_id].remove(client_id)
            if not self.user_connections[user_id]:  # If no more connections for user
                del self.user_connections[user_id]
            logger.info(f"Client {client_id} disconnected from user {user_id}")
        else:
            logger.info(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except Exception as e:
                logger.error(f"Error sending to client {client_id}: {str(e)}")
                # Remove dead connection
                del self.active_connections[client_id]
    
    async def broadcast_to_user(self, message: str, user_id: str):
        if user_id in self.user_connections:
            for client_id in list(self.user_connections[user_id]):
                await self.send_personal_message(message, client_id)
    
    async def broadcast(self, message: str):
        for client_id in list(self.active_connections.keys()):
            await self.send_personal_message(message, client_id)

    def get_user_connections_count(self, user_id: str) -> int:
        if user_id in self.user_connections:
            return len(self.user_connections[user_id])
        return 0


# Create a connection manager instance
manager = ConnectionManager()

# WebSocket endpoint
@router.websocket("/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str, user_id: str = None):
    await manager.connect(websocket, client_id, user_id)
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "client_id": client_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            # Wait for messages from the client
            data = await websocket.receive_text()
            
            try:
                # Try to parse as JSON
                message = json.loads(data)
                
                # Process message based on type
                message_type = message.get("type", "unknown")
                
                if message_type == "ping":
                    # Respond to ping with pong
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                elif message_type == "user_message":
                    # Process user message
                    # Here you would typically add logic to handle the message
                    # For now, we'll just echo it back
                    await websocket.send_json({
                        "type": "assistant_message",
                        "content": f"Echo: {message.get('content', '')}",
                        "timestamp": datetime.now().isoformat()
                    })
                elif message_type == "notification_read":
                    # Update read status for notifications
                    notification_id = message.get("notification_id")
                    if notification_id:
                        # Here you would update the notification status in your database
                        await websocket.send_json({
                            "type": "notification_updated",
                            "notification_id": notification_id,
                            "status": "read",
                            "timestamp": datetime.now().isoformat()
                        })
            except json.JSONDecodeError:
                # Not JSON, treat as plain text
                logger.warning(f"Received non-JSON message: {data[:100]}")
                await websocket.send_text(f"Received: {data}")
            except Exception as e:
                # Handle other errors
                logger.error(f"Error processing message: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Error processing your message",
                    "timestamp": datetime.now().isoformat()
                })
    except WebSocketDisconnect:
        manager.disconnect(client_id, user_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(client_id, user_id)

# API endpoints for sending messages through WebSockets
@router.post("/send/{user_id}")
async def send_to_user(user_id: str, message: Dict[str, Any]):
    """Send a message to all connections of a specific user"""
    if not manager.get_user_connections_count(user_id):
        return {"status": "error", "message": "User has no active connections"}
    
    await manager.broadcast_to_user(json.dumps(message), user_id)
    return {"status": "success", "connections": manager.get_user_connections_count(user_id)}

@router.post("/broadcast")
async def broadcast_message(message: Dict[str, Any]):
    """Broadcast a message to all active connections"""
    await manager.broadcast(json.dumps(message))
    return {"status": "success", "connections": len(manager.active_connections)}

# Methods to be called from other parts of the application
async def notify_user(user_id: str, notification_type: str, content: Any):
    """Send a notification to a user"""
    message = {
        "type": "notification",
        "notification_type": notification_type,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast_to_user(json.dumps(message), user_id)

async def send_activity_update(user_id: str, activity_data: Dict[str, Any]):
    """Send activity update to a user"""
    message = {
        "type": "activity_update",
        "activity": activity_data,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast_to_user(json.dumps(message), user_id)

async def send_learning_update(user_id: str, learning_data: Dict[str, Any]):
    """Send learning progress update to a user"""
    message = {
        "type": "learning_update",
        "learning": learning_data,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast_to_user(json.dumps(message), user_id)
