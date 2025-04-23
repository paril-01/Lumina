import { v4 as uuidv4 } from 'uuid';

class WebSocketService {
  constructor() {
    this.socket = null;
    this.clientId = localStorage.getItem('lumina_client_id') || uuidv4();
    this.connected = false;
    this.messageQueue = [];
    this.listeners = new Map();
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 2000; // Start with 2 seconds
    this.reconnectTimer = null;
    this.pingInterval = null;
    this.lastPingResponse = null;
    
    // Save client ID
    localStorage.setItem('lumina_client_id', this.clientId);
  }

  connect(userId = null, authToken = null) {
    if (this.socket) {
      this.disconnect();
    }

    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = process.env.REACT_APP_API_URL || 'localhost:8000';
      let wsUrl = `${protocol}//${host.replace(/^https?:\/\//, '')}/ws/${this.clientId}`;
      
      // Add user ID if provided
      if (userId) {
        wsUrl += `?user_id=${userId}`;
      }

      this.socket = new WebSocket(wsUrl);

      this.socket.onopen = () => {
        console.log('WebSocket connected');
        this.connected = true;
        this.reconnectAttempts = 0;
        
        // Process any messages in the queue
        this.processQueue();
        
        // Start ping interval
        this.startPingInterval();
        
        // Notify listeners
        this.notifyListeners('connect', { connected: true });
      };

      this.socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          // Handle pong responses
          if (message.type === 'pong') {
            this.lastPingResponse = new Date();
            return;
          }
          
          console.log('WebSocket message received:', message);
          
          // Notify listeners based on message type
          this.notifyListeners(message.type, message);
          
          // Also notify general message listeners
          this.notifyListeners('message', message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.socket.onclose = (event) => {
        this.connected = false;
        this.stopPingInterval();
        
        console.log(`WebSocket closed. Code: ${event.code}, Reason: ${event.reason}`);
        
        // Notify listeners
        this.notifyListeners('disconnect', { 
          code: event.code, 
          reason: event.reason 
        });
        
        // Try to reconnect if not closed cleanly
        if (event.code !== 1000) {
          this.scheduleReconnect();
        }
      };

      this.socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        
        // Notify listeners
        this.notifyListeners('error', { error });
      };
      
      return true;
    } catch (error) {
      console.error('Error connecting to WebSocket:', error);
      return false;
    }
  }

  disconnect() {
    if (this.socket) {
      this.stopPingInterval();
      clearTimeout(this.reconnectTimer);
      
      // Close with a normal closure code
      this.socket.close(1000, 'User initiated disconnect');
      this.socket = null;
      this.connected = false;
    }
  }

  sendMessage(message) {
    if (typeof message === 'object') {
      message = JSON.stringify(message);
    }
    
    if (this.connected && this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(message);
      return true;
    } else {
      // Queue message for later
      this.messageQueue.push(message);
      
      // If not connected, try to connect
      if (!this.socket || this.socket.readyState === WebSocket.CLOSED) {
        this.connect();
      }
      return false;
    }
  }

  processQueue() {
    while (this.messageQueue.length > 0 && this.connected) {
      const message = this.messageQueue.shift();
      this.sendMessage(message);
    }
  }

  startPingInterval() {
    this.stopPingInterval(); // Clear any existing interval
    
    this.pingInterval = setInterval(() => {
      if (this.connected) {
        this.sendMessage({ type: 'ping', timestamp: new Date().toISOString() });
      }
    }, 30000); // Ping every 30 seconds
  }

  stopPingInterval() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  scheduleReconnect() {
    clearTimeout(this.reconnectTimer);
    
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      const delay = this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts);
      console.log(`Scheduling WebSocket reconnect in ${delay}ms`);
      
      this.reconnectTimer = setTimeout(() => {
        console.log(`Attempting WebSocket reconnect (${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`);
        this.reconnectAttempts += 1;
        this.connect();
      }, delay);
    } else {
      console.log('Maximum WebSocket reconnect attempts reached');
      this.notifyListeners('reconnect_failed', {
        attempts: this.reconnectAttempts
      });
    }
  }

  // Add event listener
  addEventListener(eventType, callback) {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, new Set());
    }
    this.listeners.get(eventType).add(callback);
    
    return {
      remove: () => {
        if (this.listeners.has(eventType)) {
          this.listeners.get(eventType).delete(callback);
        }
      }
    };
  }

  // Remove event listener
  removeEventListener(eventType, callback) {
    if (this.listeners.has(eventType)) {
      this.listeners.get(eventType).delete(callback);
    }
  }

  // Notify all listeners for a given event type
  notifyListeners(eventType, data) {
    if (this.listeners.has(eventType)) {
      this.listeners.get(eventType).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in WebSocket ${eventType} listener:`, error);
        }
      });
    }
  }
}

// Create a singleton instance
const webSocketService = new WebSocketService();

export default webSocketService;
