import os
import json
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict, Counter

# For advanced deep learning capabilities
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Bidirectional, Input, Attention, MultiHeadAttention
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# For transformer models
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# For classical ML
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningEngine:
    """
    LearningEngine processes user activity data to build behavioral models
    and identify patterns that can be used for automation.
    """
    
    def __init__(self, user_id: int, settings: Dict = None):
        """
        Initialize the learning engine
        
        Args:
            user_id: ID of the user whose activities are being learned
            settings: Configuration settings for learning
        """
        self.user_id = user_id
        self.settings = settings or {
            "learning_rate": 0.01,
            "min_confidence_threshold": 0.7,
            "min_pattern_occurrences": 3,
            "max_sequence_length": 20,
            "temporal_weight_decay": 0.9,  # Newer activities have more weight
            "model_update_frequency": 100,  # Update models every N activities
            "embedding_dim": 128,      # Embedding dimensions for sequence model
            "attention_heads": 4,      # Number of attention heads in transformer
            "anomaly_threshold": 0.95, # Threshold for anomaly detection
            "learning_methods": {
                "sequence_prediction": True,
                "temporal_patterns": True,
                "action_clustering": True,
                "nlp_analysis": True,
                "transformer_modeling": TRANSFORMERS_AVAILABLE,
                "anomaly_detection": True
            }
        }
        
        # Data structures to hold patterns and models
        self.activity_sequences = []
        self.temporal_patterns = defaultdict(list)
        self.user_profile = {
            "behavioral_patterns": {},
            "communication_style": {},
            "frequent_actions": {},
            "application_preferences": {},
            "sequence_models": {},
            "anomaly_scores": {}
        }
        
        # Activity encoding maps for model input
        self.activity_type_map = {}
        self.application_map = {}
        self.action_map = {}
        self.next_type_id = 1
        self.next_app_id = 1
        self.next_action_id = 1
        
        # Track learning progress
        self.learning_stage = "initial"  # initial, active, mature
        self.learning_progress = 0.0  # 0 to 1.0
        self.last_updated = datetime.now()
        self.activities_processed = 0
        
        # Redis client for event queue if available
        self.redis_client = None
        try:
            import redis
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            logger.info("Redis client initialized for event queuing")
        except (ImportError, Exception) as e:
            logger.warning(f"Redis client not available: {e}")
        
        # Initialize ML models if available
        self.models = {}
        if TF_AVAILABLE:
            self._init_deep_learning_models()
        
        if TRANSFORMERS_AVAILABLE:
            self._init_transformer_models()
    
    def process_activity(self, activity: Dict[str, Any]):
        """
        Process a new user activity and update the learning models
        
        Args:
            activity: Dictionary containing activity data
        """
        # Add to activity sequence for pattern recognition
        self.activity_sequences.append(activity)
        
        # Keep sequence at manageable size
        if len(self.activity_sequences) > 1000:
            self.activity_sequences = self.activity_sequences[-1000:]
        
        # Process by activity type
        activity_type = activity.get("activity_type")
        application = activity.get("application")
        action = activity.get("action")
        
        # Update temporal patterns
        day_of_week = datetime.now().weekday()
        hour_of_day = datetime.now().hour
        
        # Add to temporal records
        self.temporal_patterns[f"dow_{day_of_week}"].append(activity)
        self.temporal_patterns[f"hour_{hour_of_day}"].append(activity)
        self.temporal_patterns[f"app_{application}"].append(activity)
        
        # Limit memory usage
        for key in self.temporal_patterns:
            if len(self.temporal_patterns[key]) > 500:
                self.temporal_patterns[key] = self.temporal_patterns[key][-500:]
        
        # Process based on activity type
        if activity_type == "keyboard":
            self._process_keyboard_activity(activity)
        elif activity_type == "mouse":
            self._process_mouse_activity(activity)
        elif activity_type == "system":
            self._process_system_activity(activity)
        
        # Update learning progress
        self._update_learning_progress()
        
        # Perform periodic model updates when enough data is collected
        if len(self.activity_sequences) % 100 == 0:
            self._update_models()
    
    def _process_keyboard_activity(self, activity: Dict[str, Any]):
        """Process keyboard-related activities"""
        # Update communication style based on keyboard patterns
        app = activity.get("application", "")
        
        # In a real implementation, we'd analyze typing speed, patterns, etc.
        # Here we're just incrementing a counter for demonstration
        if "communication_style" not in self.user_profile:
            self.user_profile["communication_style"] = {}
        
        # Track applications used for text input
        if app not in self.user_profile["communication_style"]:
            self.user_profile["communication_style"][app] = {
                "count": 0,
                "last_activity": None
            }
        
        self.user_profile["communication_style"][app]["count"] += 1
        self.user_profile["communication_style"][app]["last_activity"] = datetime.now().isoformat()
    
    def _process_mouse_activity(self, activity: Dict[str, Any]):
        """Process mouse-related activities"""
        action = activity.get("action", "")
        app = activity.get("application", "")
        
        # Track mouse usage patterns
        if "mouse_patterns" not in self.user_profile:
            self.user_profile["mouse_patterns"] = {}
        
        key = f"{app}_{action}"
        if key not in self.user_profile["mouse_patterns"]:
            self.user_profile["mouse_patterns"][key] = {
                "count": 0,
                "positions": []
            }
        
        self.user_profile["mouse_patterns"][key]["count"] += 1
        
        # Store some position data (with limits to prevent memory issues)
        if "activity_metadata" in activity and "position" in activity["activity_metadata"]:
            positions = self.user_profile["mouse_patterns"][key]["positions"]
            positions.append(activity["activity_metadata"]["position"])
            
            # Limit stored positions
            if len(positions) > 100:
                positions = positions[-100:]
            self.user_profile["mouse_patterns"][key]["positions"] = positions
    
    def _process_system_activity(self, activity: Dict[str, Any]):
        """Process system-level activities like application changes"""
        action = activity.get("action", "")
        app = activity.get("application", "")
        
        # Track application usage
        if "application_usage" not in self.user_profile:
            self.user_profile["application_usage"] = {}
        
        if app not in self.user_profile["application_usage"]:
            self.user_profile["application_usage"][app] = {
                "count": 0,
                "duration": 0,
                "last_start": None
            }
        
        if action == "application_focus":
            # Application was opened or brought to focus
            self.user_profile["application_usage"][app]["count"] += 1
            self.user_profile["application_usage"][app]["last_start"] = datetime.now().isoformat()
        elif action == "application_blur" and self.user_profile["application_usage"][app]["last_start"]:
            # Calculate duration of application usage
            try:
                last_start = datetime.fromisoformat(self.user_profile["application_usage"][app]["last_start"])
                duration = (datetime.now() - last_start).total_seconds()
                self.user_profile["application_usage"][app]["duration"] += duration
            except (ValueError, TypeError):
                pass
            self.user_profile["application_usage"][app]["last_start"] = None
    
    def detect_patterns(self) -> List[Dict[str, Any]]:
        """
        Analyze stored activities to detect patterns that could be automated
        
        Returns:
            List of detected patterns with confidence scores
        """
        patterns = []
        
        # Detect application usage patterns
        app_patterns = self._detect_application_patterns()
        patterns.extend(app_patterns)
        
        # Detect temporal patterns (time-based)
        temporal_patterns = self._detect_temporal_patterns()
        patterns.extend(temporal_patterns)
        
        # Detect action sequences
        sequence_patterns = self._detect_action_sequences()
        patterns.extend(sequence_patterns)
        
        # Sort by confidence
        patterns.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return patterns
    
    def _detect_application_patterns(self) -> List[Dict[str, Any]]:
        """Detect patterns in application usage"""
        patterns = []
        
        if "application_usage" not in self.user_profile:
            return patterns
        
        # Find most frequently used applications
        usage = self.user_profile["application_usage"]
        if not usage:
            return patterns
            
        # Sort by count
        apps_by_count = sorted(usage.items(), key=lambda x: x[1]["count"], reverse=True)
        
        # Create patterns for top applications
        for app, data in apps_by_count[:5]:  # Top 5 most used apps
            # Skip if not used enough times
            if data["count"] < self.settings["min_pattern_occurrences"]:
                continue
                
            # Calculate confidence based on usage frequency
            total_usage = sum(a[1]["count"] for a in apps_by_count)
            confidence = data["count"] / total_usage if total_usage > 0 else 0
            
            if confidence >= self.settings["min_confidence_threshold"]:
                patterns.append({
                    "type": "application_preference",
                    "description": f"Frequently uses {app}",
                    "application": app,
                    "confidence": confidence,
                    "occurrences": data["count"],
                    "pattern_data": {
                        "app": app,
                        "usage_count": data["count"],
                        "total_duration": data["duration"]
                    }
                })
        
        return patterns
    
    def _detect_temporal_patterns(self) -> List[Dict[str, Any]]:
        """Detect time-based patterns in user activity"""
        patterns = []
        
        # Analyze time of day patterns
        for hour in range(24):
            hour_key = f"hour_{hour}"
            if hour_key in self.temporal_patterns:
                activities = self.temporal_patterns[hour_key]
                
                # Skip if not enough activities
                if len(activities) < self.settings["min_pattern_occurrences"]:
                    continue
                
                # Count application usage by hour
                app_counts = Counter([a.get("application", "") for a in activities])
                
                # Find the most common application for this hour
                if app_counts:
                    top_app, count = app_counts.most_common(1)[0]
                    confidence = count / len(activities)
                    
                    if confidence >= self.settings["min_confidence_threshold"]:
                        patterns.append({
                            "type": "temporal_app_usage",
                            "description": f"Uses {top_app} at {hour}:00",
                            "application": top_app,
                            "time": f"{hour}:00",
                            "confidence": confidence,
                            "occurrences": count,
                            "pattern_data": {
                                "hour": hour,
                                "app": top_app,
                                "count": count
                            }
                        })
        
        # Analyze day of week patterns (similar to hourly analysis)
        for day in range(7):
            day_key = f"dow_{day}"
            if day_key in self.temporal_patterns:
                # Similar analysis could be done for day of week patterns
                pass
        
        return patterns
    
    def _detect_action_sequences(self) -> List[Dict[str, Any]]:
        """Detect repeated sequences of actions"""
        patterns = []
        
        # This requires more complex sequence analysis algorithms
        # For demonstration, we'll use a simplified approach
        
        # In a real implementation, this would use algorithms like:
        # - Apriori algorithm for frequent itemset mining
        # - Sequential pattern mining
        # - N-gram analysis
        
        # Placeholder for sequence detection
        # In a complete implementation, this would be much more sophisticated
        
        return patterns
    
    def _update_learning_progress(self):
        """Update the learning stage and progress"""
        # Count total activities processed
        total_activities = len(self.activity_sequences)
        
        # Update learning stage based on activity count
        if total_activities < 100:
            self.learning_stage = "initial"
            self.learning_progress = min(0.3, total_activities / 100)
        elif total_activities < 1000:
            self.learning_stage = "active"
            self.learning_progress = 0.3 + min(0.5, (total_activities - 100) / 900)
        else:
            self.learning_stage = "mature"
            self.learning_progress = min(0.8 + (total_activities - 1000) / 10000, 1.0)
        
        self.last_updated = datetime.now()
    
    def _update_models(self):
        """Update machine learning models with new data"""
        # Skip if we don't have enough data yet
        if len(self.activity_sequences) < 50:
            return
        
        self.activities_processed += 1
        
        # Check if it's time to update models based on frequency setting
        if self.activities_processed % self.settings["model_update_frequency"] != 0:
            return
            
        logger.info(f"Updating models with {len(self.activity_sequences)} activities")
        
        try:
            # Update sequence models if TensorFlow is available
            if TF_AVAILABLE and "sequence_model" in self.models:
                self._update_sequence_model()
                
            # Update transformer models if available
            if TRANSFORMERS_AVAILABLE and "activity_transformer" in self.models:
                self._update_transformer_model()
                
            # Update anomaly detection model
            if SKLEARN_AVAILABLE:
                self._update_anomaly_detection()
                
            logger.info("Models updated successfully")
        except Exception as e:
            logger.error(f"Error updating models: {e}")
    
    def _init_deep_learning_models(self):
        """Initialize deep learning models for user behavior prediction"""
        # If we don't have TensorFlow available, don't try to initialize
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available, skipping deep learning model initialization")
            return
        
        try:
            # LSTM model for sequence prediction
            seq_input = Input(shape=(self.settings["max_sequence_length"], 3))  # Input shape: [seq_len, features]
            
            # Bidirectional LSTM for capturing context in both directions
            x = Bidirectional(LSTM(128, return_sequences=True))(seq_input)
            x = Dropout(0.2)(x)
            
            # Add attention mechanism to focus on important parts of sequence
            if hasattr(keras.layers, 'Attention'):  # Check if Attention layer is available
                query = Dense(64)(x)
                key = Dense(64)(x)
                value = Dense(64)(x)
                attention = Attention()([query, key, value])
                x = attention
            else:
                # Use dense layer if attention not available
                x = Dense(64, activation='relu')(x)
            
            # Final prediction layers
            x = Dropout(0.1)(x)
            x = Dense(32, activation='relu')(x)
            next_app_output = Dense(100, activation='softmax', name='next_app')(x)  # Predict next application
            next_action_output = Dense(100, activation='softmax', name='next_action')(x)  # Predict next action
            
            # Create multi-output model
            self.models["sequence_model"] = Model(
                inputs=seq_input,
                outputs=[next_app_output, next_action_output]
            )
            
            # Compile with appropriate loss and metrics
            self.models["sequence_model"].compile(
                optimizer='adam',
                loss={'next_app': 'sparse_categorical_crossentropy', 'next_action': 'sparse_categorical_crossentropy'},
                metrics=['accuracy']
            )
            
            logger.info("Successfully initialized deep learning sequence models")
            
        except Exception as e:
            # Log but don't crash if model initialization fails
            logger.error(f"Error initializing deep learning models: {e}")
    
    def _init_transformer_models(self):
        """Initialize transformer-based models for understanding user behavior patterns"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library not available, skipping transformer model initialization")
            return
        
        try:
            # Initialize BERT tokenizer and model for text understanding
            self.models["bert_tokenizer"] = BertTokenizer.from_pretrained('bert-base-uncased')
            self.models["bert_model"] = BertModel.from_pretrained('bert-base-uncased')
            
            # Initialize GPT-2 for next-text prediction if available
            try:
                self.models["gpt2_tokenizer"] = GPT2Tokenizer.from_pretrained('gpt2')
                self.models["gpt2_model"] = GPT2Model.from_pretrained('gpt2')
            except Exception as e:
                logger.warning(f"Error initializing GPT-2 model: {e}")
            
            # PyTorch Transformer for activity sequence modeling
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("Using GPU for transformer models")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU for transformer models")
            
            # Create a simple transformer encoder model
            class ActivityTransformer(nn.Module):
                def __init__(self, input_dim, embed_dim, num_heads, num_classes, seq_len):
                    super().__init__()
                    self.embedding = nn.Linear(input_dim, embed_dim)
                    self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
                    encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
                    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
                    self.output = nn.Linear(embed_dim, num_classes)
                
                def forward(self, x):
                    # x shape: [batch_size, seq_len, input_dim]
                    x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
                    x = x + self.position_embedding
                    x = x.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim] for transformer
                    x = self.transformer_encoder(x)
                    x = x.permute(1, 0, 2)  # [batch_size, seq_len, embed_dim]
                    return self.output(x)  # [batch_size, seq_len, num_classes]
            
            # Initialize the model
            input_dim = 3  # activity_type, application, action IDs
            embed_dim = self.settings["embedding_dim"]
            num_heads = self.settings["attention_heads"]
            num_classes = 100  # Placeholder for output classes (will expand as needed)
            seq_len = self.settings["max_sequence_length"]
            
            self.models["activity_transformer"] = ActivityTransformer(
                input_dim=input_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_classes=num_classes,
                seq_len=seq_len
            ).to(device)
            
            # Initialize optimizer
            self.models["transformer_optimizer"] = torch.optim.Adam(
                self.models["activity_transformer"].parameters(),
                lr=self.settings["learning_rate"]
            )
            
            logger.info("Successfully initialized transformer models")
        
        except Exception as e:
            logger.error(f"Error initializing transformer models: {e}")
    
    def _update_sequence_model(self):
        """Update the LSTM sequence model with new activity data"""
        if not TF_AVAILABLE or "sequence_model" not in self.models:
            return
            
        try:
            # Prepare training data from activity sequences
            sequences = []
            next_apps = []
            next_actions = []
            
            # Process each sequence with at least max_sequence_length+1 items
            if len(self.activity_sequences) < self.settings["max_sequence_length"] + 1:
                return
                
            # Convert activities to training sequences
            for i in range(len(self.activity_sequences) - self.settings["max_sequence_length"]):
                seq = self.activity_sequences[i:i+self.settings["max_sequence_length"]]
                next_item = self.activity_sequences[i+self.settings["max_sequence_length"]]
                
                # Encode sequence
                encoded_seq = []
                for item in seq:
                    # Encode activity type
                    act_type = item.get("activity_type", "unknown")
                    if act_type not in self.activity_type_map:
                        self.activity_type_map[act_type] = self.next_type_id
                        self.next_type_id += 1
                    
                    # Encode application
                    app = item.get("application", "unknown")
                    if app not in self.application_map:
                        self.application_map[app] = self.next_app_id
                        self.next_app_id += 1
                    
                    # Encode action
                    action = item.get("action", "unknown")
                    if action not in self.action_map:
                        self.action_map[action] = self.next_action_id
                        self.next_action_id += 1
                    
                    # Add encoded values
                    encoded_seq.append([
                        self.activity_type_map[act_type],
                        self.application_map[app],
                        self.action_map[action]
                    ])
                
                # Encode next item (target)
                next_app = next_item.get("application", "unknown")
                next_action = next_item.get("action", "unknown")
                
                if next_app not in self.application_map:
                    self.application_map[next_app] = self.next_app_id
                    self.next_app_id += 1
                
                if next_action not in self.action_map:
                    self.action_map[next_action] = self.next_action_id
                    self.next_action_id += 1
                
                # Add to training data
                sequences.append(encoded_seq)
                next_apps.append(self.application_map[next_app])
                next_actions.append(self.action_map[next_action])
            
            # Convert to numpy arrays
            X = np.array(sequences)
            y_app = np.array(next_apps)
            y_action = np.array(next_actions)
            
            # Train the model with a small number of epochs to avoid overfitting
            self.models["sequence_model"].fit(
                X, 
                {"next_app": y_app, "next_action": y_action},
                epochs=5,
                batch_size=32,
                verbose=0
            )
            
            logger.info(f"Updated sequence model with {len(sequences)} sequences")
        
        except Exception as e:
            logger.error(f"Error updating sequence model: {e}")
    
    def _update_transformer_model(self):
        """Update the transformer model with new activity data"""
        if not TRANSFORMERS_AVAILABLE or "activity_transformer" not in self.models:
            return
            
        try:
            # Similar to sequence model update but for PyTorch transformer
            # This is simplified for demonstration
            model = self.models["activity_transformer"]
            optimizer = self.models["transformer_optimizer"]
            
            # Set model to training mode
            model.train()
            
            # Process batches of data similar to sequence model update
            # This is a simplified implementation
            logger.info("Transformer model updated")
            
        except Exception as e:
            logger.error(f"Error updating transformer model: {e}")
    
    def _update_anomaly_detection(self):
        """Update anomaly detection model to identify unusual user behavior"""
        if not SKLEARN_AVAILABLE:
            return
            
        try:
            # Prepare features for anomaly detection
            features = []
            
            # Extract temporal patterns
            for act in self.activity_sequences[-500:]:  # Use recent activities
                # Extract hour of day and day of week
                timestamp = act.get("activity_metadata", {}).get("timestamp")
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        hour = dt.hour
                        dow = dt.weekday()
                    except (ValueError, TypeError):
                        hour = 12  # Default
                        dow = 3    # Default (Wednesday)
                else:
                    hour = 12
                    dow = 3
                
                # Encode activity
                act_type = act.get("activity_type", "unknown")
                app = act.get("application", "unknown")
                action = act.get("action", "unknown")
                
                type_id = self.activity_type_map.get(act_type, 0)
                app_id = self.application_map.get(app, 0)
                action_id = self.action_map.get(action, 0)
                
                # Create feature vector
                feature = [hour/24.0, dow/7.0, type_id/10.0, app_id/100.0, action_id/100.0]
                features.append(feature)
            
            if len(features) < 50:  # Need enough data for meaningful anomaly detection
                return
                
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Train isolation forest for anomaly detection
            model = IsolationForest(contamination=0.05, random_state=42)
            model.fit(features_scaled)
            
            # Store model for later use
            self.models["anomaly_detector"] = model
            self.models["feature_scaler"] = scaler
            
            logger.info("Anomaly detection model updated")
            
        except Exception as e:
            logger.error(f"Error updating anomaly detection: {e}")
    
    def get_user_profile(self) -> Dict[str, Any]:
        """
        Get the current user behavioral profile
        
        Returns:
            Dictionary containing the user's behavioral profile
        """
        # Add learning progress information
        profile = {
            **self.user_profile,
            "learning_stage": self.learning_stage,
            "learning_progress": self.learning_progress,
            "last_updated": self.last_updated.isoformat()
        }
        
        return profile

# Example usage:
if __name__ == "__main__":
    # Simple test
    learning_engine = LearningEngine(user_id=1)
    
    # Simulate processing some activities
    for i in range(100):
        activity = {
            "application": f"App{i % 5 + 1}",
            "activity_type": "keyboard" if i % 3 == 0 else "mouse" if i % 3 == 1 else "system",
            "action": "key_press" if i % 3 == 0 else "click" if i % 3 == 1 else "application_focus",
            "activity_metadata": {
                "timestamp": datetime.now().isoformat(),
                "position": {"x": i % 100, "y": i % 100} if i % 3 == 1 else None
            }
        }
        learning_engine.process_activity(activity)
        time.sleep(0.01)
    
    # Get detected patterns
    patterns = learning_engine.detect_patterns()
    print(f"Detected {len(patterns)} patterns")
    for pattern in patterns[:3]:  # Show top 3
        print(f"Pattern: {pattern['description']} (confidence: {pattern['confidence']:.2f})")
    
    # Get user profile
    profile = learning_engine.get_user_profile()
    print(f"Learning progress: {profile['learning_progress']:.2f} ({profile['learning_stage']} stage)")
