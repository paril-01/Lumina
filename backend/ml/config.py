"""
Configuration settings for the ML system.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = os.path.join(BASE_DIR, 'models_data')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    # Behavior model settings
    'behavior_model': {
        'input_dim': 1000,             # Vocabulary size for activities
        'embedding_dim': 64,           # Embedding dimension
        'lstm_units': 128,             # LSTM layer units
        'sequence_length': 20,         # Length of activity sequences to analyze
        'batch_size': 64,              # Training batch size
        'epochs': 50,                  # Max training epochs
        'learning_rate': 0.001,        # Learning rate
        'dropout_rate': 0.2,           # Dropout for regularization
    },
    
    # Communication model settings
    'communication_model': {
        'vocab_size': 10000,           # Vocabulary size for text
        'embedding_dim': 256,          # Embedding dimension
        'lstm_units': 256,             # LSTM layer units
        'max_sequence_length': 100,    # Max sequence length for text
        'batch_size': 32,              # Training batch size
        'epochs': 30,                  # Max training epochs
        'learning_rate': 0.001,        # Learning rate
        'dropout_rate': 0.3,           # Dropout for regularization
    }
}

# Training settings
TRAINING_CONFIG = {
    'validation_split': 0.2,           # Fraction of data for validation
    'early_stopping_patience': 5,      # Patience for early stopping
    'save_best_only': True,            # Only save the best model
    'monitor_metric': 'val_loss',      # Metric to monitor for early stopping
    'min_samples_for_training': 100,   # Minimum samples required for training
}

# Data collection settings
DATA_COLLECTION_CONFIG = {
    'max_samples_per_user': 10000,     # Max samples to store per user
    'session_timeout_minutes': 30,     # Session timeout for activity segmentation
    'min_session_activities': 5,       # Minimum activities to define a session
    'privacy_sensitive_fields': [      # Fields containing sensitive data
        'password', 'token', 'key', 'secret', 'credit', 'card', 'ssn', 'social'
    ],
}

# Feature extraction settings
FEATURE_EXTRACTION_CONFIG = {
    'text_max_features': 5000,         # Max features for text vectorization
    'sentiment_analysis_enabled': True,# Enable sentiment analysis
    'time_feature_extraction': True,   # Extract temporal features
    'min_feature_occurrence': 5,       # Min occurrences for categorical features
}

# Inference settings
INFERENCE_CONFIG = {
    'prediction_batch_size': 32,       # Batch size for prediction
    'top_k_predictions': 3,            # Number of top predictions to return
    'prediction_temperature': 0.7,     # Temperature for text generation
    'minimum_confidence': 0.4,         # Minimum confidence threshold for predictions
}

# Privacy and security settings
PRIVACY_CONFIG = {
    'data_retention_days': 90,         # Number of days to retain user data
    'anonymize_personal_data': True,   # Whether to anonymize personal data
    'encryption_enabled': True,        # Whether to encrypt sensitive data
    'user_consent_required': True,     # Whether user consent is required
    'consent_expiry_days': 365,        # Days until consent expires
}

# Integration settings 
INTEGRATION_CONFIG = {
    'kafka_enabled': False,            # Whether Kafka integration is enabled
    'redis_enabled': False,            # Whether Redis integration is enabled
    'event_batch_size': 100,           # Batch size for event processing
    'event_processing_interval': 60,   # Seconds between event processing
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model configuration dictionary
    """
    return MODEL_CONFIGS.get(model_name, {})
