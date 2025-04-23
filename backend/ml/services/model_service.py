import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import json
import pickle

from ..models.base_model import BaseModel
from ..models.behavior_model import BehaviorModel
from ..models.communication_model import CommunicationModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService:
    """
    Service for managing ML models, their training, and inference.
    
    This service provides a unified interface for the AI assistant to interact
    with various ML models. It handles model lifecycle, training, storage, and 
    inference operations.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model service.
        
        Args:
            models_dir: Directory where models are stored
        """
        self.models_dir = models_dir
        self.models: Dict[str, BaseModel] = {}
        self._ensure_models_dir()
        
    def _ensure_models_dir(self) -> None:
        """Ensure the models directory exists."""
        os.makedirs(self.models_dir, exist_ok=True)
        
    def get_model(self, model_name: str) -> Optional[BaseModel]:
        """
        Get a model by name.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            The model if found, None otherwise
        """
        return self.models.get(model_name)
    
    def register_model(self, model: BaseModel) -> None:
        """
        Register a model with the service.
        
        Args:
            model: Model instance to register
        """
        self.models[model.name] = model
        logger.info(f"Registered model: {model.name} (version: {model.version})")
    
    def create_behavior_model(self, name: str = "behavior_model", 
                             input_dim: int = 100, 
                             embedding_dim: int = 64,
                             lstm_units: int = 128,
                             sequence_length: int = 10) -> BehaviorModel:
        """
        Create a new behavior model.
        
        Args:
            name: Name for the model
            input_dim: Input dimension (vocabulary size)
            embedding_dim: Embedding dimension
            lstm_units: Number of LSTM units
            sequence_length: Input sequence length
            
        Returns:
            Newly created behavior model
        """
        model = BehaviorModel(
            name=name,
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
            sequence_length=sequence_length
        )
        self.register_model(model)
        return model
    
    def create_communication_model(self, name: str = "communication_model",
                                  vocab_size: int = 10000,
                                  embedding_dim: int = 256,
                                  lstm_units: int = 256,
                                  max_sequence_length: int = 100) -> CommunicationModel:
        """
        Create a new communication model.
        
        Args:
            name: Name for the model
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            lstm_units: Number of LSTM units
            max_sequence_length: Maximum sequence length
            
        Returns:
            Newly created communication model
        """
        model = CommunicationModel(
            name=name,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
            max_sequence_length=max_sequence_length
        )
        self.register_model(model)
        return model
    
    def save_model(self, model_name: str) -> Optional[str]:
        """
        Save a model to disk.
        
        Args:
            model_name: Name of the model to save
            
        Returns:
            Path to the saved model, or None if model not found
        """
        model = self.get_model(model_name)
        if model is None:
            logger.error(f"Model {model_name} not found")
            return None
        
        # Create model directory
        model_dir = os.path.join(self.models_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model
        model_path = model.save(model_dir)
        logger.info(f"Saved model {model_name} to {model_path}")
        
        return model_path
    
    def load_behavior_model(self, model_path: str, metadata_path: Optional[str] = None) -> Optional[BehaviorModel]:
        """
        Load a behavior model from disk.
        
        Args:
            model_path: Path to the model file
            metadata_path: Optional path to metadata file
            
        Returns:
            Loaded behavior model, or None if loading fails
        """
        try:
            model = BehaviorModel.load(model_path, metadata_path)
            self.register_model(model)
            return model
        except Exception as e:
            logger.error(f"Error loading behavior model: {e}")
            return None
    
    def load_communication_model(self, model_path: str, metadata_path: Optional[str] = None,
                               tokenizer_path: Optional[str] = None) -> Optional[CommunicationModel]:
        """
        Load a communication model from disk.
        
        Args:
            model_path: Path to the model file
            metadata_path: Optional path to metadata file
            tokenizer_path: Optional path to tokenizer file
            
        Returns:
            Loaded communication model, or None if loading fails
        """
        try:
            model = CommunicationModel.load(model_path, metadata_path, tokenizer_path)
            self.register_model(model)
            return model
        except Exception as e:
            logger.error(f"Error loading communication model: {e}")
            return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.
        
        Returns:
            List of model information dictionaries
        """
        return [
            {
                'name': model.name,
                'version': model.version,
                'type': model.__class__.__name__,
                'created_at': model.created_at,
                'updated_at': model.updated_at,
                'training_history': len(model.training_history)
            }
            for model in self.models.values()
        ]
    
    def predict_behavior(self, model_name: str, sequences: List[List[int]], top_k: int = 3) -> List[List[Tuple[int, float]]]:
        """
        Predict next likely activities based on behavior patterns.
        
        Args:
            model_name: Name of the behavior model to use
            sequences: List of activity sequences
            top_k: Number of top predictions to return
            
        Returns:
            Predictions for each sequence
        """
        model = self.get_model(model_name)
        if model is None or not isinstance(model, BehaviorModel):
            logger.error(f"Model {model_name} not found or not a BehaviorModel")
            return []
        
        return model.predict(sequences, top_k=top_k)
    
    def generate_communication(self, model_name: str, seed_texts: List[str], 
                              max_length: int = 50,
                              temperature: float = 0.7,
                              top_k: int = 3) -> List[str]:
        """
        Generate communication text in the user's style.
        
        Args:
            model_name: Name of the communication model to use
            seed_texts: List of seed texts to start generation from
            max_length: Maximum length of generated text
            temperature: Temperature for controlling randomness
            top_k: Number of top predictions to sample from
            
        Returns:
            List of generated texts
        """
        model = self.get_model(model_name)
        if model is None or not isinstance(model, CommunicationModel):
            logger.error(f"Model {model_name} not found or not a CommunicationModel")
            return []
        
        return model.predict(
            seed_texts,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k
        )
    
    def train_behavior_model(self, model_name: str, X: List[List[int]], y: List[int],
                           validation_split: float = 0.2,
                           epochs: int = 50,
                           batch_size: int = 64,
                           patience: int = 5) -> Dict[str, Any]:
        """
        Train a behavior model.
        
        Args:
            model_name: Name of the model to train
            X: Input activity sequences
            y: Target next activities
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            patience: Early stopping patience
            
        Returns:
            Training metrics
        """
        model = self.get_model(model_name)
        if model is None or not isinstance(model, BehaviorModel):
            logger.error(f"Model {model_name} not found or not a BehaviorModel")
            return {'error': f"Model {model_name} not found or not a BehaviorModel"}
        
        metrics = model.train(
            X=X,
            y=y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        
        # Save model after training
        self.save_model(model_name)
        
        return metrics
    
    def train_communication_model(self, model_name: str, texts: List[str],
                                validation_split: float = 0.2,
                                epochs: int = 50,
                                batch_size: int = 64,
                                patience: int = 5) -> Dict[str, Any]:
        """
        Train a communication model.
        
        Args:
            model_name: Name of the model to train
            texts: List of text samples
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            patience: Early stopping patience
            
        Returns:
            Training metrics
        """
        model = self.get_model(model_name)
        if model is None or not isinstance(model, CommunicationModel):
            logger.error(f"Model {model_name} not found or not a CommunicationModel")
            return {'error': f"Model {model_name} not found or not a CommunicationModel"}
        
        metrics = model.train(
            texts=texts,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        
        # Save model after training
        self.save_model(model_name)
        
        return metrics
