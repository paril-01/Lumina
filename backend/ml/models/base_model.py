import os
import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np

class BaseModel(ABC):
    """
    Abstract base class for all ML models in the AI Assistant.
    
    This class defines the interface that all models must implement
    and provides common functionality for model management.
    """
    
    def __init__(self, name: str, version: str = "0.1.0"):
        """
        Initialize a new model instance.
        
        Args:
            name: A unique name for the model
            version: Semantic versioning of the model
        """
        self.name = name
        self.version = version
        self.model = None
        self.training_history = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.metadata = {}
    
    @abstractmethod
    def train(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """
        Train the model on provided data.
        
        Args:
            X: Features for training
            y: Target values for training
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: Any, **kwargs) -> Any:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to make predictions on
            **kwargs: Additional prediction parameters
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """
        Evaluate the model on provided data.
        
        Args:
            X: Features for evaluation
            y: Target values for evaluation
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    def save(self, path: str) -> str:
        """
        Save the model to disk.
        
        Args:
            path: Directory path to save the model
            
        Returns:
            Path to the saved model
        """
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, f"{self.name}_{self.version}.pkl")
        
        # Save model object
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        metadata = {
            'name': self.name,
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': datetime.now().isoformat(),
            'training_history': self.training_history,
            'metadata': self.metadata
        }
        
        metadata_path = os.path.join(path, f"{self.name}_{self.version}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return model_path
    
    @classmethod
    def load(cls, model_path: str, metadata_path: Optional[str] = None) -> 'BaseModel':
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the saved model file
            metadata_path: Optional path to the metadata file
            
        Returns:
            Loaded model instance
        """
        # This is a class method that needs to be implemented by subclasses
        # because we need to know which specific model class to instantiate
        raise NotImplementedError("Subclasses must implement the load method")
    
    def add_training_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Add training metrics to the model's history.
        
        Args:
            metrics: Dictionary of metrics from training
        """
        metrics['timestamp'] = datetime.now().isoformat()
        self.training_history.append(metrics)
        self.updated_at = metrics['timestamp']
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': self.name,
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'training_history': self.training_history,
            'metadata': self.metadata
        }
