from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from datetime import datetime
import os
import json
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Embedding, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .base_model import BaseModel

class BehaviorModel(BaseModel):
    """
    A sequence modeling LSTM neural network for predicting user behavior patterns.
    
    This model analyzes sequences of user activities to predict the next likely action or task.
    It can be used for suggesting task automation based on learned patterns.
    """
    
    def __init__(self, name: str = "behavior_model", version: str = "0.1.0", 
                 input_dim: int = 100, embedding_dim: int = 64, lstm_units: int = 128,
                 sequence_length: int = 10):
        """
        Initialize a new BehaviorModel.
        
        Args:
            name: Name identifier for the model
            version: Semantic version of the model
            input_dim: Dimensionality of the input features (vocabulary size)
            embedding_dim: Dimensionality of the embedding layer
            lstm_units: Number of units in the LSTM layer
            sequence_length: Length of input sequences
        """
        super().__init__(name, version)
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
        self.metadata.update({
            'model_type': 'LSTM Sequence Model',
            'input_dim': input_dim,
            'embedding_dim': embedding_dim,
            'lstm_units': lstm_units,
            'sequence_length': sequence_length
        })
        
        # Create the model architecture
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the LSTM sequence model architecture."""
        model = Sequential()
        
        # Add embedding layer
        model.add(Embedding(input_dim=self.input_dim, 
                           output_dim=self.embedding_dim, 
                           input_length=self.sequence_length))
        
        # Add LSTM layers with dropout for regularization
        model.add(LSTM(units=self.lstm_units, 
                      return_sequences=True,
                      dropout=0.2,
                      recurrent_dropout=0.2))
        model.add(LayerNormalization())
        
        model.add(LSTM(units=self.lstm_units // 2,
                      dropout=0.2,
                      recurrent_dropout=0.2))
        model.add(LayerNormalization())
        
        # Add output layer
        model.add(Dense(units=self.input_dim, activation='softmax'))
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
    
    def train(self, X: List[List[int]], y: List[int], 
              validation_split: float = 0.2,
              epochs: int = 50, 
              batch_size: int = 64,
              patience: int = 5,
              **kwargs) -> Dict[str, Any]:
        """
        Train the behavior model on sequences of user activities.
        
        Args:
            X: List of sequences of activity IDs (each sequence is a list of integers)
            y: List of next activity IDs (integers)
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            patience: Patience for early stopping
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training history
        """
        # Convert inputs to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ]
        
        # Train the model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Record training metrics
        metrics = {
            'epochs_completed': len(history.history['loss']),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        }
        
        self.add_training_metrics(metrics)
        
        return metrics
    
    def predict(self, X: List[List[int]], top_k: int = 3, **kwargs) -> List[List[Tuple[int, float]]]:
        """
        Predict the next likely activities based on a sequence of past activities.
        
        Args:
            X: List of activity sequences (each sequence is a list of integers)
            top_k: Number of top predictions to return for each sequence
            **kwargs: Additional prediction parameters
            
        Returns:
            List of lists, where each inner list contains tuples of (activity_id, probability)
            for the top k predicted next activities
        """
        X = np.array(X)
        
        # Get raw prediction probabilities
        predictions = self.model.predict(X)
        
        # For each prediction, get the top k activities and their probabilities
        results = []
        for pred in predictions:
            # Get indices of top k predictions
            top_indices = np.argsort(pred)[-top_k:][::-1]
            # Get probabilities for those indices
            top_probs = pred[top_indices]
            # Create list of (activity_id, probability) tuples
            top_predictions = [(int(idx), float(prob)) for idx, prob in zip(top_indices, top_probs)]
            results.append(top_predictions)
        
        return results
    
    def evaluate(self, X: List[List[int]], y: List[int], **kwargs) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X: List of sequences of activity IDs
            y: List of next activity IDs
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation metrics
        """
        X = np.array(X)
        y = np.array(y)
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(X, y, verbose=0)
        
        # Make predictions to calculate top-k accuracy
        predictions = self.model.predict(X)
        top_3_accuracy = self._calculate_top_k_accuracy(predictions, y, k=3)
        top_5_accuracy = self._calculate_top_k_accuracy(predictions, y, k=5)
        
        metrics = {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'top_3_accuracy': float(top_3_accuracy),
            'top_5_accuracy': float(top_5_accuracy)
        }
        
        return metrics
    
    def _calculate_top_k_accuracy(self, predictions: np.ndarray, y_true: np.ndarray, k: int = 3) -> float:
        """
        Calculate top-k accuracy for multiclass predictions.
        
        Args:
            predictions: Model prediction probabilities
            y_true: True labels
            k: Top-k parameter
            
        Returns:
            Top-k accuracy score
        """
        top_k_indices = np.argsort(predictions, axis=1)[:, -k:]
        match_count = 0
        
        for i, true_label in enumerate(y_true):
            if true_label in top_k_indices[i]:
                match_count += 1
        
        return match_count / len(y_true)
    
    @classmethod
    def load(cls, model_path: str, metadata_path: Optional[str] = None) -> 'BehaviorModel':
        """
        Load a behavior model from disk.
        
        Args:
            model_path: Path to the saved model file
            metadata_path: Optional path to the metadata file
            
        Returns:
            Loaded BehaviorModel instance
        """
        # Determine metadata path if not provided
        if metadata_path is None:
            metadata_path = model_path.replace(".pkl", "_metadata.json")
            if not os.path.exists(metadata_path):
                metadata_path = model_path.replace(".h5", "_metadata.json")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract parameters from metadata
        params = metadata.get('metadata', {})
        input_dim = params.get('input_dim', 100)
        embedding_dim = params.get('embedding_dim', 64)
        lstm_units = params.get('lstm_units', 128)
        sequence_length = params.get('sequence_length', 10)
        
        # Create a new model instance
        instance = cls(
            name=metadata.get('name', 'behavior_model'),
            version=metadata.get('version', '0.1.0'),
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
            sequence_length=sequence_length
        )
        
        # Load the model weights
        if model_path.endswith('.h5'):
            instance.model = load_model(model_path)
        else:
            with open(model_path, 'rb') as f:
                instance.model = pickle.load(f)
        
        # Set the instance attributes from metadata
        instance.created_at = metadata.get('created_at', instance.created_at)
        instance.updated_at = metadata.get('updated_at', instance.updated_at)
        instance.training_history = metadata.get('training_history', [])
        instance.metadata = metadata.get('metadata', {})
        
        return instance
