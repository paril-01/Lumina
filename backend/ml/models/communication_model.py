from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.layers import Bidirectional, Attention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import json
from datetime import datetime

from .base_model import BaseModel

class CommunicationModel(BaseModel):
    """
    Model for generating and styling communications in the user's writing style.
    
    This model can generate email drafts, chat messages, etc. in a style that
    mimics the user's writing patterns and preferences.
    """
    
    def __init__(self, name: str = "communication_model", version: str = "0.1.0",
                vocab_size: int = 10000, embedding_dim: int = 256, 
                lstm_units: int = 256, max_sequence_length: int = 100):
        """
        Initialize a new communication style model.
        
        Args:
            name: Name of the model
            version: Semantic version of the model
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the embedding layer
            lstm_units: Number of units in LSTM layers
            max_sequence_length: Maximum sequence length for text processing
        """
        super().__init__(name, version)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.max_sequence_length = max_sequence_length
        self.tokenizer = None
        
        self.metadata.update({
            'model_type': 'Bi-directional LSTM for Text Generation',
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'lstm_units': lstm_units,
            'max_sequence_length': max_sequence_length
        })
        
        # Build the model architecture
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the model architecture for text generation."""
        # Input layer
        inputs = Input(shape=(self.max_sequence_length,))
        
        # Embedding layer
        x = Embedding(input_dim=self.vocab_size, 
                     output_dim=self.embedding_dim, 
                     input_length=self.max_sequence_length)(inputs)
        
        # Bidirectional LSTM layers
        x = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Bidirectional(LSTM(self.lstm_units // 2, return_sequences=False))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Output layer - predict the next word
        outputs = Dense(self.vocab_size, activation='softmax')(x)
        
        # Create and compile the model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
    
    def prepare_data(self, texts: List[str], fit_tokenizer: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare text data for training or prediction.
        
        Args:
            texts: List of text samples
            fit_tokenizer: Whether to fit the tokenizer on the data
            
        Returns:
            X, y: Features and targets for sequence modeling
        """
        # Create or update tokenizer
        if fit_tokenizer or self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Create input-target pairs
        X, y = [], []
        for sequence in sequences:
            for i in range(1, len(sequence)):
                X.append(sequence[:i])
                y.append(sequence[i])
        
        # Pad sequences
        X = pad_sequences(X, maxlen=self.max_sequence_length, padding='pre')
        
        return np.array(X), np.array(y)
    
    def train(self, texts: List[str], validation_split: float = 0.2,
             epochs: int = 50, batch_size: int = 64, patience: int = 5, 
             **kwargs) -> Dict[str, Any]:
        """
        Train the communication model on text samples.
        
        Args:
            texts: List of text samples (emails, messages, etc.)
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            patience: Early stopping patience
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training metrics
        """
        # Prepare the data
        X, y = self.prepare_data(texts, fit_tokenizer=True)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
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
            'vocabulary_size': len(self.tokenizer.word_index) + 1
        }
        
        self.add_training_metrics(metrics)
        
        return metrics
    
    def generate_text(self, seed_text: str, max_length: int = 100, 
                     temperature: float = 0.7, top_k: int = 3) -> str:
        """
        Generate text in the user's style starting from a seed text.
        
        Args:
            seed_text: Initial text to start generation from
            max_length: Maximum number of words to generate
            temperature: Control randomness (higher = more random)
            top_k: Number of top predictions to sample from
            
        Returns:
            Generated text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Model must be trained first.")
        
        # Convert seed text to sequence
        current_sequence = self.tokenizer.texts_to_sequences([seed_text])[0]
        generated_text = seed_text
        
        # Generate words
        for _ in range(max_length):
            # Pad sequence to required input length
            padded_sequence = pad_sequences([current_sequence], 
                                           maxlen=self.max_sequence_length, 
                                           padding='pre')
            
            # Predict next word probabilities
            predictions = self.model.predict(padded_sequence, verbose=0)[0]
            
            # Apply temperature scaling
            predictions = np.log(predictions) / temperature
            exp_preds = np.exp(predictions)
            predictions = exp_preds / np.sum(exp_preds)
            
            # Get top k predictions
            top_indices = np.argsort(predictions)[-top_k:]
            top_probs = predictions[top_indices]
            top_probs = top_probs / np.sum(top_probs)  # Renormalize
            
            # Sample from top k
            predicted_index = np.random.choice(top_indices, p=top_probs)
            
            # Convert index to word
            for word, index in self.tokenizer.word_index.items():
                if index == predicted_index:
                    generated_text += " " + word
                    current_sequence.append(predicted_index)
                    break
            
            # Stop if we generate an end token (could be customized)
            if word == "<END>" or word == "." or word == "!":
                break
                
        return generated_text
    
    def predict(self, texts: List[str], **kwargs) -> List[str]:
        """
        Generate completions for each text in the input list.
        
        Args:
            texts: List of seed texts to generate completions for
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated completions
        """
        max_length = kwargs.get('max_length', 50)
        temperature = kwargs.get('temperature', 0.7)
        top_k = kwargs.get('top_k', 3)
        
        results = []
        for text in texts:
            generated = self.generate_text(
                seed_text=text,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k
            )
            results.append(generated)
            
        return results
    
    def evaluate(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            texts: List of test text samples
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Prepare the data
        X, y = self.prepare_data(texts, fit_tokenizer=False)
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(X, y, verbose=0)
        
        # Calculate perplexity
        perplexity = np.exp(loss)
        
        metrics = {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'perplexity': float(perplexity)
        }
        
        return metrics
    
    def save(self, path: str) -> str:
        """
        Save the model and tokenizer to disk.
        
        Args:
            path: Directory path to save the model
            
        Returns:
            Path to the saved model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model in keras format
        model_path = os.path.join(path, f"{self.name}_{self.version}.h5")
        self.model.save(model_path)
        
        # Save tokenizer
        tokenizer_path = os.path.join(path, f"{self.name}_{self.version}_tokenizer.pkl")
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
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
    def load(cls, model_path: str, metadata_path: Optional[str] = None, 
             tokenizer_path: Optional[str] = None) -> 'CommunicationModel':
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the saved model file
            metadata_path: Optional path to the metadata file
            tokenizer_path: Optional path to the tokenizer file
            
        Returns:
            Loaded model instance
        """
        # Determine paths if not provided
        if metadata_path is None:
            metadata_path = model_path.replace(".h5", "_metadata.json")
        
        if tokenizer_path is None:
            tokenizer_path = model_path.replace(".h5", "_tokenizer.pkl")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract parameters from metadata
        params = metadata.get('metadata', {})
        vocab_size = params.get('vocab_size', 10000)
        embedding_dim = params.get('embedding_dim', 256)
        lstm_units = params.get('lstm_units', 256)
        max_sequence_length = params.get('max_sequence_length', 100)
        
        # Create a new model instance
        instance = cls(
            name=metadata.get('name', 'communication_model'),
            version=metadata.get('version', '0.1.0'),
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
            max_sequence_length=max_sequence_length
        )
        
        # Load the model
        instance.model = load_model(model_path)
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            instance.tokenizer = pickle.load(f)
        
        # Set the instance attributes from metadata
        instance.created_at = metadata.get('created_at', instance.created_at)
        instance.updated_at = metadata.get('updated_at', instance.updated_at)
        instance.training_history = metadata.get('training_history', [])
        instance.metadata = metadata.get('metadata', {})
        
        return instance
