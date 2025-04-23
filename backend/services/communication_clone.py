import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import numpy as np
import re

# Optional: For NLP capabilities
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Optional: For transformer-based models
try:
    import torch
    from transformers import (
        GPT2Tokenizer, GPT2LMHeadModel,
        BertTokenizer, BertModel,
        T5Tokenizer, T5ForConditionalGeneration,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommunicationClone:
    """
    CommunicationClone analyzes a user's writing style and communication patterns
    to generate text that mimics their style for email drafts, messages, etc.
    """
    
    def __init__(self, user_id: int, settings: Dict = None, api_key: str = None):
        """
        Initialize the communication clone
        
        Args:
            user_id: ID of the user whose communication style is being cloned
            settings: Configuration settings for the clone
            api_key: Optional API key for external AI services
        """
        self.user_id = user_id
        self.settings = settings or {
            "enabled": False,  # Default disabled for privacy
            "consent_given": False,  # Explicit user consent required
            "style_mimicry_level": 0.7,  # How closely to mimic (0.0 to 1.0)
            "allowed_contexts": ["email_draft", "message_draft", "note"],
            "prohibited_contexts": ["financial", "legal", "password"],
            "max_token_length": 500,
            "preferred_model": "local" if TRANSFORMERS_AVAILABLE else "openai",
            "collect_training_data": True
        }
        
        # Communication data stores
        self.communication_samples = []  # User's writing samples
        self.style_profile = {
            "complexity_score": 0.5,  # Sentence complexity level
            "formality_score": 0.5,   # Formality level
            "emoji_usage_rate": 0.0,  # Rate of emoji usage
            "avg_sentence_length": 15,  # Average words per sentence
            "avg_word_length": 4.5,    # Average letters per word
            "favorite_phrases": [],    # Commonly used phrases
            "word_frequencies": {},    # Word usage frequencies
            "word_embeddings": None,   # Embedding space representation
            "named_entities": {},      # Commonly referenced entities
        }
        
        # Initialize models
        self.models = {}
        self.openai_client = None
        
        # Set up OpenAI if available and API key is provided
        if OPENAI_AVAILABLE and api_key:
            try:
                openai.api_key = api_key
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}")
        
        # Initialize local models if available
        if TRANSFORMERS_AVAILABLE:
            self._init_transformer_models()
    
    def _init_transformer_models(self):
        """Initialize transformer-based models for style analysis and generation"""
        try:
            # Initialize style classifier model
            self.models["style_classifier"] = pipeline(
                "text-classification", 
                model="distilbert-base-uncased-finetuned-sst-2-english",
                tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            # Initialize GPT-2 for text generation
            self.models["gpt2_tokenizer"] = GPT2Tokenizer.from_pretrained("gpt2")
            self.models["gpt2_model"] = GPT2LMHeadModel.from_pretrained("gpt2")
            
            # Initialize T5 for summarization and paraphrasing
            self.models["t5_tokenizer"] = T5Tokenizer.from_pretrained("t5-small")
            self.models["t5_model"] = T5ForConditionalGeneration.from_pretrained("t5-small")
            
            # Initialize BERT for embeddings and feature extraction
            self.models["bert_tokenizer"] = BertTokenizer.from_pretrained("bert-base-uncased")
            self.models["bert_model"] = BertModel.from_pretrained("bert-base-uncased")
            
            logger.info("Transformer models initialized for communication cloning")
        except Exception as e:
            logger.error(f"Error initializing transformer models: {e}")
    
    def add_communication_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Add a communication sample to train the clone
        
        Args:
            sample: Dictionary containing sample text and metadata
                {
                    "text": str,
                    "context": str (e.g., "email", "chat", "document"),
                    "timestamp": str (ISO format),
                    "recipient": str (optional),
                    "sentiment": str (optional)
                }
        
        Returns:
            success: Whether the sample was successfully added
        """
        # Skip if feature is disabled or no consent
        if not self.settings.get("enabled") or not self.settings.get("consent_given"):
            logger.warning("Communication clone is disabled or consent not given")
            return False
        
        # Skip if empty text
        if not sample.get("text"):
            return False
        
        # Add timestamp if not provided
        if not sample.get("timestamp"):
            sample["timestamp"] = datetime.now().isoformat()
        
        # Add to samples
        self.communication_samples.append(sample)
        
        # Limit storage
        max_samples = 1000
        if len(self.communication_samples) > max_samples:
            self.communication_samples = self.communication_samples[-max_samples:]
        
        # Update style profile with new sample
        self._update_style_profile(sample)
        
        return True
    
    def _update_style_profile(self, sample: Dict[str, Any]):
        """Update the style profile with a new sample"""
        text = sample.get("text", "")
        if not text:
            return
        
        # Extract sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Calculate sentence complexity
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        self.style_profile["avg_sentence_length"] = (
            self.style_profile["avg_sentence_length"] * 0.8 + avg_sentence_length * 0.2
        )
        
        # Calculate word complexity
        words = [w for w in re.findall(r'\b\w+\b', text.lower()) if w]
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            self.style_profile["avg_word_length"] = (
                self.style_profile["avg_word_length"] * 0.8 + avg_word_length * 0.2
            )
        
        # Count emoji usage
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251" 
            "]+"
        )
        emojis = emoji_pattern.findall(text)
        emoji_rate = len(emojis) / max(len(words), 1)
        self.style_profile["emoji_usage_rate"] = (
            self.style_profile["emoji_usage_rate"] * 0.8 + emoji_rate * 0.2
        )
        
        # Update word frequencies
        for word in words:
            if len(word) > 2:  # Skip very short words
                if word not in self.style_profile["word_frequencies"]:
                    self.style_profile["word_frequencies"][word] = 1
                else:
                    self.style_profile["word_frequencies"][word] += 1
        
        # Limit word frequencies dictionary
        if len(self.style_profile["word_frequencies"]) > 1000:
            # Keep most frequent words
            sorted_words = sorted(
                self.style_profile["word_frequencies"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            self.style_profile["word_frequencies"] = dict(sorted_words[:1000])
        
        # Find common phrases (n-grams)
        self._update_phrase_analysis(text)
        
        # Update embeddings if transformer models are available
        if TRANSFORMERS_AVAILABLE and "bert_model" in self.models:
            self._update_embeddings(text)
        
        # Estimate formality based on words, sentence length, etc.
        formality_indicators = [
            "please", "thank", "would", "could", "sincerely", "regards",
            "dear", "hello", "hi", "hey", "formal", "officially", "appreciate"
        ]
        informal_indicators = [
            "lol", "haha", "yeah", "cool", "awesome", "btw", "gonna", "wanna",
            "gotta", "dunno", "kinda", "sorta", "yep", "nope"
        ]
        
        formality_matches = sum(text.lower().count(w) for w in formality_indicators)
        informal_matches = sum(text.lower().count(w) for w in informal_indicators)
        
        # Calculate formality score (0.0 to 1.0)
        if formality_matches + informal_matches > 0:
            formality = formality_matches / (formality_matches + informal_matches)
            # Smooth transition in formality score
            self.style_profile["formality_score"] = (
                self.style_profile["formality_score"] * 0.8 + formality * 0.2
            )
    
    def _update_phrase_analysis(self, text: str):
        """Analyze and extract common phrases"""
        # Simple n-gram extraction for common phrases
        words = text.lower().split()
        
        # Extract 2-grams and 3-grams
        if len(words) >= 2:
            for i in range(len(words) - 1):
                bigram = ' '.join(words[i:i+2])
                if bigram not in self.style_profile["favorite_phrases"]:
                    self.style_profile["favorite_phrases"].append(bigram)
        
        if len(words) >= 3:
            for i in range(len(words) - 2):
                trigram = ' '.join(words[i:i+3])
                if trigram not in self.style_profile["favorite_phrases"]:
                    self.style_profile["favorite_phrases"].append(trigram)
        
        # Limit favorite phrases
        if len(self.style_profile["favorite_phrases"]) > 100:
            self.style_profile["favorite_phrases"] = self.style_profile["favorite_phrases"][-100:]
    
    def _update_embeddings(self, text: str):
        """Update the embedding representation of the user's style"""
        try:
            tokenizer = self.models["bert_tokenizer"]
            model = self.models["bert_model"]
            
            # Prepare inputs
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            # Update embeddings
            if self.style_profile["word_embeddings"] is None:
                self.style_profile["word_embeddings"] = embeddings
            else:
                # Moving average of embeddings
                self.style_profile["word_embeddings"] = (
                    self.style_profile["word_embeddings"] * 0.8 + embeddings * 0.2
                )
                
            logger.debug("Updated word embeddings for style profile")
        except Exception as e:
            logger.error(f"Error updating embeddings: {e}")
    
    def analyze_style(self) -> Dict[str, Any]:
        """
        Analyze the current user communication style
        
        Returns:
            analysis: Dictionary with style analysis results
        """
        # Return current style profile with summary
        summary = {
            "writing_style": "formal" if self.style_profile["formality_score"] > 0.6 else "casual",
            "complexity": "complex" if self.style_profile["avg_sentence_length"] > 20 else "simple",
            "emoji_usage": "frequent" if self.style_profile["emoji_usage_rate"] > 0.05 else "rare",
            "top_words": sorted(
                self.style_profile["word_frequencies"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:20],
            "sample_phrases": self.style_profile["favorite_phrases"][:10],
            "mimicry_confidence": min(
                len(self.communication_samples) / 100, 1.0
            )  # Confidence based on amount of training data
        }
        
        return {**self.style_profile, "summary": summary}
    
    def generate_text(self, prompt: str, context: str = None, max_length: int = None) -> Dict[str, Any]:
        """
        Generate text in the user's style
        
        Args:
            prompt: Text prompt to start generation
            context: Context of the generation (e.g., "email", "chat")
            max_length: Maximum length of generated text
        
        Returns:
            result: Dictionary with generated text and metadata
                {
                    "text": str,
                    "confidence": float,
                    "model_used": str
                }
        """
        # Check if enabled and consent given
        if not self.settings.get("enabled") or not self.settings.get("consent_given"):
            return {
                "text": "",
                "confidence": 0.0,
                "model_used": None,
                "error": "Communication clone is disabled or consent not given"
            }
        
        # Check context restrictions
        if context and context in self.settings.get("prohibited_contexts", []):
            return {
                "text": "",
                "confidence": 0.0,
                "model_used": None,
                "error": f"Generation in {context} context is prohibited by user settings"
            }
        
        # Set default max length if not provided
        if max_length is None:
            max_length = self.settings.get("max_token_length", 500)
        
        # Choose generation method
        preferred_model = self.settings.get("preferred_model", "local")
        
        # Try preferred method, fall back to others if not available
        result = None
        
        if preferred_model == "openai" and self.openai_client:
            result = self._generate_text_with_openai(prompt, context, max_length)
        elif preferred_model == "local" and TRANSFORMERS_AVAILABLE:
            result = self._generate_text_with_transformers(prompt, context, max_length)
        elif preferred_model == "rule_based":
            result = self._generate_text_rule_based(prompt, context, max_length)
        
        # Fall back if preferred method failed
        if not result and TRANSFORMERS_AVAILABLE:
            result = self._generate_text_with_transformers(prompt, context, max_length)
        if not result and self.openai_client:
            result = self._generate_text_with_openai(prompt, context, max_length)
        if not result:
            result = self._generate_text_rule_based(prompt, context, max_length)
        
        return result
    
    def _generate_text_with_openai(self, prompt: str, context: str, max_length: int) -> Optional[Dict[str, Any]]:
        """Generate text using OpenAI API"""
        if not self.openai_client:
            return None
        
        try:
            # Create style guidance from profile
            style_guidance = (
                f"Please write in a {self.style_profile['formality_score'] > 0.5 and 'formal' or 'casual'} tone. "
                f"Use {'longer' if self.style_profile['avg_sentence_length'] > 15 else 'shorter'} sentences "
                f"averaging about {int(self.style_profile['avg_sentence_length'])} words per sentence. "
                f"Use {'complex' if self.style_profile['avg_word_length'] > 5 else 'simple'} vocabulary. "
            )
            
            # Add favorite phrases if available
            if self.style_profile["favorite_phrases"]:
                phrases = self.style_profile["favorite_phrases"][:5]
                style_guidance += f"Consider using phrases like: {', '.join(phrases)}. "
            
            # Add emoji guidance
            if self.style_profile["emoji_usage_rate"] > 0.02:
                style_guidance += f"Use emojis occasionally. "
            else:
                style_guidance += f"Don't use emojis. "
                
            # Create full prompt
            full_prompt = (
                f"{style_guidance}\n\n"
                f"Context: {context}\n\n"
                f"{prompt}"
            )
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are mimicking the writing style of a specific person. {style_guidance}"},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=max_length,
                temperature=0.7,
            )
            
            # Extract generated text
            generated_text = response.choices[0].message.content
            
            return {
                "text": generated_text,
                "confidence": 0.85,
                "model_used": "openai"
            }
        
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            return None
    
    def _generate_text_with_transformers(self, prompt: str, context: str, max_length: int) -> Optional[Dict[str, Any]]:
        """Generate text using local transformer models"""
        if not TRANSFORMERS_AVAILABLE or "gpt2_model" not in self.models:
            return None
        
        try:
            # Get models
            tokenizer = self.models["gpt2_tokenizer"]
            model = self.models["gpt2_model"]
            
            # Encode the prompt
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate text
            output = model.generate(
                inputs,
                max_length=min(max_length, 1024),
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
            
            # Decode the output
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Remove the input prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            # Apply style adjustments
            generated_text = self._apply_style_adjustments(generated_text)
            
            return {
                "text": generated_text,
                "confidence": 0.7,
                "model_used": "transformers"
            }
        
        except Exception as e:
            logger.error(f"Error generating text with transformers: {e}")
            return None
    
    def _generate_text_rule_based(self, prompt: str, context: str, max_length: int) -> Dict[str, Any]:
        """Generate text using rule-based methods (fallback)"""
        # This is a simplified implementation
        # In a real system, this would use more sophisticated techniques
        
        # Start with the prompt
        result = prompt
        
        # Generate some simple continuation
        if context == "email_draft":
            result += "\n\nThank you for your attention to this matter."
            
            # Add formal closing if the style is formal
            if self.style_profile["formality_score"] > 0.6:
                result += "\n\nBest regards,"
            else:
                result += "\n\nThanks,"
        
        elif context == "message_draft":
            result += " Let me know what you think."
            
            # Add emoji if the user uses them
            if self.style_profile["emoji_usage_rate"] > 0.02:
                result += " 👍"
        
        elif context == "note":
            result += "\n- Follow up later\n- Add more details"
        
        else:
            result += " I hope this helps."
        
        # Apply style adjustments
        result = self._apply_style_adjustments(result)
        
        # Truncate if too long
        if len(result) > max_length:
            result = result[:max_length]
        
        return {
            "text": result,
            "confidence": 0.3,
            "model_used": "rule_based"
        }
    
    def _apply_style_adjustments(self, text: str) -> str:
        """Apply style adjustments to match user's style"""
        # This is a simplified implementation
        # In a real system, this would use more sophisticated techniques
        
        # Replace rare words with more common words if style is simple
        if self.style_profile["avg_word_length"] < 4.5:
            text = text.replace("utilize", "use")
            text = text.replace("implement", "use")
            text = text.replace("assistance", "help")
            text = text.replace("therefore", "so")
        
        # Add occasional emoji if user uses them
        if self.style_profile["emoji_usage_rate"] > 0.05:
            text = text.replace("Thanks", "Thanks 😊")
            text = text.replace("great", "great 👍")
            text = text.replace("good", "good 👌")
        
        return text
    
    def get_settings(self) -> Dict[str, Any]:
        """Get the current settings"""
        return self.settings
    
    def update_settings(self, updated_settings: Dict[str, Any]) -> bool:
        """Update settings"""
        # Update only valid settings
        for key, value in updated_settings.items():
            if key in self.settings:
                # Special case for enabled: require consent to be true
                if key == "enabled" and value and not self.settings["consent_given"]:
                    continue
                self.settings[key] = value
        
        return True
