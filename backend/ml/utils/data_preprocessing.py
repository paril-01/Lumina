import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import json
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

class DataPreprocessor:
    """
    Preprocesses data for ML models in the AI Assistant.
    
    Handles data cleaning, transformation, normalization, and feature engineering
    for various data types (text, time series, user actions, etc.)
    """
    
    def __init__(self):
        """Initialize the data preprocessor with necessary tools."""
        self.text_scaler = None
        self.time_scaler = None
        self.numerical_scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = None
        self.english_stopwords = set(stopwords.words('english'))
    
    def clean_text(self, texts: List[str], remove_stopwords: bool = True, 
                  lemmatize: bool = True) -> List[str]:
        """
        Clean and normalize text data.
        
        Args:
            texts: List of text strings to clean
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to apply lemmatization
            
        Returns:
            List of cleaned texts
        """
        cleaned_texts = []
        
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
                
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and numbers
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords if requested
            if remove_stopwords:
                tokens = [token for token in tokens if token not in self.english_stopwords]
            
            # Lemmatize if requested
            if lemmatize:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            # Join tokens back into a string
            cleaned_text = ' '.join(tokens)
            cleaned_texts.append(cleaned_text)
        
        return cleaned_texts
    
    def vectorize_text(self, texts: List[str], max_features: int = 1000, 
                      fit: bool = True) -> np.ndarray:
        """
        Convert text to TF-IDF feature vectors.
        
        Args:
            texts: List of text strings to vectorize
            max_features: Maximum number of features for TF-IDF
            fit: Whether to fit the vectorizer on this data
            
        Returns:
            TF-IDF feature matrix
        """
        if fit or self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english'
            )
            return self.tfidf_vectorizer.fit_transform(texts).toarray()
        else:
            return self.tfidf_vectorizer.transform(texts).toarray()
    
    def normalize_numerical(self, data: np.ndarray, method: str = 'standard', 
                          fit: bool = True) -> np.ndarray:
        """
        Normalize numerical data.
        
        Args:
            data: Numerical data to normalize
            method: Normalization method ('standard' or 'minmax')
            fit: Whether to fit the scaler on this data
            
        Returns:
            Normalized data
        """
        if method == 'standard':
            if fit:
                return self.numerical_scaler.fit_transform(data)
            else:
                return self.numerical_scaler.transform(data)
        elif method == 'minmax':
            if fit:
                return self.min_max_scaler.fit_transform(data)
            else:
                return self.min_max_scaler.transform(data)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def process_timestamps(self, timestamps: List[Union[str, datetime]], 
                         extract_features: bool = True) -> np.ndarray:
        """
        Process timestamp data for ML models.
        
        Args:
            timestamps: List of timestamps (string or datetime)
            extract_features: Whether to extract features (hour, day, etc.)
            
        Returns:
            Processed timestamp features
        """
        # Convert string timestamps to datetime objects
        datetime_objects = []
        for ts in timestamps:
            if isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    datetime_objects.append(dt)
                except ValueError:
                    try:
                        dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                        datetime_objects.append(dt)
                    except ValueError:
                        logger.warning(f"Could not parse timestamp: {ts}")
                        datetime_objects.append(None)
            else:
                datetime_objects.append(ts)
        
        if not extract_features:
            # Convert to unix timestamps
            unix_timestamps = [
                dt.timestamp() if dt is not None else np.nan 
                for dt in datetime_objects
            ]
            return np.array(unix_timestamps).reshape(-1, 1)
        
        # Extract features
        features = []
        for dt in datetime_objects:
            if dt is None:
                features.append([np.nan] * 6)
                continue
                
            hour = dt.hour
            minute = dt.minute
            day = dt.day
            month = dt.month
            weekday = dt.weekday()
            quarter = (dt.month - 1) // 3 + 1
            
            features.append([hour, minute, day, month, weekday, quarter])
        
        return np.array(features)
    
    def create_sequence_data(self, data: List[Any], sequence_length: int,
                            step: int = 1) -> Tuple[List[List[Any]], List[Any]]:
        """
        Create sequence data for time series or sequential models.
        
        Args:
            data: List of data points
            sequence_length: Length of each sequence
            step: Step size between sequences
            
        Returns:
            X, y: Sequences and target values
        """
        X, y = [], []
        for i in range(0, len(data) - sequence_length, step):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        
        return X, y
    
    def one_hot_encode(self, categories: List[Any], 
                     all_categories: Optional[List[Any]] = None) -> np.ndarray:
        """
        One-hot encode categorical data.
        
        Args:
            categories: List of categories to encode
            all_categories: Optional list of all possible categories
            
        Returns:
            One-hot encoded matrix
        """
        if all_categories is None:
            all_categories = sorted(list(set(categories)))
        
        result = np.zeros((len(categories), len(all_categories)))
        for i, category in enumerate(categories):
            if category in all_categories:
                idx = all_categories.index(category)
                result[i, idx] = 1
        
        return result
    
    def extract_email_features(self, emails: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract features from email data.
        
        Args:
            emails: List of email dictionaries
            
        Returns:
            DataFrame of extracted features
        """
        features = []
        
        for email in emails:
            # Basic counts
            subject_length = len(email.get('subject', ''))
            body_length = len(email.get('body', ''))
            word_count = len(email.get('body', '').split())
            
            # Time features
            send_time = email.get('timestamp')
            if isinstance(send_time, str):
                try:
                    dt = datetime.fromisoformat(send_time.replace('Z', '+00:00'))
                    hour = dt.hour
                    weekday = dt.weekday()
                except ValueError:
                    hour = None
                    weekday = None
            else:
                hour = None
                weekday = None
            
            # Recipients
            num_recipients = len(email.get('recipients', []))
            has_attachments = int(len(email.get('attachments', [])) > 0)
            
            # Priority/flags
            is_important = int(email.get('important', False))
            
            row = {
                'subject_length': subject_length,
                'body_length': body_length,
                'word_count': word_count,
                'send_hour': hour,
                'send_weekday': weekday,
                'num_recipients': num_recipients,
                'has_attachments': has_attachments,
                'is_important': is_important
            }
            
            features.append(row)
        
        return pd.DataFrame(features)
    
    def extract_calendar_features(self, events: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract features from calendar events.
        
        Args:
            events: List of calendar event dictionaries
            
        Returns:
            DataFrame of extracted features
        """
        features = []
        
        for event in events:
            # Basic info
            title_length = len(event.get('title', ''))
            description_length = len(event.get('description', ''))
            
            # Time features
            start_time = event.get('start_time')
            end_time = event.get('end_time')
            
            if isinstance(start_time, str) and isinstance(end_time, str):
                try:
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    
                    duration_minutes = (end_dt - start_dt).total_seconds() / 60
                    hour = start_dt.hour
                    weekday = start_dt.weekday()
                    is_weekend = int(weekday >= 5)
                    is_working_hours = int(9 <= hour < 18)
                except ValueError:
                    duration_minutes = None
                    hour = None
                    weekday = None
                    is_weekend = None
                    is_working_hours = None
            else:
                duration_minutes = None
                hour = None
                weekday = None
                is_weekend = None
                is_working_hours = None
            
            # Participants
            num_participants = len(event.get('participants', []))
            
            # Other features
            is_recurring = int(event.get('recurring', False))
            is_all_day = int(event.get('all_day', False))
            has_location = int(len(event.get('location', '')) > 0)
            
            row = {
                'title_length': title_length,
                'description_length': description_length,
                'duration_minutes': duration_minutes,
                'start_hour': hour,
                'start_weekday': weekday,
                'is_weekend': is_weekend,
                'is_working_hours': is_working_hours,
                'num_participants': num_participants,
                'is_recurring': is_recurring,
                'is_all_day': is_all_day,
                'has_location': has_location
            }
            
            features.append(row)
        
        return pd.DataFrame(features)
    
    def segment_user_activity(self, activities: List[Dict[str, Any]], 
                            window_size: timedelta = timedelta(minutes=30)) -> List[List[Dict[str, Any]]]:
        """
        Segment user activities into sessions based on time windows.
        
        Args:
            activities: List of activity dictionaries
            window_size: Time window for segmentation
            
        Returns:
            List of activity segments
        """
        if not activities:
            return []
        
        # Sort activities by timestamp
        sorted_activities = sorted(activities, key=lambda x: x.get('timestamp', ''))
        
        segments = []
        current_segment = [sorted_activities[0]]
        
        for i in range(1, len(sorted_activities)):
            current = sorted_activities[i]
            previous = sorted_activities[i-1]
            
            current_time = None
            previous_time = None
            
            if isinstance(current.get('timestamp'), str):
                try:
                    current_time = datetime.fromisoformat(current['timestamp'].replace('Z', '+00:00'))
                except ValueError:
                    pass
            elif isinstance(current.get('timestamp'), datetime):
                current_time = current['timestamp']
                
            if isinstance(previous.get('timestamp'), str):
                try:
                    previous_time = datetime.fromisoformat(previous['timestamp'].replace('Z', '+00:00'))
                except ValueError:
                    pass
            elif isinstance(previous.get('timestamp'), datetime):
                previous_time = previous['timestamp']
            
            if current_time and previous_time:
                time_diff = current_time - previous_time
                if time_diff > window_size:
                    segments.append(current_segment)
                    current_segment = [current]
                else:
                    current_segment.append(current)
            else:
                current_segment.append(current)
        
        # Add the last segment
        if current_segment:
            segments.append(current_segment)
        
        return segments
