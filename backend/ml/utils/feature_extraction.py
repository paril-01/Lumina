import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import re
from datetime import datetime, timedelta
import json
import logging
from collections import Counter
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure nltk resources are available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

class FeatureExtractor:
    """
    Extracts features from various data sources for ML models.
    
    This class contains methods to extract meaningful features from emails,
    calendar events, chat messages, documents, and web activity for use in 
    behavior modeling and other ML tasks.
    """
    
    def __init__(self):
        """Initialize the feature extractor with necessary tools."""
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.english_stopwords = set(stopwords.words('english'))
    
    def extract_time_features(self, dt: Union[str, datetime]) -> Dict[str, Any]:
        """
        Extract time-based features from a datetime.
        
        Args:
            dt: Timestamp as string or datetime object
            
        Returns:
            Dictionary of time features
        """
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
            except ValueError:
                try:
                    dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    logger.warning(f"Could not parse timestamp: {dt}")
                    return {}
        
        features = {
            'hour': dt.hour,
            'minute': dt.minute,
            'day': dt.day,
            'month': dt.month,
            'year': dt.year,
            'weekday': dt.weekday(),
            'is_weekend': int(dt.weekday() >= 5),
            'is_working_hours': int(9 <= dt.hour < 18),
            'quarter': (dt.month - 1) // 3 + 1,
            'week_of_year': dt.isocalendar()[1],
            'day_of_year': dt.timetuple().tm_yday,
            'part_of_day': self._get_part_of_day(dt.hour)
        }
        
        return features
    
    def _get_part_of_day(self, hour: int) -> str:
        """
        Determine part of day from hour.
        
        Args:
            hour: Hour of the day (0-23)
            
        Returns:
            Part of day as string
        """
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'
    
    def extract_email_features(self, email: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from a single email.
        
        Args:
            email: Email object with fields like subject, body, etc.
            
        Returns:
            Dictionary of email features
        """
        features = {}
        
        # Basic metadata
        subject = email.get('subject', '')
        body = email.get('body', '')
        sender = email.get('sender', '')
        recipients = email.get('recipients', [])
        cc = email.get('cc', [])
        bcc = email.get('bcc', [])
        timestamp = email.get('timestamp')
        
        # Text features
        features['subject_length'] = len(subject)
        features['body_length'] = len(body)
        features['word_count'] = len(body.split())
        features['sentence_count'] = len(sent_tokenize(body)) if body else 0
        features['avg_word_length'] = np.mean([len(w) for w in body.split()]) if body else 0
        features['has_greeting'] = int(bool(re.search(r'(hi|hello|dear|greetings)', body.lower())))
        features['has_question'] = int('?' in body)
        features['has_urls'] = int(bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', body)))
        
        # Recipient features
        features['recipient_count'] = len(recipients)
        features['cc_count'] = len(cc)
        features['bcc_count'] = len(bcc)
        features['total_recipients'] = len(recipients) + len(cc) + len(bcc)
        
        # Sentiment analysis
        sentiment = self.sentiment_analyzer.polarity_scores(body)
        features['sentiment_pos'] = sentiment['pos']
        features['sentiment_neg'] = sentiment['neg']
        features['sentiment_neu'] = sentiment['neu']
        features['sentiment_compound'] = sentiment['compound']
        
        # Time features
        if timestamp:
            time_features = self.extract_time_features(timestamp)
            features.update(time_features)
        
        # Other features
        features['has_attachments'] = int(len(email.get('attachments', [])) > 0)
        features['attachment_count'] = len(email.get('attachments', []))
        features['is_reply'] = int(bool(re.match(r'^re:', subject.lower())))
        features['is_forward'] = int(bool(re.match(r'^fw:', subject.lower())))
        
        return features
    
    def extract_calendar_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from a calendar event.
        
        Args:
            event: Calendar event object
            
        Returns:
            Dictionary of calendar features
        """
        features = {}
        
        # Basic metadata
        title = event.get('title', '')
        description = event.get('description', '')
        location = event.get('location', '')
        start_time = event.get('start_time')
        end_time = event.get('end_time')
        participants = event.get('participants', [])
        
        # Text features
        features['title_length'] = len(title)
        features['description_length'] = len(description)
        features['has_location'] = int(len(location) > 0)
        
        # Time features
        if start_time and end_time:
            if isinstance(start_time, str):
                try:
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                except ValueError:
                    start_dt = None
            else:
                start_dt = start_time
                
            if isinstance(end_time, str):
                try:
                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                except ValueError:
                    end_dt = None
            else:
                end_dt = end_time
            
            if start_dt and end_dt:
                features['duration_minutes'] = (end_dt - start_dt).total_seconds() / 60
                features.update(self.extract_time_features(start_dt))
        
        # Participant features
        features['participant_count'] = len(participants)
        
        # Other features
        features['is_recurring'] = int(event.get('recurring', False))
        features['is_all_day'] = int(event.get('all_day', False))
        features['is_private'] = int(event.get('private', False))
        features['has_reminder'] = int(event.get('reminder', False))
        
        # Derived features
        features['is_meeting'] = int(features['participant_count'] > 1)
        
        # Keywords in title
        keywords = ['meeting', 'call', 'discussion', 'review', 'sync', 'interview', 'demo']
        for keyword in keywords:
            features[f'has_{keyword}'] = int(keyword.lower() in title.lower())
        
        return features
    
    def extract_chat_features(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from a chat message.
        
        Args:
            message: Chat message object
            
        Returns:
            Dictionary of chat features
        """
        features = {}
        
        # Basic metadata
        content = message.get('content', '')
        sender = message.get('sender', '')
        chat_id = message.get('chat_id', '')
        timestamp = message.get('timestamp')
        
        # Text features
        features['content_length'] = len(content)
        features['word_count'] = len(content.split())
        features['has_emoji'] = int(bool(re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]', content)))
        features['has_mention'] = int('@' in content)
        features['has_question'] = int('?' in content)
        features['has_exclamation'] = int('!' in content)
        features['has_urls'] = int(bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)))
        
        # Count mentions
        mention_pattern = r'@\w+'
        mentions = re.findall(mention_pattern, content)
        features['mention_count'] = len(mentions)
        
        # Sentiment analysis
        sentiment = self.sentiment_analyzer.polarity_scores(content)
        features['sentiment_pos'] = sentiment['pos']
        features['sentiment_neg'] = sentiment['neg']
        features['sentiment_neu'] = sentiment['neu']
        features['sentiment_compound'] = sentiment['compound']
        
        # Time features
        if timestamp:
            time_features = self.extract_time_features(timestamp)
            features.update(time_features)
        
        # Other features
        features['has_attachment'] = int(len(message.get('attachments', [])) > 0)
        features['is_reply'] = int(message.get('reply_to', '') != '')
        
        return features
    
    def extract_web_activity_features(self, activity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from web browsing activity.
        
        Args:
            activity: Web activity object
            
        Returns:
            Dictionary of web activity features
        """
        features = {}
        
        # Basic metadata
        url = activity.get('url', '')
        title = activity.get('title', '')
        timestamp = activity.get('timestamp')
        
        # URL features
        domain = self._extract_domain(url)
        features['domain'] = domain
        features['url_length'] = len(url)
        features['path_depth'] = url.count('/') - 2 if '://' in url else url.count('/')
        features['has_query_params'] = int('?' in url)
        
        # Common domains
        common_domains = ['google.com', 'youtube.com', 'facebook.com', 'amazon.com', 
                         'twitter.com', 'linkedin.com', 'github.com', 'stackoverflow.com']
        for d in common_domains:
            features[f'is_{d.split(".")[0]}'] = int(d in domain)
        
        # Content type features
        features['is_search'] = int(bool(re.search(r'(google|bing|yahoo|search).*?q=', url.lower())))
        features['is_video'] = int(bool(re.search(r'(youtube|vimeo|netflix|video)', url.lower())))
        features['is_social'] = int(bool(re.search(r'(facebook|twitter|instagram|linkedin)', url.lower())))
        features['is_shopping'] = int(bool(re.search(r'(amazon|ebay|shop|product|cart)', url.lower())))
        features['is_news'] = int(bool(re.search(r'(news|cnn|bbc|nytimes)', url.lower())))
        features['is_email'] = int(bool(re.search(r'(gmail|outlook|mail)', url.lower())))
        features['is_docs'] = int(bool(re.search(r'(docs|sheets|slides|office)', url.lower())))
        
        # Time features
        if timestamp:
            time_features = self.extract_time_features(timestamp)
            features.update(time_features)
        
        return features
    
    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL.
        
        Args:
            url: URL string
            
        Returns:
            Domain string
        """
        if not url:
            return ''
        
        try:
            if '://' in url:
                domain = url.split('://')[1].split('/')[0]
            else:
                domain = url.split('/')[0]
            
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
                
            return domain
        except Exception:
            return ''
    
    def build_user_interaction_graph(self, emails: List[Dict[str, Any]], 
                                   chats: List[Dict[str, Any]]) -> nx.Graph:
        """
        Build a graph of user interactions from emails and chats.
        
        Args:
            emails: List of email objects
            chats: List of chat message objects
            
        Returns:
            NetworkX graph of user interactions
        """
        G = nx.Graph()
        
        # Process emails
        for email in emails:
            sender = email.get('sender', '')
            recipients = email.get('recipients', []) + email.get('cc', []) + email.get('bcc', [])
            
            if sender and recipients:
                for recipient in recipients:
                    if sender != recipient:  # Avoid self-loops
                        if G.has_edge(sender, recipient):
                            G[sender][recipient]['weight'] += 1
                        else:
                            G.add_edge(sender, recipient, weight=1)
        
        # Process chats
        for chat in chats:
            sender = chat.get('sender', '')
            chat_id = chat.get('chat_id', '')
            mentions = re.findall(r'@(\w+)', chat.get('content', ''))
            
            # Add edges for direct mentions
            if sender and mentions:
                for mention in mentions:
                    if sender != mention:  # Avoid self-loops
                        if G.has_edge(sender, mention):
                            G[sender][mention]['weight'] += 1
                        else:
                            G.add_edge(sender, mention, weight=1)
        
        return G
    
    def extract_interaction_features(self, G: nx.Graph, user: str) -> Dict[str, Any]:
        """
        Extract network features for a specific user.
        
        Args:
            G: NetworkX graph of user interactions
            user: User to extract features for
            
        Returns:
            Dictionary of interaction features
        """
        features = {}
        
        if user not in G:
            return {
                'degree_centrality': 0,
                'betweenness_centrality': 0,
                'closeness_centrality': 0,
                'eigenvector_centrality': 0,
                'clustering_coefficient': 0,
                'total_interactions': 0,
                'unique_contacts': 0
            }
        
        # Node centrality metrics
        features['degree_centrality'] = nx.degree_centrality(G)[user]
        
        # Only calculate these for connected graphs
        if nx.is_connected(G):
            features['betweenness_centrality'] = nx.betweenness_centrality(G)[user]
            features['closeness_centrality'] = nx.closeness_centrality(G)[user]
        else:
            # For disconnected graphs, calculate on user's component
            user_component = nx.node_connected_component(G, user)
            subgraph = G.subgraph(user_component)
            if len(subgraph) > 1:
                features['betweenness_centrality'] = nx.betweenness_centrality(subgraph)[user]
                features['closeness_centrality'] = nx.closeness_centrality(subgraph)[user]
            else:
                features['betweenness_centrality'] = 0
                features['closeness_centrality'] = 0
        
        try:
            features['eigenvector_centrality'] = nx.eigenvector_centrality(G)[user]
        except nx.PowerIterationFailedConvergence:
            features['eigenvector_centrality'] = 0
            
        features['clustering_coefficient'] = nx.clustering(G, user)
        
        # Interaction counts
        features['total_interactions'] = sum(G[user][nbr]['weight'] for nbr in G[user])
        features['unique_contacts'] = len(G[user])
        
        return features
    
    def extract_sequence_features(self, activities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract features from a sequence of user activities.
        
        Args:
            activities: List of activity dictionaries
            
        Returns:
            Dictionary of sequence features
        """
        if not activities:
            return {}
        
        features = {}
        
        # Activity types
        activity_types = [activity.get('type') for activity in activities]
        type_counts = Counter(activity_types)
        
        # Add count for each activity type
        for activity_type, count in type_counts.items():
            if activity_type:
                features[f'{activity_type}_count'] = count
        
        # Time-based features
        timestamps = []
        for activity in activities:
            ts = activity.get('timestamp')
            if ts:
                if isinstance(ts, str):
                    try:
                        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        timestamps.append(dt)
                    except ValueError:
                        pass
                elif isinstance(ts, datetime):
                    timestamps.append(ts)
        
        if timestamps:
            # Sort timestamps
            timestamps.sort()
            
            # Session duration
            if len(timestamps) > 1:
                start_time = timestamps[0]
                end_time = timestamps[-1]
                duration_seconds = (end_time - start_time).total_seconds()
                features['session_duration_seconds'] = duration_seconds
                
                # Average time between activities
                time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                             for i in range(len(timestamps) - 1)]
                features['avg_time_between_activities'] = np.mean(time_diffs)
                features['max_time_between_activities'] = np.max(time_diffs)
                features['min_time_between_activities'] = np.min(time_diffs)
                features['std_time_between_activities'] = np.std(time_diffs)
            
            # Time of day distribution
            hours = [ts.hour for ts in timestamps]
            features['morning_activities'] = sum(1 for h in hours if 5 <= h < 12)
            features['afternoon_activities'] = sum(1 for h in hours if 12 <= h < 17)
            features['evening_activities'] = sum(1 for h in hours if 17 <= h < 21)
            features['night_activities'] = sum(1 for h in hours if h >= 21 or h < 5)
            
            # Day of week distribution
            weekdays = [ts.weekday() for ts in timestamps]
            features['weekday_activities'] = sum(1 for d in weekdays if d < 5)
            features['weekend_activities'] = sum(1 for d in weekdays if d >= 5)
        
        # Sequence pattern features
        if len(activity_types) > 1:
            # Transition counts between activity types
            transitions = {}
            for i in range(len(activity_types) - 1):
                transition = (activity_types[i], activity_types[i+1])
                transitions[transition] = transitions.get(transition, 0) + 1
            
            # Most common transitions
            common_transitions = Counter(transitions).most_common(3)
            for i, ((from_type, to_type), count) in enumerate(common_transitions):
                features[f'common_transition_{i+1}'] = f'{from_type}_to_{to_type}'
                features[f'common_transition_{i+1}_count'] = count
        
        # Total activities
        features['total_activities'] = len(activities)
        
        return features
