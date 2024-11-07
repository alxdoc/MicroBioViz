import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from typing import List, Dict, Tuple
import networkx as nx
import numpy as np
import logging

class TextAnalyzer:
    def __init__(self, df: pd.DataFrame):
        try:
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)
        except Exception as e:
            print(f"Warning: Failed to initialize logger: {str(e)}")
            self.logger = None
            
        self.df = df
        self._initialize_nltk()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['study', 'research', 'analysis', 'results'])
    
    def _log(self, level: str, message: str):
        """Safe logging wrapper"""
        try:
            if hasattr(self, 'logger') and self.logger is not None:
                if level == 'debug':
                    self.logger.debug(message)
                elif level == 'warning':
                    self.logger.warning(message)
                elif level == 'error':
                    self.logger.error(message)
        except Exception:
            print(f"{level.upper()}: {message}")
    
    def _initialize_nltk(self):
        """Initialize NLTK resources with error handling"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self._log('debug', "Successfully initialized NLTK resources")
        except Exception as e:
            self._log('error', f"Error downloading NLTK resources: {str(e)}")
            raise Exception("Failed to initialize NLTK resources")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for analysis"""
        if not isinstance(text, str) or not text.strip():
            self._log('debug', "Empty or invalid text input for preprocessing")
            return []
        try:
            self._log('debug', "Starting text preprocessing")
            # Tokenize
            tokens = word_tokenize(text.lower())
            # Remove stopwords and non-alphabetic tokens
            tokens = [token for token in tokens 
                     if token.isalpha() 
                     and token not in self.stop_words 
                     and len(token) > 2]
            self._log('debug', f"Preprocessed {len(tokens)} valid tokens")
            return tokens
        except Exception as e:
            self._log('error', f"Error preprocessing text: {str(e)}")
            return []
    
    def get_common_themes(self, filtered_df: pd.DataFrame, top_n: int = 10) -> Dict[str, int]:
        """Extract common themes from article descriptions"""
        try:
            self._log('debug', "Starting theme extraction")
            
            # Validate input data
            if filtered_df.empty or 'article description' not in filtered_df.columns:
                self._log('warning', "Empty dataframe or missing article description column")
                return {'no data': 1}
            
            # Combine all valid descriptions
            descriptions = filtered_df['article description'].dropna()
            if descriptions.empty:
                self._log('warning', "No valid article descriptions found")
                return {'no data': 1}
            
            all_text = ' '.join(descriptions)
            if not all_text.strip():
                self._log('warning', "Empty combined text after joining descriptions")
                return {'no data': 1}
            
            # Process text
            self._log('debug', "Processing combined text")
            tokens = self._preprocess_text(all_text)
            
            # Handle empty tokens
            if not tokens:
                self._log('warning', "No valid tokens found after preprocessing")
                return {'no data': 1}
            
            # Calculate word frequencies
            word_freq = Counter(tokens)
            result = dict(word_freq.most_common(top_n))
            
            self._log('debug', f"Extracted {len(result)} themes")
            return result if result else {'no data': 1}
            
        except Exception as e:
            self._log('error', f"Error in theme extraction: {str(e)}")
            return {'error': 1}
    
    def get_keyword_context(self, keyword: str) -> List[str]:
        """Get context around keyword mentions"""
        contexts = []
        for desc in self.df['article description'].dropna():
            if keyword.lower() in desc.lower():
                try:
                    sentences = nltk.sent_tokenize(desc)
                    for sentence in sentences:
                        if keyword.lower() in sentence.lower():
                            contexts.append(sentence)
                except Exception as e:
                    self._log('error', f"Error processing keyword context: {str(e)}")
        return contexts
    
    def get_technology_network(self, filtered_df: pd.DataFrame) -> Tuple[List[str], List[Tuple[str, str, float]]]:
        """Generate technology co-occurrence network data"""
        try:
            tech_pairs = []
            all_techs = set()
            
            for techs in filtered_df['Technologies'].dropna():
                if not isinstance(techs, str):
                    continue
                tech_list = techs.split(';')
                if len(tech_list) < 2:
                    continue
                for i in range(len(tech_list)):
                    for j in range(i + 1, len(tech_list)):
                        if tech_list[i] and tech_list[j]:
                            tech_pairs.append((tech_list[i], tech_list[j]))
                            all_techs.add(tech_list[i])
                            all_techs.add(tech_list[j])
            
            # Calculate edge weights based on co-occurrence frequency
            edge_weights = Counter(tech_pairs)
            edges = [(t1, t2, weight) for (t1, t2), weight in edge_weights.items()]
            
            return list(all_techs), edges
        except Exception as e:
            self._log('error', f"Error generating technology network: {str(e)}")
            return [], []
    
    def analyze_topic_trends(self, filtered_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze topic trends over time"""
        try:
            if 'Year' not in filtered_df.columns:
                self._log('warning', "Year column not found for topic trends analysis")
                return pd.DataFrame()
                
            topic_by_year = pd.crosstab(filtered_df['Year'], filtered_df['rank'], normalize='index')
            return topic_by_year.reset_index()
        except Exception as e:
            self._log('error', f"Error analyzing topic trends: {str(e)}")
            return pd.DataFrame()
    
    def get_trl_year_correlation(self, filtered_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlation between TRL and publication year"""
        result = {
            'correlation': 0.0,
            'avg_trl_by_year': {},
            'has_sufficient_data': False
        }
        
        try:
            # Check if required columns exist
            if 'TRL' not in filtered_df.columns or 'Year' not in filtered_df.columns:
                self._log('warning', "Missing required columns for TRL-Year correlation")
                return result
            
            # Get valid data (non-null values)
            valid_data = filtered_df[['Year', 'TRL']].dropna()
            
            # Check if we have enough data points
            if len(valid_data) < 2:
                self._log('warning', "Insufficient data points for TRL-Year correlation")
                return result
            
            # Calculate correlation
            correlation = valid_data['Year'].corr(valid_data['TRL'])
            result['correlation'] = correlation if not pd.isna(correlation) else 0.0
            
            # Calculate average TRL by year
            avg_trl = valid_data.groupby('Year')['TRL'].mean()
            if not avg_trl.empty:
                result['avg_trl_by_year'] = avg_trl.to_dict()
                result['has_sufficient_data'] = True
            
            self._log('debug', f"Calculated TRL-Year correlation: {result['correlation']:.2f}")
            return result
            
        except Exception as e:
            self._log('error', f"Error calculating TRL-Year correlation: {str(e)}")
            return result
