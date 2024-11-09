from typing import Dict, Any, List, Counter, Tuple
from collections import Counter
import pandas as pd
import numpy as np
from scipy import stats
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import networkx as nx
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

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

    def _initialize_nltk(self) -> None:
        """Initialize NLTK resources with error handling"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self._log('debug', "Successfully initialized NLTK resources")
        except Exception as e:
            self._log('error', f"Error downloading NLTK resources: {str(e)}")
            raise Exception("Failed to initialize NLTK resources")

    def _log(self, level: str, message: str) -> None:
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

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for analysis"""
        if not isinstance(text, str) or not text.strip():
            self._log('debug', "Empty or invalid text input for preprocessing")
            return []
        try:
            self._log('debug', "Starting text preprocessing")
            tokens = word_tokenize(text.lower())
            tokens = [token for token in tokens 
                     if token.isalpha() 
                     and token not in self.stop_words 
                     and len(token) > 2]
            self._log('debug', f"Preprocessed {len(tokens)} valid tokens")
            return tokens
        except Exception as e:
            self._log('error', f"Error preprocessing text: {str(e)}")
            return []

    def analyze_text_patterns(self, text_column: str) -> Dict[str, Any]:
        """Detect patterns in text data using NLP techniques."""
        try:
            if text_column not in self.df.columns:
                return {
                    'error': f"Column {text_column} not found",
                    'patterns': {}
                }

            texts = self.df[text_column].dropna().astype(str)
            if texts.empty:
                return {
                    'error': "No valid text data",
                    'patterns': {}
                }

            # Analyze text length patterns
            lengths = texts.str.len()
            length_stats = {
                'mean': lengths.mean(),
                'std': lengths.std(),
                'min': lengths.min(),
                'max': lengths.max()
            }

            # Analyze word frequency patterns
            all_words = []
            for text in texts:
                words = self._preprocess_text(text)
                all_words.extend(words)

            word_freq = Counter(all_words)
            common_patterns = dict(word_freq.most_common(10))

            return {
                'error': None,
                'patterns': {
                    'length_statistics': length_stats,
                    'common_patterns': common_patterns
                }
            }

        except Exception as e:
            return {
                'error': str(e),
                'patterns': {}
            }

    def analyze_topic_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between topics and other variables."""
        try:
            if 'rank' not in self.df.columns:
                return {
                    'error': "Topic column not found",
                    'correlations': {}
                }

            results = {
                'topic_trl_correlation': {},
                'topic_year_distribution': {},
                'topic_technology_association': {}
            }

            # Analyze topic-TRL relationships
            if 'TRL' in self.df.columns:
                topic_trl = self.df.groupby('rank')['TRL'].agg(['mean', 'std']).to_dict('index')
                results['topic_trl_correlation'] = topic_trl

            # Analyze topic distribution over years
            if 'Year' in self.df.columns:
                topic_year = pd.crosstab(
                    self.df['Year'],
                    self.df['rank'],
                    normalize='index'
                ).to_dict()
                results['topic_year_distribution'] = topic_year

            # Analyze topic-technology associations
            if 'Technologies' in self.df.columns:
                topic_tech = defaultdict(lambda: defaultdict(int))
                for _, row in self.df.iterrows():
                    if pd.notna(row['Technologies']):
                        topic = row['rank']
                        techs = row['Technologies'].split(';')
                        for tech in techs:
                            if tech:
                                topic_tech[topic][tech] += 1
                results['topic_technology_association'] = dict(topic_tech)

            return {
                'error': None,
                'correlations': results
            }

        except Exception as e:
            return {
                'error': str(e),
                'correlations': {}
            }

    def analyze_trl_progression(self) -> Dict[str, Any]:
        """Analyze TRL progression patterns over time and across topics."""
        try:
            if 'TRL' not in self.df.columns or 'Year' not in self.df.columns:
                return {
                    'error': "Required columns missing",
                    'progression': {}
                }

            results = {}

            # Calculate yearly TRL progression
            yearly_progression = self.df.groupby('Year')['TRL'].agg([
                'mean',
                'std',
                'count'
            ]).to_dict('index')
            results['yearly_progression'] = yearly_progression

            # Calculate TRL progression by topic
            topic_progression = self.df.groupby(['rank', 'Year'])['TRL'].mean().unstack().to_dict()
            results['topic_progression'] = topic_progression

            # Identify TRL trends
            valid_data = self.df[['Year', 'TRL']].dropna()
            if len(valid_data) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    valid_data['Year'],
                    valid_data['TRL']
                )
                results['trend_analysis'] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'std_err': std_err
                }

            return {
                'error': None,
                'progression': results
            }

        except Exception as e:
            return {
                'error': str(e),
                'progression': {}
            }

    def get_common_themes(self, filtered_df: pd.DataFrame, top_n: int = 10) -> Dict[str, int]:
        """Extract common themes from article descriptions"""
        try:
            self._log('debug', "Starting theme extraction")
            
            if filtered_df.empty or 'article description' not in filtered_df.columns:
                self._log('warning', "Empty dataframe or missing article description column")
                return {'no data': 1}
            
            descriptions = filtered_df['article description'].dropna()
            if descriptions.empty:
                self._log('warning', "No valid article descriptions found")
                return {'no data': 1}
            
            all_text = ' '.join(descriptions)
            if not all_text.strip():
                self._log('warning', "Empty combined text after joining descriptions")
                return {'no data': 1}
            
            self._log('debug', "Processing combined text")
            tokens = self._preprocess_text(all_text)
            
            if not tokens:
                self._log('warning', "No valid tokens found after preprocessing")
                return {'no data': 1}
            
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

    def get_trl_year_correlation(self, filtered_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation between TRL and publication year"""
        result = {
            'correlation': 0.0,
            'avg_trl_by_year': {},
            'has_sufficient_data': False
        }
        
        try:
            if 'TRL' not in filtered_df.columns or 'Year' not in filtered_df.columns:
                self._log('warning', "Missing required columns for TRL-Year correlation")
                return result
            
            valid_data = filtered_df[['Year', 'TRL']].dropna()
            
            if len(valid_data) < 2:
                self._log('warning', "Insufficient data points for TRL-Year correlation")
                return result
            
            correlation = valid_data['Year'].corr(valid_data['TRL'])
            result['correlation'] = correlation if not pd.isna(correlation) else 0.0
            
            avg_trl = valid_data.groupby('Year')['TRL'].mean()
            if not avg_trl.empty:
                result['avg_trl_by_year'] = avg_trl.to_dict()
                result['has_sufficient_data'] = True
            
            self._log('debug', f"Calculated TRL-Year correlation: {result['correlation']:.2f}")
            return result
            
        except Exception as e:
            self._log('error', f"Error calculating TRL-Year correlation: {str(e)}")
            return result

    def analyze_semantic_relationships(self, text_column: str) -> Dict[str, Any]:
        """Analyze semantic relationships between texts using TF-IDF and clustering"""
        try:
            # Get non-empty texts
            texts = self.df[text_column].dropna().astype(str)
            if texts.empty:
                return {'error': 'No valid text data'}
                
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words=list(self.stop_words))
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate similarity matrix
            similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
            
            # Perform hierarchical clustering
            n_clusters = min(10, len(texts))
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            clusters = clustering.fit_predict(similarity_matrix)
            
            # Group similar texts
            clustered_texts = {}
            for idx, cluster in enumerate(clusters):
                if cluster not in clustered_texts:
                    clustered_texts[cluster] = []
                clustered_texts[cluster].append({
                    'text': texts.iloc[idx],
                    'similarity_score': np.mean(similarity_matrix[idx])
                })
                
            return {
                'error': None,
                'clusters': clustered_texts,
                'similarity_matrix': similarity_matrix,
                'feature_names': vectorizer.get_feature_names_out()
            }
            
        except Exception as e:
            return {'error': str(e)}