import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from typing import Dict, List, Any

class TextAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.stop_words = ['и', 'в', 'на', 'с', 'по', 'для', 'the', 'and', 'in', 'of', 'to', 'a', 'is']
        
    def analyze_text(self, column: str) -> Dict[str, Any]:
        try:
            if column not in self.df.columns:
                return {'error': f'Column {column} not found'}
                
            texts = self.df[column].dropna().astype(str)
            if texts.empty:
                return {'error': 'No text data available'}
                
            # Word frequency analysis
            words = ' '.join(texts).lower().split()
            word_freq = Counter(words)
            for stop_word in self.stop_words:
                word_freq.pop(stop_word, None)
                
            return {
                'word_frequencies': dict(word_freq.most_common(20)),
                'total_words': len(words),
                'unique_words': len(word_freq)
            }
            
        except Exception as e:
            return {'error': str(e)}
            
    def semantic_search(self, query: str, text_columns: List[str]) -> Dict[str, Any]:
        try:
            results = {}
            total_mentions = 0
            mentions_by_column = {}
            
            vectorizer = TfidfVectorizer(stop_words=self.stop_words)
            
            for column in text_columns:
                if column not in self.df.columns:
                    continue
                    
                texts = self.df[column].fillna('').astype(str)
                if texts.empty:
                    continue
                    
                tfidf_matrix = vectorizer.fit_transform(texts)
                query_vector = vectorizer.transform([query])
                
                similarities = (query_vector * tfidf_matrix.T).toarray()[0]
                
                threshold = 0.3
                matches = [(idx, score) for idx, score in enumerate(similarities) if score > threshold]
                
                if matches:
                    mentions_by_column[column] = {
                        'count': len(matches),
                        'mentions': [
                            {
                                'text': texts.iloc[idx],
                                'similarity': score,
                                'metadata': {
                                    key: self.df.iloc[idx][key] 
                                    for key in ['Title', 'Authors', 'Year'] 
                                    if key in self.df.columns
                                }
                            }
                            for idx, score in sorted(matches, key=lambda x: x[1], reverse=True)
                        ]
                    }
                    total_mentions += len(matches)
            
            return {
                'total_mentions': total_mentions,
                'mentions_by_column': mentions_by_column
            }
            
        except Exception as e:
            return {'error': str(e)}
