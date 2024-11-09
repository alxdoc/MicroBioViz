import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from typing import Dict, List, Any
import re

class TextAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Enhanced stop words lists
        self.ru_stop_words = ['и', 'в', 'на', 'с', 'по', 'для', 'не', 'что', 'это', 'как', 'от', 'к', 'из']
        self.en_stop_words = ['the', 'and', 'in', 'of', 'to', 'a', 'is', 'that', 'for', 'on', 'with']
        
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
            for stop_word in self.ru_stop_words + self.en_stop_words:
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
            from sklearn.feature_extraction.text import TfidfVectorizer
            from langdetect import detect
            import numpy as np
            
            # Preprocessing function
            def preprocess_text(text: str) -> str:
                # Convert to lowercase
                text = text.lower()
                # Remove special characters but keep spaces between words
                text = re.sub(r'[^\w\s]', ' ', text)
                # Remove extra whitespace
                text = ' '.join(text.split())
                return text
            
            if not query.strip():
                return {'error': 'Empty search query / Пустой поисковый запрос'}
                
            results = {}
            total_mentions = 0
            mentions_by_column = {}
            
            # Preprocess query
            query = preprocess_text(query)
            
            try:
                query_lang = detect(query)
            except:
                query_lang = 'en'
            
            # Lower similarity thresholds
            threshold_ru = 0.1
            threshold_en = 0.15
            
            # Create vectorizers with better parameters
            vectorizer_params = {
                'ngram_range': (1, 3),  # Include up to trigrams
                'min_df': 1,
                'max_df': 0.95,
                'analyzer': 'word'
            }
            
            vectorizer_ru = TfidfVectorizer(stop_words=self.ru_stop_words, **vectorizer_params)
            vectorizer_en = TfidfVectorizer(stop_words=self.en_stop_words, **vectorizer_params)
            
            for column in text_columns:
                if column not in self.df.columns:
                    continue
                
                texts = self.df[column].fillna('').astype(str)
                if texts.empty:
                    continue
                
                # Preprocess all texts
                processed_texts = texts.apply(preprocess_text)
                
                # Process each text based on its language
                ru_texts = []
                en_texts = []
                text_indices = []
                
                for idx, text in enumerate(processed_texts):
                    try:
                        lang = detect(text)
                        if lang == 'ru':
                            ru_texts.append(text)
                            text_indices.append(idx)
                        else:
                            en_texts.append(text)
                            text_indices.append(idx)
                    except:
                        en_texts.append(text)
                        text_indices.append(idx)
                
                similarities = np.zeros(len(texts))
                
                # Process Russian texts
                if ru_texts:
                    try:
                        tfidf_matrix_ru = vectorizer_ru.fit_transform(ru_texts)
                        query_vector_ru = vectorizer_ru.transform([query])
                        ru_similarities = (query_vector_ru * tfidf_matrix_ru.T).toarray()[0]
                        for idx, sim in zip(range(len(ru_texts)), ru_similarities):
                            similarities[text_indices[idx]] = sim
                    except Exception as e:
                        print(f"Error processing Russian texts: {str(e)}")
                
                # Process English texts
                if en_texts:
                    try:
                        tfidf_matrix_en = vectorizer_en.fit_transform(en_texts)
                        query_vector_en = vectorizer_en.transform([query])
                        en_similarities = (query_vector_en * tfidf_matrix_en.T).toarray()[0]
                        for idx, sim in zip(range(len(en_texts)), en_similarities):
                            similarities[text_indices[idx]] = sim
                    except Exception as e:
                        print(f"Error processing English texts: {str(e)}")
                
                # Use appropriate threshold based on query language
                threshold = threshold_ru if query_lang == 'ru' else threshold_en
                matches = [(idx, score) for idx, score in enumerate(similarities) if score > threshold]
                
                if matches:
                    mentions_by_column[column] = {
                        'count': len(matches),
                        'mentions': [
                            {
                                'text': texts.iloc[idx],
                                'similarity': float(score),  # Convert to float for JSON serialization
                                'metadata': {
                                    key: str(self.df.iloc[idx][key])  # Convert all values to strings
                                    for key in ['Title', 'Authors', 'Year']
                                    if key in self.df.columns
                                }
                            }
                            for idx, score in sorted(matches, key=lambda x: x[1], reverse=True)
                        ]
                    }
                    total_mentions += len(matches)
            
            if total_mentions == 0:
                return {
                    'error': 'No matches found / Совпадений не найдено',
                    'query_language': query_lang
                }
            
            return {
                'total_mentions': total_mentions,
                'mentions_by_column': mentions_by_column,
                'query_language': query_lang
            }
            
        except Exception as e:
            return {'error': f'Search error: {str(e)} / Ошибка поиска: {str(e)}'}
