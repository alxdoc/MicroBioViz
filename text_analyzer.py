import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from typing import Dict, List, Any

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
            
            results = {}
            total_mentions = 0
            mentions_by_column = {}
            
            try:
                query_lang = detect(query)
            except:
                query_lang = 'en'  # default to English if detection fails
            
            # Create separate vectorizers for each language
            vectorizer_ru = TfidfVectorizer(stop_words=self.ru_stop_words, ngram_range=(1, 2))
            vectorizer_en = TfidfVectorizer(stop_words=self.en_stop_words, ngram_range=(1, 2))
            
            for column in text_columns:
                if column not in self.df.columns:
                    continue
                
                texts = self.df[column].fillna('').astype(str)
                if texts.empty:
                    continue
                
                # Process each text based on its language
                ru_texts = []
                en_texts = []
                text_indices = []  # Keep track of original indices
                
                for idx, text in enumerate(texts):
                    try:
                        lang = detect(text)
                        if lang == 'ru':
                            ru_texts.append(text)
                            text_indices.append(idx)
                        else:  # Default to English for other languages
                            en_texts.append(text)
                            text_indices.append(idx)
                    except:
                        en_texts.append(text)  # Default to English if detection fails
                        text_indices.append(idx)
                
                # Calculate similarities for each language
                similarities = np.zeros(len(texts))
                
                if ru_texts and query_lang == 'ru':
                    tfidf_matrix_ru = vectorizer_ru.fit_transform(ru_texts)
                    query_vector_ru = vectorizer_ru.transform([query])
                    ru_similarities = (query_vector_ru * tfidf_matrix_ru.T).toarray()[0]
                    for idx, sim in zip(range(len(ru_texts)), ru_similarities):
                        similarities[text_indices[idx]] = sim
                
                if en_texts and query_lang != 'ru':
                    tfidf_matrix_en = vectorizer_en.fit_transform(en_texts)
                    query_vector_en = vectorizer_en.transform([query])
                    en_similarities = (query_vector_en * tfidf_matrix_en.T).toarray()[0]
                    for idx, sim in zip(range(len(en_texts)), en_similarities):
                        similarities[text_indices[idx]] = sim
                
                # Adjust threshold based on language
                threshold = 0.2 if query_lang == 'ru' else 0.3
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
                'mentions_by_column': mentions_by_column,
                'query_language': query_lang
            }
            
        except Exception as e:
            return {'error': str(e)}
