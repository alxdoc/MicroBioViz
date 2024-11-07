import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from typing import List, Dict

class TextAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['study', 'research', 'analysis', 'results'])
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for analysis"""
        # Tokenize
        tokens = word_tokenize(text.lower())
        # Remove stopwords and non-alphabetic tokens
        tokens = [token for token in tokens 
                 if token.isalpha() 
                 and token not in self.stop_words 
                 and len(token) > 2]
        return tokens
    
    def get_common_themes(self, filtered_df: pd.DataFrame, top_n: int = 10) -> Dict[str, int]:
        """Extract common themes from abstracts"""
        # Combine all abstracts
        all_text = ' '.join(filtered_df['Abstract'].fillna(''))
        
        # Process text
        tokens = self._preprocess_text(all_text)
        
        # Get most common terms
        word_freq = Counter(tokens)
        return dict(word_freq.most_common(top_n))
    
    def get_keyword_context(self, keyword: str) -> List[str]:
        """Get context around keyword mentions"""
        contexts = []
        for abstract in self.df['Abstract'].dropna():
            if keyword.lower() in abstract.lower():
                # Get the sentence containing the keyword
                sentences = nltk.sent_tokenize(abstract)
                for sentence in sentences:
                    if keyword.lower() in sentence.lower():
                        contexts.append(sentence)
        return contexts
