import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from typing import List, Dict

class TextAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Download required NLTK data with proper error handling
        self._initialize_nltk()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['study', 'research', 'analysis', 'results'])
    
    def _initialize_nltk(self):
        """Initialize NLTK resources with error handling"""
        required_packages = ['punkt', 'stopwords']
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
            except LookupError:
                try:
                    print(f"Downloading NLTK package: {package}")
                    nltk.download(package, quiet=True)
                except Exception as e:
                    print(f"Error downloading NLTK package {package}: {str(e)}")
                    raise Exception(f"Failed to initialize NLTK package: {package}")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for analysis"""
        if not isinstance(text, str):
            return []
        try:
            # Tokenize
            tokens = word_tokenize(text.lower())
            # Remove stopwords and non-alphabetic tokens
            tokens = [token for token in tokens 
                     if token.isalpha() 
                     and token not in self.stop_words 
                     and len(token) > 2]
            return tokens
        except Exception as e:
            print(f"Error preprocessing text: {str(e)}")
            return []
    
    def get_common_themes(self, filtered_df: pd.DataFrame, top_n: int = 10) -> Dict[str, int]:
        """Extract common themes from article descriptions"""
        # Combine all article descriptions
        all_text = ' '.join(filtered_df['article description'].fillna(''))
        
        # Process text
        tokens = self._preprocess_text(all_text)
        
        # Get most common terms
        word_freq = Counter(tokens)
        return dict(word_freq.most_common(top_n))
    
    def get_keyword_context(self, keyword: str) -> List[str]:
        """Get context around keyword mentions"""
        contexts = []
        for desc in self.df['article description'].dropna():
            if keyword.lower() in desc.lower():
                try:
                    # Get the sentence containing the keyword
                    sentences = nltk.sent_tokenize(desc)
                    for sentence in sentences:
                        if keyword.lower() in sentence.lower():
                            contexts.append(sentence)
                except Exception as e:
                    print(f"Error processing keyword context: {str(e)}")
        return contexts
