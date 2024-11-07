import pandas as pd
import numpy as np
from typing import List, Dict

class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess the dataframe for analysis"""
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        
        # Convert year to numeric
        self.df['Year'] = pd.to_numeric(self.df['Year'], errors='coerce')
        
        # Clean text columns
        text_columns = ['Title', 'Abstract', 'Authors', 'Institution', 'Topic']
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('').str.strip()
        
        # Extract technology mentions from abstract
        self.df['Technologies'] = self.df['Abstract'].apply(self._extract_technologies)
    
    def _extract_technologies(self, text: str) -> List[str]:
        """Extract technology mentions from text"""
        # Common technology keywords in microbiology
        tech_keywords = ['PCR', 'sequencing', 'microscopy', 'CRISPR', 'NGS', 
                        'spectroscopy', 'chromatography', 'microarray']
        return [tech for tech in tech_keywords if tech.lower() in text.lower()]
    
    def get_topic_distribution(self) -> Dict[str, int]:
        """Get distribution of articles across topics"""
        return self.df['Topic'].value_counts().to_dict()
    
    def get_yearly_stats(self) -> pd.DataFrame:
        """Get yearly statistics"""
        return self.df.groupby('Year').agg({
            'Title': 'count',
            'Topic': 'nunique',
            'Institution': 'nunique'
        }).reset_index()
    
    def search_articles(self, query: str) -> pd.DataFrame:
        """Search articles by title or keywords"""
        mask = (
            self.df['Title'].str.contains(query, case=False, na=False) |
            self.df['Abstract'].str.contains(query, case=False, na=False) |
            self.df['Topic'].str.contains(query, case=False, na=False)
        )
        return self.df[mask]
    
    def get_institution_collaborations(self) -> List[tuple]:
        """Get institution collaboration pairs"""
        collaborations = []
        for _, row in self.df.iterrows():
            institutions = [inst.strip() for inst in row['Institution'].split(';') if inst.strip()]
            for i in range(len(institutions)):
                for j in range(i + 1, len(institutions)):
                    collaborations.append((institutions[i], institutions[j]))
        return collaborations
