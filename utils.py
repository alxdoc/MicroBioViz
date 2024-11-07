import pandas as pd
from typing import List

def load_data(file) -> pd.DataFrame:
    """Load and validate Excel file"""
    try:
        df = pd.read_excel(file)
        required_columns = ['Title', 'Abstract', 'Authors', 'Institution', 'Year', 'Topic']
        
        # Validate required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        return df
    
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def filter_dataframe(df: pd.DataFrame, years: List[int], topics: List[str]) -> pd.DataFrame:
    """Filter dataframe based on selected years and topics"""
    filtered_df = df.copy()
    
    if years:
        filtered_df = filtered_df[filtered_df['Year'].isin(years)]
    
    if topics:
        filtered_df = filtered_df[filtered_df['Topic'].isin(topics)]
    
    return filtered_df
