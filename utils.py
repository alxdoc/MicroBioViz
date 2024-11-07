import pandas as pd
from typing import List

def load_data(file) -> pd.DataFrame:
    """Load and validate Excel file"""
    try:
        df = pd.read_excel(file)
        
        # Define column mappings from Russian to English
        column_mapping = {
            'Название статьи': 'Title',
            'О чем статья': 'Abstract',
            'Соавторство': 'Authors',
            'Ранжирование по тематике': 'Topic',
            'TRL': 'TRL'  # Keep TRL as is
        }
        
        # Rename columns if they exist
        for rus_col, eng_col in column_mapping.items():
            if rus_col in df.columns:
                df = df.rename(columns={rus_col: eng_col})
        
        # Define required columns (using English names)
        required_columns = ['Title', 'Abstract', 'Authors', 'Topic']
        
        # Validate required columns (check both Russian and English names)
        missing_columns = []
        for eng_col in required_columns:
            rus_col = [k for k, v in column_mapping.items() if v == eng_col][0]
            if eng_col not in df.columns and rus_col not in df.columns:
                missing_columns.append(f"{rus_col} ({eng_col})")
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Add TRL column if not present
        if 'TRL' not in df.columns:
            df['TRL'] = None
            
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
