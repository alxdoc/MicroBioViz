import pandas as pd
from typing import List

def load_data(file) -> pd.DataFrame:
    """Load and validate Excel file"""
    try:
        df = pd.read_excel(file)
        
        # Define comprehensive column mappings from Russian to English
        column_mapping = {
            'Название статьи': 'Title',
            'Заголовок': 'Title',
            'О чем статья': 'article description',
            'Аннотация': 'article description',
            'Краткое содержание': 'article description',
            'Соавторство': 'Authors',
            'Авторы': 'Authors',
            'Ранжирование по тематике': 'rank',
            'Тематика': 'rank',
            'Направление': 'rank',
            'TRL': 'TRL',
            'УГТ': 'TRL',
            'Год': 'Year',
            'Год публикации': 'Year'
        }
        
        # Rename columns if they exist
        for rus_col, eng_col in column_mapping.items():
            if rus_col in df.columns:
                df = df.rename(columns={rus_col: eng_col})
        
        # Define required columns (using English names)
        required_columns = ['Title', 'article description', 'Authors', 'rank']
        
        # Validate required columns (check both Russian and English names)
        missing_columns = []
        for eng_col in required_columns:
            rus_cols = [k for k, v in column_mapping.items() if v == eng_col]
            if eng_col not in df.columns and not any(col in df.columns for col in rus_cols):
                missing_columns.append(f"{', '.join(rus_cols)} ({eng_col})")
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Add TRL column if not present
        if 'TRL' not in df.columns:
            df['TRL'] = None
            
        # Add Year column if not present
        if 'Year' not in df.columns:
            df['Year'] = None
            
        # Data validation and cleaning
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            
        if 'TRL' in df.columns:
            df['TRL'] = pd.to_numeric(df['TRL'], errors='coerce')
            # Validate TRL range
            valid_mask = df['TRL'].between(1, 9, inclusive='both') | df['TRL'].isna()
            if not valid_mask.all():
                invalid_rows = (~valid_mask).sum()
                print(f"Warning: {invalid_rows} rows have TRL values outside the valid range (1-9)")
                df.loc[~valid_mask, 'TRL'] = None
        
        # Clean text columns
        text_columns = ['Title', 'article description', 'Authors', 'rank']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str).str.strip()
        
        return df
    
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def filter_dataframe(df: pd.DataFrame, years: List[int], topics: List[str]) -> pd.DataFrame:
    """Filter dataframe based on selected years and topics"""
    filtered_df = df.copy()
    
    if years:
        filtered_df = filtered_df[filtered_df['Year'].isin(years)]
    
    if topics:
        filtered_df = filtered_df[filtered_df['rank'].isin(topics)]
    
    return filtered_df