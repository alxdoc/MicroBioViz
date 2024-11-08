import pandas as pd
from typing import List, Dict, Any

def load_data(file) -> pd.DataFrame:
    """Load and validate Excel file"""
    try:
        df = pd.read_excel(file)
        
        # Define simplified column mappings from Russian to English
        column_mapping = {
            'Название статьи': 'Title',
            'О чем статья': 'article description',
            'Аннотация': 'article description',
            'Авторы': 'Authors',
            'Тематика': 'rank',
            'Направление': 'rank',
            'УГТ': 'TRL',
            'Год публикации': 'Year'
        }
        
        # Rename columns if they exist
        for rus_col, eng_col in column_mapping.items():
            if rus_col in df.columns:
                df = df.rename(columns={rus_col: eng_col})
        
        # Define required columns (using English names)
        required_columns = ['Title', 'article description', 'Authors', 'rank']
        
        # Validate required columns
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

def filter_dataframe(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Filter dataframe based on selected filters"""
    filtered_df = df.copy()
    
    for column, filter_value in filters.items():
        if filter_value:
            if isinstance(filter_value, list):
                filtered_df = filtered_df[filtered_df[column].isin(filter_value)]
            elif isinstance(filter_value, tuple) and len(filter_value) == 2:
                filtered_df = filtered_df[
                    (filtered_df[column] >= filter_value[0]) & 
                    (filtered_df[column] <= filter_value[1])
                ]
    
    return filtered_df
