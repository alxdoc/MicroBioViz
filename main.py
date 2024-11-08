import streamlit as st
import numpy as np
import pandas as pd
from data_processor import DataProcessor
from visualizations import Visualizer
from text_analyzer import TextAnalyzer
from utils import load_data, filter_dataframe
from typing import Dict, Any

st.set_page_config(page_title="Microbiology Articles Dashboard", layout="wide")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processor' not in st.session_state:
    st.session_state.processor = None

def create_visualization(func, data, error_message):
    """Wrapper function for creating visualizations with error handling"""
    try:
        return func(data)
    except Exception as e:
        st.error(f"{error_message}: {str(e)}")
        return None

def main():
    st.title("Microbiology Articles Dashboard")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Excel file containing article data", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            if st.session_state.data is None:
                # Load and process data
                df = load_data(uploaded_file)
                if df.empty:
                    st.warning("Uploaded file contains no data / Загруженный файл не содержит данных")
                    return
                
                st.session_state.data = df
                st.session_state.processor = DataProcessor(df)
                st.session_state.visualizer = Visualizer(df)
                st.session_state.text_analyzer = TextAnalyzer(df)
            
            # Initialize filters dictionary and filtered dataframe
            filters: Dict[str, Any] = {}
            filtered_df = st.session_state.data.copy()
            
            # Sidebar filters
            st.sidebar.title("Filters / Фильтры")
            
            # Year filter
            years = []
            if 'Year' in st.session_state.data.columns:
                valid_years = sorted(st.session_state.data['Year'].dropna().unique())
                if valid_years:
                    years = st.sidebar.multiselect("Select Years / Выберите годы", options=valid_years)
                    if years:
                        filtered_df = filtered_df[filtered_df['Year'].isin(years)]
            
            # Topic filter
            valid_ranks = sorted(st.session_state.data['rank'].dropna().unique())
            ranks = st.sidebar.multiselect("Select Topics / Выберите темы", options=valid_ranks)
            if ranks:
                filtered_df = filtered_df[filtered_df['rank'].isin(ranks)]

            # Add numeric range filters
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['Year']:  # Skip already handled columns
                    # Convert to numeric and handle NaN values
                    valid_values = pd.to_numeric(st.session_state.data[col], errors='coerce').dropna()
                    if not valid_values.empty:
                        min_val = float(valid_values.min())
                        max_val = float(valid_values.max())
                        if min_val != max_val and not (np.isnan(min_val) or np.isnan(max_val)):
                            st.sidebar.write(f"{col} Range / Диапазон")
                            range_vals = st.sidebar.slider(
                                f"Select range for {col}",
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val, max_val),
                                key=f'{col}_range'
                            )
                            filtered_df = filtered_df[
                                (filtered_df[col] >= range_vals[0]) & 
                                (filtered_df[col] <= range_vals[1])
                            ]

            # Add categorical filters
            categorical_cols = st.session_state.data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col not in ['rank']:  # Skip already handled columns
                    unique_vals = sorted(st.session_state.data[col].dropna().unique())
                    if len(unique_vals) > 0 and len(unique_vals) <= 50:  # Only show if reasonable number of options
                        selected_vals = st.sidebar.multiselect(
                            f"Select {col} / Выберите {col}",
                            options=unique_vals,
                            key=f'{col}_filter'
                        )
                        if selected_vals:
                            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
            
            if filtered_df.empty:
                st.warning("No data matches the selected filters / Нет данных, соответствующих выбранным фильтрам")
                return
            
            # Main content layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Article Distribution by Topic / Распределение статей по темам")
                fig_topics = create_visualization(
                    st.session_state.visualizer.plot_topic_distribution,
                    filtered_df,
                    "Error creating topic distribution / Ошибка создания распределения тем"
                )
                if fig_topics:
                    st.plotly_chart(fig_topics, use_container_width=True)
            
            with col2:
                if 'TRL' in filtered_df.columns:
                    st.subheader("TRL Distribution / Распределение УГТ")
                    fig_trl = create_visualization(
                        st.session_state.visualizer.plot_trl_distribution,
                        filtered_df,
                        "Error creating TRL distribution / Ошибка создания распределения УГТ"
                    )
                    if fig_trl:
                        st.plotly_chart(fig_trl, use_container_width=True)
            
            # Advanced Analysis Section
            st.header("Advanced Analysis / Расширенный анализ")
            
            # Categorical Data Analysis with Hierarchical Visualization
            st.subheader("Categorical Data Analysis / Анализ категориальных данных")
            categorical_columns = ['Method', 'Active Agent', 'Environmental Safety', 'Economic Efficiency', 'rank']
            
            # Add sorting options
            sort_options = {
                'Frequency (High to Low)': 'frequency_desc',
                'Frequency (Low to High)': 'frequency_asc',
                'Alphabetical (A-Z)': 'alpha_asc',
                'Alphabetical (Z-A)': 'alpha_desc'
            }
            
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_categorical = st.multiselect(
                    "Select categorical variables to analyze / Выберите категориальные переменные для анализа",
                    [col for col in categorical_columns if col in filtered_df.columns]
                )
            with col2:
                sort_method = st.selectbox(
                    "Sort by / Сортировать по",
                    options=list(sort_options.keys()),
                    index=0
                )

            if selected_categorical:
                for column in selected_categorical:
                    # Create hierarchical visualization
                    fig_tree = create_visualization(
                        lambda df: st.session_state.visualizer.plot_categorical_tree(df, column),
                        filtered_df,
                        f"Error analyzing {column} / Ошибка анализа {column}"
                    )
                    if fig_tree:
                        st.plotly_chart(fig_tree, use_container_width=True)
            
            # Text Analysis
            st.subheader("Text Analysis / Анализ текста")
            text_columns = ['Conclusions', 'Results', 'article description']
            selected_text = st.multiselect(
                "Select text fields to analyze / Выберите текстовые поля для анализа",
                [col for col in text_columns if col in filtered_df.columns]
            )
            
            if selected_text:
                for column in selected_text:
                    fig_text = create_visualization(
                        lambda df: st.session_state.visualizer.plot_text_analysis(df, column),
                        filtered_df,
                        f"Error analyzing {column} / Ошибка анализа {column}"
                    )
                    if fig_text:
                        st.plotly_chart(fig_text, use_container_width=True)
            
            # Correlation Analysis
            st.subheader("Correlation Analysis / Корреляционный анализ")
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                fig_corr = create_visualization(
                    lambda df: st.session_state.visualizer.plot_correlation_matrix(df, numeric_cols),
                    filtered_df,
                    "Error creating correlation matrix / Ошибка создания корреляционной матрицы"
                )
                if fig_corr:
                    st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Insufficient numerical variables for correlation analysis / "
                       "Недостаточно числовых переменных для корреляционного анализа")

            # Scatter Plot Analysis
            st.subheader("Scatter Plot Analysis / Анализ диаграммы рассеяния")
            
            scatter_container = st.container()
            with scatter_container:
                # Get numerical and categorical columns
                numeric_columns = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_columns = filtered_df.select_dtypes(include=['object']).columns.tolist()
                all_columns = numeric_columns + categorical_columns
                
                # Column selection
                col1, col2 = st.columns(2)
                with col1:
                    x_column = st.selectbox(
                        "Select X-axis variable / Выберите переменную оси X",
                        options=all_columns,
                        key="scatter_x"
                    )
                
                with col2:
                    y_column = st.selectbox(
                        "Select Y-axis variable / Выберите переменную оси Y",
                        options=all_columns,
                        key="scatter_y"
                    )
                
                # Optional parameters
                col1, col2 = st.columns(2)
                with col1:
                    color_column = st.selectbox(
                        "Color by (optional) / Цвет по (необязательно)",
                        options=['None'] + categorical_columns,
                        key="scatter_color"
                    )
                
                with col2:
                    size_column = st.selectbox(
                        "Size by (optional) / Размер по (необязательно)",
                        options=['None'] + numeric_columns,
                        key="scatter_size"
                    )
                
                # Create scatter plot
                if x_column and y_column:
                    fig_scatter = create_visualization(
                        lambda df: st.session_state.visualizer.plot_scatter(
                            df, 
                            x_column, 
                            y_column,
                            None if color_column == 'None' else color_column,
                            None if size_column == 'None' else size_column
                        ),
                        filtered_df,
                        "Error creating scatter plot / Ошибка создания диаграммы рассеяния"
                    )
                    if fig_scatter:
                        st.plotly_chart(fig_scatter, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.data = None
    
    else:
        st.info("Please upload an Excel file to begin analysis / "
                "Пожалуйста, загрузите файл Excel для начала анализа")

if __name__ == "__main__":
    main()
