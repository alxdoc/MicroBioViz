import streamlit as st
import numpy as np
import pandas as pd
from data_processor import DataProcessor
from visualizations import Visualizer
from text_analyzer import TextAnalyzer
from utils import load_data, filter_dataframe

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
            
            # Sidebar filters
            st.sidebar.title("Filters")
            years = []
            if 'Year' in st.session_state.data.columns:
                valid_years = sorted(st.session_state.data['Year'].dropna().unique())
                if valid_years:
                    years = st.sidebar.multiselect("Select Years / Выберите годы", options=valid_years)
            
            valid_ranks = sorted(st.session_state.data['rank'].dropna().unique())
            ranks = st.sidebar.multiselect("Select Topics / Выберите темы", options=valid_ranks)
            
            # Apply filters
            filtered_df = filter_dataframe(st.session_state.data, years, ranks)
            
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
                    # Create a copy of the dataframe for sorting
                    df_sorted = filtered_df.copy()
                    
                    # Apply sorting based on selection
                    sort_key = sort_options[sort_method]
                    if sort_key == 'frequency_desc':
                        df_sorted[column] = pd.Categorical(
                            df_sorted[column],
                            categories=df_sorted[column].value_counts().index,
                            ordered=True
                        )
                    elif sort_key == 'frequency_asc':
                        df_sorted[column] = pd.Categorical(
                            df_sorted[column],
                            categories=df_sorted[column].value_counts().index[::-1],
                            ordered=True
                        )
                    elif sort_key == 'alpha_asc':
                        df_sorted[column] = pd.Categorical(
                            df_sorted[column],
                            categories=sorted(df_sorted[column].unique()),
                            ordered=True
                        )
                    else:  # alpha_desc
                        df_sorted[column] = pd.Categorical(
                            df_sorted[column],
                            categories=sorted(df_sorted[column].unique(), reverse=True),
                            ordered=True
                        )
                    
                    # Create hierarchical visualization
                    fig_tree = create_visualization(
                        lambda df: st.session_state.visualizer.plot_categorical_tree(df, column),
                        df_sorted,
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
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.data = None
    
    else:
        st.info("Please upload an Excel file to begin analysis / "
                "Пожалуйста, загрузите файл Excel для начала анализа")

if __name__ == "__main__":
    main()
