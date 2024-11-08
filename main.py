import streamlit as st
import pandas as pd
import numpy as np
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
                
                if 'TRL' in filtered_df.columns:
                    st.subheader("TRL Distribution / Распределение УГТ")
                    fig_trl = create_visualization(
                        st.session_state.visualizer.plot_trl_distribution,
                        filtered_df,
                        "Error creating TRL distribution / Ошибка создания распределения УГТ"
                    )
                    if fig_trl:
                        st.plotly_chart(fig_trl, use_container_width=True)
                    
                    # Add TRL-Year correlation analysis
                    try:
                        trl_correlation = st.session_state.text_analyzer.get_trl_year_correlation(filtered_df)
                        st.subheader("TRL Analysis / Анализ УГТ")
                        
                        correlation_value = trl_correlation.get('correlation', 0.0)
                        if correlation_value != 0.0:
                            st.write(f"Correlation between TRL and Publication Year / "
                                   f"Корреляция между УГТ и годом публикации: {correlation_value:.2f}")
                        
                        if trl_correlation.get('has_sufficient_data', False):
                            avg_trl_by_year = trl_correlation.get('avg_trl_by_year', {})
                            if avg_trl_by_year:
                                st.write("Average TRL by Year / Средний УГТ по годам:")
                                for year, avg_trl in sorted(avg_trl_by_year.items()):
                                    st.write(f"- {int(year)}: {avg_trl:.1f}")
                        else:
                            st.info("Insufficient data for TRL analysis / Недостаточно данных для анализа УГТ")
                    except Exception as e:
                        st.error(f"Error in TRL analysis / Ошибка в анализе УГТ: {str(e)}")
            
            with col2:
                st.subheader("Technology Mentions Over Time / Упоминания технологий с течением времени")
                fig_tech = create_visualization(
                    st.session_state.visualizer.plot_technology_trends,
                    filtered_df,
                    "Error creating technology trends / Ошибка создания трендов технологий"
                )
                if fig_tech:
                    st.plotly_chart(fig_tech, use_container_width=True)
                
                # Add word cloud visualization
                st.subheader("Theme Word Cloud / Облако слов тем")
                try:
                    themes = st.session_state.text_analyzer.get_common_themes(filtered_df)
                    if themes:
                        fig_cloud = create_visualization(
                            st.session_state.visualizer.plot_word_cloud,
                            themes,
                            "Error creating word cloud / Ошибка создания облака слов"
                        )
                        if fig_cloud:
                            st.plotly_chart(fig_cloud, use_container_width=True)
                    else:
                        st.info("No themes available for visualization / Нет тем для визуализации")
                except Exception as e:
                    st.error(f"Error processing themes / Ошибка обработки тем: {str(e)}")
            
            # Detailed Analysis Section
            st.header("Detailed Analysis / Детальный анализ")
            
            # Numerical Data Analysis
            st.subheader("Numerical Data Analysis / Анализ числовых данных")
            numerical_columns = ['Temperature', 'Duration', 'Pressure', 'TRL']
            selected_numerical = st.multiselect(
                "Select numerical variables to analyze / Выберите числовые переменные для анализа",
                [col for col in numerical_columns if col in filtered_df.columns]
            )
            
            if selected_numerical:
                for column in selected_numerical:
                    fig_num = create_visualization(
                        lambda df: st.session_state.visualizer.plot_numerical_distribution(df, column),
                        filtered_df,
                        f"Error analyzing {column} / Ошибка анализа {column}"
                    )
                    if fig_num:
                        st.plotly_chart(fig_num, use_container_width=True)
            
            # Categorical Data Analysis
            st.subheader("Categorical Data Analysis / Анализ категориальных данных")
            categorical_columns = ['Method', 'Active Agent', 'Environmental Safety', 'Economic Efficiency']
            selected_categorical = st.multiselect(
                "Select categorical variables to analyze / Выберите категориальные переменные для анализа",
                [col for col in categorical_columns if col in filtered_df.columns]
            )
            
            if selected_categorical:
                for column in selected_categorical:
                    fig_cat = create_visualization(
                        lambda df: st.session_state.visualizer.plot_categorical_distribution(df, column),
                        filtered_df,
                        f"Error analyzing {column} / Ошибка анализа {column}"
                    )
                    if fig_cat:
                        st.plotly_chart(fig_cat, use_container_width=True)
            
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
            if st.checkbox("Show correlation matrix / Показать корреляционную матрицу"):
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
            
            # Relationship Analysis
            st.subheader("Relationship Analysis / Анализ зависимостей")
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox(
                    "Select X variable / Выберите переменную X",
                    filtered_df.select_dtypes(include=[np.number]).columns
                )
            with col2:
                y_var = st.selectbox(
                    "Select Y variable / Выберите переменную Y",
                    filtered_df.select_dtypes(include=[np.number]).columns
                )
            
            if x_var and y_var:
                fig_rel = create_visualization(
                    lambda df: st.session_state.visualizer.plot_relationship(df, x_var, y_var),
                    filtered_df,
                    "Error creating relationship plot / Ошибка создания графика зависимости"
                )
                if fig_rel:
                    st.plotly_chart(fig_rel, use_container_width=True)
            
            # Advanced Analysis Section
            st.header("Advanced Analysis / Расширенный анализ")
            
            # Technology Network Analysis
            st.subheader("Technology Co-occurrence Network / Сеть совместного появления технологий")
            try:
                nodes, edges = st.session_state.text_analyzer.get_technology_network(filtered_df)
                if nodes and edges:
                    fig_network = create_visualization(
                        lambda df: st.session_state.visualizer.plot_tech_network(nodes, edges),
                        filtered_df,
                        "Error creating network visualization / Ошибка создания визуализации сети"
                    )
                    if fig_network:
                        st.plotly_chart(fig_network, use_container_width=True)
                else:
                    st.info("Not enough technology co-occurrence data / Недостаточно данных о совместном появлении технологий")
            except Exception as e:
                st.error(f"Error in network analysis / Ошибка в анализе сети: {str(e)}")
            
            # Topic Trends Analysis
            st.subheader("Topic Trends Over Time / Тренды тем с течением времени")
            try:
                topic_trends = st.session_state.text_analyzer.analyze_topic_trends(filtered_df)
                if not topic_trends.empty:
                    fig_trends = create_visualization(
                        st.session_state.visualizer.plot_topic_trends,
                        topic_trends,
                        "Error creating topic trends / Ошибка создания трендов тем"
                    )
                    if fig_trends:
                        st.plotly_chart(fig_trends, use_container_width=True)
                else:
                    st.info("Not enough temporal data to analyze topic trends / "
                           "Недостаточно временных данных для анализа трендов тем")
            except Exception as e:
                st.error(f"Error in topic trends analysis / Ошибка в анализе трендов тем: {str(e)}")
            
            # Article Search and Details
            st.header("Article Search / Поиск статей")
            search_term = st.text_input("Search by title or keywords / Поиск по названию или ключевым словам")
            if search_term:
                try:
                    search_results = st.session_state.processor.search_articles(search_term)
                    if search_results.empty:
                        st.info("No articles found matching the search criteria / "
                               "Не найдено статей, соответствующих критериям поиска")
                    else:
                        for _, article in search_results.iterrows():
                            with st.expander(f"{article['Title']}"):
                                col_left, col_right = st.columns(2)
                                
                                with col_left:
                                    st.markdown("#### Basic Information / Основная информация")
                                    st.write(f"**Title / Название:** {article['Title']}")
                                    st.write(f"**Authors / Авторы:** {article['Authors']}")
                                    st.write(f"**Topic / Тема:** {article['rank']}")
                                    if 'Year' in article and pd.notna(article['Year']):
                                        st.write(f"**Year / Год:** {int(article['Year'])}")
                                    if 'TRL' in article and pd.notna(article['TRL']):
                                        st.write(f"**TRL / УГТ:** {int(article['TRL'])}")
                                
                                with col_right:
                                    st.markdown("#### Technical Details / Технические детали")
                                    st.write("**Description / Описание:**")
                                    st.write(article['article description'])
                                    
                                    if 'Technologies' in article and article['Technologies']:
                                        st.write("**Technologies Mentioned / Упомянутые технологии:**")
                                        techs = article['Technologies'].split(';')
                                        for tech in techs:
                                            if tech:
                                                st.write(f"- {tech}")
                except Exception as e:
                    st.error(f"Error in article search / Ошибка в поиске статей: {str(e)}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.data = None
    
    else:
        st.info("Please upload an Excel file to begin analysis / "
                "Пожалуйста, загрузите файл Excel для начала анализа")

if __name__ == "__main__":
    main()
