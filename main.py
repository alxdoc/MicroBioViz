import streamlit as st
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

def main():
    st.title("Microbiology Articles Dashboard")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Excel file containing article data", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        if st.session_state.data is None:
            # Load and process data
            df = load_data(uploaded_file)
            st.session_state.data = df
            st.session_state.processor = DataProcessor(df)
            st.session_state.visualizer = Visualizer(df)
            st.session_state.text_analyzer = TextAnalyzer(df)
    
        # Sidebar filters
        st.sidebar.title("Filters")
        years = st.sidebar.multiselect("Select Years", options=sorted(st.session_state.data['Year'].unique()))
        topics = st.sidebar.multiselect("Select Topics", options=sorted(st.session_state.data['Topic'].unique()))
        
        # Apply filters
        filtered_df = filter_dataframe(st.session_state.data, years, topics)
        
        # Main content
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Article Distribution by Topic")
            fig_topics = st.session_state.visualizer.plot_topic_distribution(filtered_df)
            st.plotly_chart(fig_topics, use_container_width=True)
            
            st.subheader("Technology Mentions Over Time")
            fig_tech = st.session_state.visualizer.plot_technology_trends(filtered_df)
            st.plotly_chart(fig_tech, use_container_width=True)
        
        with col2:
            st.subheader("Institutional Collaboration Network")
            fig_network = st.session_state.visualizer.plot_collaboration_network(filtered_df)
            st.plotly_chart(fig_network, use_container_width=True)
            
            st.subheader("Common Themes")
            themes = st.session_state.text_analyzer.get_common_themes(filtered_df)
            st.write(themes)
        
        # Article search and details
        st.subheader("Search Articles")
        search_term = st.text_input("Search by title or keywords")
        if search_term:
            search_results = st.session_state.processor.search_articles(search_term)
            for _, article in search_results.iterrows():
                with st.expander(f"{article['Title']} ({article['Year']})"):
                    st.write(f"**Authors:** {article['Authors']}")
                    st.write(f"**Institution:** {article['Institution']}")
                    st.write(f"**Topic:** {article['Topic']}")
                    st.write(f"**Abstract:** {article['Abstract']}")
    
    else:
        st.info("Please upload an Excel file to begin analysis")

if __name__ == "__main__":
    main()
