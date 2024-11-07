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
        try:
            if st.session_state.data is None:
                # Load and process data
                df = load_data(uploaded_file)
                st.session_state.data = df
                st.session_state.processor = DataProcessor(df)
                st.session_state.visualizer = Visualizer(df)
                st.session_state.text_analyzer = TextAnalyzer(df)
        
            # Sidebar filters
            st.sidebar.title("Filters")
            if 'Year' in st.session_state.data.columns:
                years = st.sidebar.multiselect("Select Years", options=sorted(st.session_state.data['Year'].unique()))
            else:
                years = []
            ranks = st.sidebar.multiselect("Select Topics", options=sorted(st.session_state.data['rank'].unique()))
            
            # Apply filters
            filtered_df = filter_dataframe(st.session_state.data, years, ranks)
            
            # Main content
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Article Distribution by Topic")
                fig_topics = st.session_state.visualizer.plot_topic_distribution(filtered_df)
                st.plotly_chart(fig_topics, use_container_width=True)
                
                if 'TRL' in filtered_df.columns:
                    st.subheader("TRL Distribution")
                    fig_trl = st.session_state.visualizer.plot_trl_distribution(filtered_df)
                    st.plotly_chart(fig_trl, use_container_width=True)
            
            with col2:
                st.subheader("Technology Mentions Over Time")
                fig_tech = st.session_state.visualizer.plot_technology_trends(filtered_df)
                st.plotly_chart(fig_tech, use_container_width=True)
                
                st.subheader("Common Themes")
                themes = st.session_state.text_analyzer.get_common_themes(filtered_df)
                st.write(themes)
            
            # Article search and details
            st.subheader("Search Articles")
            search_term = st.text_input("Search by title or keywords")
            if search_term:
                search_results = st.session_state.processor.search_articles(search_term)
                for _, article in search_results.iterrows():
                    with st.expander(f"{article['Title']}"):
                        # Display all available columns in an organized way
                        col_left, col_right = st.columns(2)
                        
                        with col_left:
                            st.markdown("#### Basic Information")
                            st.write(f"**Title:** {article['Title']}")
                            st.write(f"**Authors:** {article['Authors']}")
                            st.write(f"**Topic:** {article['rank']}")
                            if 'Year' in article and pd.notna(article['Year']):
                                st.write(f"**Year:** {int(article['Year'])}")
                            if 'TRL' in article and pd.notna(article['TRL']):
                                st.write(f"**TRL:** {int(article['TRL'])}")
                        
                        with col_right:
                            st.markdown("#### Technical Details")
                            st.write("**Description:**")
                            st.write(article['article description'])
                            
                            if 'Technologies' in article and article['Technologies']:
                                st.write("**Technologies Mentioned:**")
                                techs = article['Technologies'].split(';')
                                for tech in techs:
                                    if tech:
                                        st.write(f"- {tech}")
                        
                        # Display any additional columns that might be present
                        additional_cols = [col for col in article.index if col not in [
                            'Title', 'Authors', 'rank', 'Year', 'TRL', 
                            'article description', 'Technologies'
                        ]]
                        
                        if additional_cols:
                            st.markdown("#### Additional Information")
                            for col in additional_cols:
                                if pd.notna(article[col]):
                                    st.write(f"**{col}:** {article[col]}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.data = None
    
    else:
        st.info("Please upload an Excel file to begin analysis")

if __name__ == "__main__":
    main()
