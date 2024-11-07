import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List

class Visualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def plot_topic_distribution(self, filtered_df: pd.DataFrame) -> go.Figure:
        """Create interactive bar chart of topic distribution"""
        topic_counts = filtered_df['Topic'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=topic_counts.index,
                y=topic_counts.values,
                marker_color='rgb(55, 83, 109)'
            )
        ])
        
        fig.update_layout(
            title="Articles by Topic",
            xaxis_title="Topic",
            yaxis_title="Number of Articles",
            hovermode='x'
        )
        return fig
    
    def plot_technology_trends(self, filtered_df: pd.DataFrame) -> go.Figure:
        """Create line chart of technology mentions over time"""
        if 'Year' not in filtered_df.columns:
            return go.Figure()
            
        tech_by_year = filtered_df.groupby(['Year', 'Technologies']).size().unstack(fill_value=0)
        
        fig = go.Figure()
        for tech in tech_by_year.columns:
            fig.add_trace(go.Scatter(
                x=tech_by_year.index,
                y=tech_by_year[tech],
                name=tech,
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title="Technology Mentions Over Time",
            xaxis_title="Year",
            yaxis_title="Mentions",
            hovermode='x unified'
        )
        return fig
    
    def plot_trl_distribution(self, filtered_df: pd.DataFrame) -> go.Figure:
        """Create histogram of TRL distribution"""
        if 'TRL' not in filtered_df.columns:
            return go.Figure()
            
        fig = go.Figure(data=[
            go.Histogram(
                x=filtered_df['TRL'].dropna(),
                nbinsx=9,  # TRL typically ranges from 1-9
                marker_color='rgb(55, 83, 109)'
            )
        ])
        
        fig.update_layout(
            title="TRL Distribution",
            xaxis_title="TRL Level",
            yaxis_title="Number of Articles",
            bargap=0.1
        )
        return fig