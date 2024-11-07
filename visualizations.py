import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from typing import List, Tuple
import numpy as np

class Visualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def plot_topic_distribution(self, filtered_df: pd.DataFrame) -> go.Figure:
        """Create interactive bar chart of topic distribution"""
        topic_counts = filtered_df['rank'].value_counts()
        
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
        
        temp_df = filtered_df.copy()
        temp_df['Technologies'] = temp_df['Technologies'].fillna('')
        temp_df = temp_df[temp_df['Technologies'] != '']
        temp_df = temp_df.assign(Technologies=temp_df['Technologies'].str.split(';')).explode('Technologies')
        
        tech_by_year = pd.crosstab(temp_df['Year'], temp_df['Technologies'])
        
        fig = go.Figure()
        for tech in tech_by_year.columns:
            if tech:
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
                nbinsx=9,
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
    
    def plot_word_cloud(self, word_freq: dict) -> go.Figure:
        """Create word cloud visualization using plotly"""
        words = list(word_freq.keys())
        frequencies = list(word_freq.values())
        
        # Normalize frequencies for sizing
        max_freq = max(frequencies)
        sizes = [30 * (f / max_freq) + 10 for f in frequencies]
        
        # Create scatter plot with text
        fig = go.Figure()
        
        # Calculate positions in a circular layout
        n_words = len(words)
        angles = np.linspace(0, 2*np.pi, n_words)
        radii = np.random.uniform(0.4, 1.0, n_words)
        
        x_pos = radii * np.cos(angles)
        y_pos = radii * np.sin(angles)
        
        fig.add_trace(go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='text',
            text=words,
            textfont={'size': sizes},
            hoverinfo='text',
            textposition='middle center'
        ))
        
        fig.update_layout(
            title="Word Cloud of Common Themes",
            showlegend=False,
            xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            yaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            hovermode='closest'
        )
        return fig
    
    def plot_tech_network(self, nodes: List[str], edges: List[Tuple[str, str, float]]) -> go.Figure:
        """Create technology co-occurrence network visualization"""
        G = nx.Graph()
        G.add_nodes_from(nodes)
        for source, target, weight in edges:
            G.add_edge(source, target, weight=weight)
        
        # Use spring layout for node positions
        pos = nx.spring_layout(G)
        
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])
        
        # Create edges trace
        edges_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        node_x = []
        node_y = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        
        # Create nodes trace
        nodes_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=20,
                color='lightblue',
                line_width=2))
        
        fig = go.Figure(data=[edges_trace, nodes_trace])
        fig.update_layout(
            title="Technology Co-occurrence Network",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig
    
    def plot_topic_trends(self, topic_trends: pd.DataFrame) -> go.Figure:
        """Create heatmap of topic trends over time"""
        if topic_trends.empty:
            return go.Figure()
            
        fig = px.imshow(
            topic_trends.set_index('Year'),
            aspect='auto',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            title="Topic Trends Over Time",
            xaxis_title="Topic",
            yaxis_title="Year"
        )
        return fig
