import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
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
    
    def plot_collaboration_network(self, filtered_df: pd.DataFrame) -> go.Figure:
        """Create network visualization of institutional collaborations"""
        G = nx.Graph()
        
        # Build network
        for _, row in filtered_df.iterrows():
            institutions = [inst.strip() for inst in row['Institution'].split(';') if inst.strip()]
            for i in range(len(institutions)):
                if not G.has_node(institutions[i]):
                    G.add_node(institutions[i])
                for j in range(i + 1, len(institutions)):
                    if G.has_edge(institutions[i], institutions[j]):
                        G[institutions[i]][institutions[j]]['weight'] += 1
                    else:
                        G.add_edge(institutions[i], institutions[j], weight=1)
        
        # Calculate layout
        pos = nx.spring_layout(G)
        
        # Create edges
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
        
        # Create nodes
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            hoverinfo='text',
            text=[node for node in G.nodes()],
            marker=dict(
                size=10,
                color='rgb(55, 83, 109)',
                line_width=2))
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           title="Institution Collaboration Network"
                       ))
        return fig
