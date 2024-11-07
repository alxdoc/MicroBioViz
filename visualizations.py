import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from typing import List, Tuple
import numpy as np

class Visualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            showlegend=False,
            xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            yaxis={'showgrid': False, 'zeroline': False, 'visible': False}
        )
        return fig
    
    def plot_trl_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create bar chart showing distribution of TRL values"""
        try:
            if df.empty or 'TRL' not in df.columns:
                return self._create_empty_figure("No TRL data available")
            
            # Remove null values and get value counts
            trl_counts = df['TRL'].dropna().astype(int).value_counts().sort_index()
            
            if trl_counts.empty:
                return self._create_empty_figure("No valid TRL data available")
            
            fig = px.bar(
                x=trl_counts.index,
                y=trl_counts.values,
                labels={'x': 'TRL Level', 'y': 'Number of Articles'},
                title='Distribution of Technology Readiness Levels (TRL)'
            )
            
            # Customize layout
            fig.update_layout(
                xaxis_title="TRL Level",
                yaxis_title="Number of Articles",
                bargap=0.2,
                height=400,
                showlegend=False,
                hovermode='x unified'
            )
            
            # Add hover template
            fig.update_traces(
                hovertemplate="<b>TRL Level:</b> %{x}<br><b>Articles:</b> %{y}<extra></extra>"
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(f"Error creating TRL distribution: {str(e)}")
    
    def plot_topic_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create interactive bar chart of topic distribution"""
        try:
            if df.empty or 'rank' not in df.columns:
                return self._create_empty_figure("No topic data available")
            
            topic_counts = df['rank'].value_counts()
            
            # Handle empty topic counts
            if topic_counts.empty:
                return self._create_empty_figure("No topic data available")
            
            fig = px.bar(
                x=topic_counts.index,
                y=topic_counts.values,
                labels={'x': 'Topic', 'y': 'Number of Articles'},
                title='Article Distribution by Topic'
            )
            
            # Customize layout
            fig.update_layout(
                xaxis_title="Topic",
                yaxis_title="Number of Articles",
                bargap=0.2,
                height=400,
                showlegend=False,
                hovermode='x unified'
            )
            
            # Add hover template
            fig.update_traces(
                hovertemplate="<b>Topic:</b> %{x}<br><b>Articles:</b> %{y}<extra></extra>"
            )
            
            return fig
        except Exception as e:
            return self._create_empty_figure(f"Error creating topic distribution: {str(e)}")
    
    def plot_word_cloud(self, word_freq: dict) -> go.Figure:
        """Create word cloud visualization using plotly"""
        # Validate input
        if not word_freq or not isinstance(word_freq, dict):
            return self._create_empty_figure("No themes available for visualization")
        
        try:
            words = list(word_freq.keys())
            frequencies = list(word_freq.values())
            
            if not words or not frequencies:
                raise ValueError("Empty word frequencies")
            
            # Normalize frequencies for sizing
            max_freq = max(frequencies)
            if max_freq <= 0:
                raise ValueError("Invalid frequency values")
                
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
                hoverinfo='text+name',
                hovertext=[f"{word}: {freq}" for word, freq in zip(words, frequencies)],
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
            
        except Exception as e:
            return self._create_empty_figure(f"Error generating word cloud: {str(e)}")
    
    def plot_technology_trends(self, filtered_df: pd.DataFrame) -> go.Figure:
        """Create line chart of technology mentions over time"""
        try:
            if filtered_df.empty or 'Year' not in filtered_df.columns:
                return self._create_empty_figure("No technology trend data available")
            
            temp_df = filtered_df.copy()
            temp_df['Technologies'] = temp_df['Technologies'].fillna('')
            temp_df = temp_df[temp_df['Technologies'] != '']
            temp_df = temp_df.assign(Technologies=temp_df['Technologies'].str.split(';')).explode('Technologies')
            
            if temp_df.empty:
                return self._create_empty_figure("No technology mentions found")
            
            tech_by_year = pd.crosstab(temp_df['Year'], temp_df['Technologies'])
            
            if tech_by_year.empty:
                return self._create_empty_figure("No technology trends to display")
            
            fig = go.Figure()
            for tech in tech_by_year.columns:
                if tech:
                    fig.add_trace(go.Scatter(
                        x=tech_by_year.index,
                        y=tech_by_year[tech],
                        name=tech,
                        mode='lines+markers',
                        hovertemplate="<b>Year:</b> %{x}<br><b>Mentions:</b> %{y}<extra></extra>"
                    ))
            
            fig.update_layout(
                title="Technology Mentions Over Time",
                xaxis_title="Year",
                yaxis_title="Mentions",
                hovermode='x unified',
                height=400,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05
                )
            )
            return fig
            
        except Exception as e:
            return self._create_empty_figure(f"Error creating technology trends: {str(e)}")
    
    def plot_tech_network(self, nodes: List[str], edges: List[Tuple[str, str, float]]) -> go.Figure:
        """Create technology co-occurrence network visualization"""
        try:
            if not nodes or not edges:
                return self._create_empty_figure("No technology network data available")
            
            G = nx.Graph()
            G.add_nodes_from(nodes)
            for source, target, weight in edges:
                G.add_edge(source, target, weight=weight)
            
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
            
            nodes_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="top center",
                marker=dict(
                    size=20,
                    color='lightblue',
                    line_width=2
                )
            )
            
            fig = go.Figure(data=[edges_trace, nodes_trace])
            fig.update_layout(
                title="Technology Co-occurrence Network",
                showlegend=False,
                hovermode='closest',
                height=600,
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return fig
            
        except Exception as e:
            return self._create_empty_figure(f"Error creating technology network: {str(e)}")
    
    def plot_topic_trends(self, topic_trends: pd.DataFrame) -> go.Figure:
        """Create heatmap of topic trends over time"""
        try:
            if topic_trends.empty or 'Year' not in topic_trends.columns:
                return self._create_empty_figure("No topic trend data available")
            
            fig = px.imshow(
                topic_trends.set_index('Year'),
                aspect='auto',
                color_continuous_scale='Viridis',
                labels=dict(x="Topic", y="Year", color="Proportion")
            )
            
            fig.update_layout(
                title="Topic Trends Over Time",
                xaxis_title="Topic",
                yaxis_title="Year",
                height=400,
                coloraxis_colorbar_title="Proportion"
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(f"Error creating topic trends: {str(e)}")
