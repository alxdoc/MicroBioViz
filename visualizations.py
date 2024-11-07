import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from typing import List, Tuple
import numpy as np

class Visualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    # ... [previous methods remain unchanged until plot_word_cloud] ...
    
    def plot_word_cloud(self, word_freq: dict) -> go.Figure:
        """Create word cloud visualization using plotly"""
        # Validate input
        if not word_freq or not isinstance(word_freq, dict):
            fig = go.Figure()
            fig.add_annotation(
                text="No themes available for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(
                title="Word Cloud of Common Themes",
                showlegend=False,
                xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
                yaxis={'showgrid': False, 'zeroline': False, 'visible': False}
            )
            return fig
        
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
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error generating word cloud: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(
                title="Word Cloud of Common Themes",
                showlegend=False,
                xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
                yaxis={'showgrid': False, 'zeroline': False, 'visible': False}
            )
            return fig
    
    # ... [rest of the methods remain unchanged] ...
