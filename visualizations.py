import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from typing import List, Tuple, Dict
import numpy as np

class Visualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Bilingual error messages
        self.error_messages = {
            'no_data': {
                'en': 'No data available',
                'ru': 'Данные отсутствуют'
            },
            'invalid_data': {
                'en': 'Invalid data format',
                'ru': 'Неверный формат данных'
            },
            'empty_data': {
                'en': 'Dataset is empty',
                'ru': 'Набор данных пуст'
            },
            'missing_column': {
                'en': 'Required column missing: {}',
                'ru': 'Отсутствует обязательный столбец: {}'
            }
        }
    
    def _get_bilingual_message(self, key: str, *args) -> str:
        """Get bilingual error message"""
        en_msg = self.error_messages[key]['en'].format(*args)
        ru_msg = self.error_messages[key]['ru'].format(*args)
        return f"{en_msg} / {ru_msg}"

    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a bilingual message"""
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

    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> Dict[str, bool]:
        """Validate dataframe and required columns"""
        validation = {'valid': True, 'message': ''}
        
        if df is None or not isinstance(df, pd.DataFrame):
            validation['valid'] = False
            validation['message'] = self._get_bilingual_message('invalid_data')
            return validation
            
        if df.empty:
            validation['valid'] = False
            validation['message'] = self._get_bilingual_message('empty_data')
            return validation
            
        for col in required_columns:
            if col not in df.columns:
                validation['valid'] = False
                validation['message'] = self._get_bilingual_message('missing_column', col)
                break
                
        return validation

    def plot_trl_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create bar chart showing distribution of TRL values"""
        try:
            # Validate input data
            validation = self._validate_dataframe(df, ['TRL'])
            if not validation['valid']:
                return self._create_empty_figure(validation['message'])

            # Remove null values and invalid TRL values
            valid_trl = df['TRL'].dropna()
            valid_trl = valid_trl[valid_trl.between(1, 9)]
            
            if valid_trl.empty:
                return self._create_empty_figure(
                    "No valid TRL data available (values should be between 1-9) / "
                    "Нет действительных данных УГТ (значения должны быть от 1 до 9)"
                )

            trl_counts = valid_trl.astype(int).value_counts().sort_index()
            
            fig = px.bar(
                x=trl_counts.index,
                y=trl_counts.values,
                labels={'x': 'TRL Level / Уровень УГТ', 'y': 'Number of Articles / Количество статей'},
                title='Distribution of Technology Readiness Levels (TRL) / Распределение уровней технологической готовности (УГТ)'
            )
            
            fig.update_layout(
                xaxis_title="TRL Level / Уровень УГТ",
                yaxis_title="Number of Articles / Количество статей",
                bargap=0.2,
                height=400,
                showlegend=False,
                hovermode='x unified'
            )
            
            fig.update_traces(
                hovertemplate="<b>TRL Level / Уровень УГТ:</b> %{x}<br><b>Articles / Статьи:</b> %{y}<extra></extra>"
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(f"Error in TRL visualization / Ошибка визуализации УГТ: {str(e)}")

    def plot_topic_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create interactive bar chart of topic distribution"""
        try:
            validation = self._validate_dataframe(df, ['rank'])
            if not validation['valid']:
                return self._create_empty_figure(validation['message'])
            
            # Remove empty topics
            valid_topics = df['rank'].dropna().replace('', np.nan).dropna()
            
            if valid_topics.empty:
                return self._create_empty_figure(
                    "No valid topic data available / Нет действительных данных по темам"
                )
            
            topic_counts = valid_topics.value_counts()
            
            fig = px.bar(
                x=topic_counts.index,
                y=topic_counts.values,
                labels={'x': 'Topic / Тема', 'y': 'Number of Articles / Количество статей'},
                title='Article Distribution by Topic / Распределение статей по темам'
            )
            
            fig.update_layout(
                xaxis_title="Topic / Тема",
                yaxis_title="Number of Articles / Количество статей",
                bargap=0.2,
                height=400,
                showlegend=False,
                hovermode='x unified'
            )
            
            fig.update_traces(
                hovertemplate="<b>Topic / Тема:</b> %{x}<br><b>Articles / Статьи:</b> %{y}<extra></extra>"
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(f"Error in topic distribution / Ошибка распределения тем: {str(e)}")

    def plot_word_cloud(self, word_freq: dict) -> go.Figure:
        """Create word cloud visualization using plotly"""
        try:
            if not word_freq or not isinstance(word_freq, dict) or len(word_freq) == 0:
                return self._create_empty_figure(
                    "No themes available for visualization / Нет тем для визуализации"
                )

            words = list(word_freq.keys())
            frequencies = list(word_freq.values())
            
            if not words or not frequencies or max(frequencies) <= 0:
                return self._create_empty_figure(
                    "Invalid word frequencies / Неверные частоты слов"
                )
            
            # Normalize frequencies for sizing
            max_freq = max(frequencies)
            sizes = [30 * (f / max_freq) + 10 for f in frequencies]
            
            # Calculate positions in a circular layout
            n_words = len(words)
            angles = np.linspace(0, 2*np.pi, n_words)
            radii = np.random.uniform(0.4, 1.0, n_words)
            
            x_pos = radii * np.cos(angles)
            y_pos = radii * np.sin(angles)
            
            fig = go.Figure()
            
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
                title="Word Cloud of Common Themes / Облако слов общих тем",
                showlegend=False,
                xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
                yaxis={'showgrid': False, 'zeroline': False, 'visible': False},
                hovermode='closest'
            )
            return fig
            
        except Exception as e:
            return self._create_empty_figure(f"Error generating word cloud / Ошибка генерации облака слов: {str(e)}")

    def plot_technology_trends(self, df: pd.DataFrame) -> go.Figure:
        """Create line chart of technology mentions over time"""
        try:
            validation = self._validate_dataframe(df, ['Year', 'Technologies'])
            if not validation['valid']:
                return self._create_empty_figure(validation['message'])
            
            temp_df = df.copy()
            temp_df['Technologies'] = temp_df['Technologies'].fillna('')
            temp_df = temp_df[temp_df['Technologies'] != '']
            temp_df = temp_df.assign(Technologies=temp_df['Technologies'].str.split(';')).explode('Technologies')
            
            if temp_df.empty:
                return self._create_empty_figure(
                    "No technology mentions found / Упоминания технологий не найдены"
                )
            
            # Remove empty technology strings and year nulls
            temp_df = temp_df[
                (temp_df['Technologies'].str.strip() != '') & 
                (temp_df['Year'].notna())
            ]
            
            if temp_df.empty:
                return self._create_empty_figure(
                    "No valid technology data available / Нет действительных данных по технологиям"
                )
            
            tech_by_year = pd.crosstab(temp_df['Year'], temp_df['Technologies'])
            
            fig = go.Figure()
            for tech in tech_by_year.columns:
                fig.add_trace(go.Scatter(
                    x=tech_by_year.index,
                    y=tech_by_year[tech],
                    name=tech,
                    mode='lines+markers',
                    hovertemplate="<b>Year / Год:</b> %{x}<br><b>Mentions / Упоминания:</b> %{y}<extra></extra>"
                ))
            
            fig.update_layout(
                title="Technology Mentions Over Time / Упоминания технологий с течением времени",
                xaxis_title="Year / Год",
                yaxis_title="Mentions / Упоминания",
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
            return self._create_empty_figure(f"Error in technology trends / Ошибка в трендах технологий: {str(e)}")

    def plot_tech_network(self, nodes: List[str], edges: List[Tuple[str, str, float]]) -> go.Figure:
        """Create technology co-occurrence network visualization"""
        try:
            if not nodes or not edges:
                return self._create_empty_figure(
                    "No technology network data available / Нет данных для сети технологий"
                )
            
            if len(nodes) < 2:
                return self._create_empty_figure(
                    "Insufficient nodes for network visualization / Недостаточно узлов для визуализации сети"
                )
            
            G = nx.Graph()
            G.add_nodes_from(nodes)
            
            # Validate edges
            valid_edges = [(s, t, w) for s, t, w in edges if s in nodes and t in nodes and isinstance(w, (int, float))]
            
            if not valid_edges:
                return self._create_empty_figure(
                    "No valid connections between technologies / Нет действительных связей между технологиями"
                )
            
            for source, target, weight in valid_edges:
                G.add_edge(source, target, weight=weight)
            
            pos = nx.spring_layout(G)
            
            edge_trace = go.Scatter(
                x=[], y=[],
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += (x0, x1, None)
                edge_trace['y'] += (y0, y1, None)
            
            node_trace = go.Scatter(
                x=[pos[node][0] for node in G.nodes()],
                y=[pos[node][1] for node in G.nodes()],
                mode='markers+text',
                text=list(G.nodes()),
                textposition="top center",
                hoverinfo='text',
                marker=dict(
                    size=20,
                    color='lightblue',
                    line_width=2
                )
            )
            
            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                title="Technology Co-occurrence Network / Сеть совместного появления технологий",
                showlegend=False,
                hovermode='closest',
                height=600,
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return fig
            
        except Exception as e:
            return self._create_empty_figure(f"Error in network visualization / Ошибка визуализации сети: {str(e)}")

    def plot_topic_trends(self, topic_trends: pd.DataFrame) -> go.Figure:
        """Create heatmap of topic trends over time"""
        try:
            validation = self._validate_dataframe(topic_trends, ['Year'])
            if not validation['valid']:
                return self._create_empty_figure(validation['message'])
            
            if topic_trends.shape[1] < 3:  # Year column plus at least 2 topics
                return self._create_empty_figure(
                    "Insufficient topic data for trend analysis / Недостаточно данных для анализа трендов"
                )
            
            fig = px.imshow(
                topic_trends.set_index('Year'),
                aspect='auto',
                color_continuous_scale='Viridis',
                labels=dict(
                    x="Topic / Тема",
                    y="Year / Год",
                    color="Proportion / Доля"
                )
            )
            
            fig.update_layout(
                title="Topic Trends Over Time / Тренды тем с течением времени",
                xaxis_title="Topic / Тема",
                yaxis_title="Year / Год",
                height=400,
                coloraxis_colorbar_title="Proportion / Доля"
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(f"Error in topic trends / Ошибка в трендах тем: {str(e)}")
