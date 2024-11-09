import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from typing import List, Tuple, Dict, Any
import numpy as np
from plotly.subplots import make_subplots
from scipy import stats
from collections import Counter

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

    def plot_scatter(self, df: pd.DataFrame, x_column: str, y_column: str, 
                    color_column: str = None, size_column: str = None) -> go.Figure:
        try:
            if df.empty or x_column not in df.columns or y_column not in df.columns:
                return self._create_empty_figure(
                    f"No data available for scatter plot / Нет данных для диаграммы рассеяния"
                )
            
            # Create scatter plot
            fig = px.scatter(
                df,
                x=x_column,
                y=y_column,
                color=color_column if color_column in df.columns else None,
                size=size_column if size_column in df.columns else None,
                title=f"{y_column} vs {x_column} / {y_column} против {x_column}",
                labels={
                    x_column: f"{x_column}",
                    y_column: f"{y_column}",
                    'color': f"{color_column}" if color_column else None,
                    'size': f"{size_column}" if size_column else None
                },
                trendline="ols" if df[x_column].dtype.kind in 'biufc' and df[y_column].dtype.kind in 'biufc' else None
            )
            
            # Update layout with new responsive settings
            fig.update_layout(
                height=500,
                margin=dict(l=50, r=50, t=50, b=50),
                hovermode='closest',
                template='plotly_white'
            )
            
            # Enhanced hover template
            hover_template = (
                f"<b>{x_column}:</b> %{{x}}<br>"
                f"<b>{y_column}:</b> %{{y}}<br>"
            )
            if color_column:
                hover_template += f"<b>{color_column}:</b> %{{color}}<br>"
            if size_column:
                hover_template += f"<b>{size_column}:</b> %{{size}}<br>"
            hover_template += "<extra></extra>"
            
            fig.update_traces(
                hovertemplate=hover_template
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(
                f"Error creating scatter plot / Ошибка создания диаграммы рассеяния: {str(e)}"
            )

    def plot_categorical_tree(self, df: pd.DataFrame, category_column: str) -> go.Figure:
        try:
            if df.empty or category_column not in df.columns:
                return self._create_empty_figure(
                    f"No {category_column} data available / Нет данных {category_column}"
                )
            
            # Get category counts and sort by frequency
            category_counts = df[category_column].value_counts()
            if category_counts.empty:
                return self._create_empty_figure(
                    f"No valid {category_column} data / Нет действительных данных {category_column}"
                )
            
            # Create hierarchical data structure
            fig = go.Figure(go.Treemap(
                labels=[f"{cat}<br>({count})" for cat, count in category_counts.items()],
                parents=["" for _ in range(len(category_counts))],
                values=category_counts.values,
                textinfo="label",
                hovertemplate=(
                    "<b>Category:</b> %{label}<br>"
                    "<b>Count:</b> %{value}<br>"
                    "<extra></extra>"
                )
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{category_column} Hierarchical Distribution / Иерархическое распределение {category_column}",
                width=800,
                height=600,
                margin=dict(t=50, l=0, r=0, b=0)
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(
                f"Error creating hierarchical distribution / Ошибка создания иерархического распределения: {str(e)}"
            )

    def plot_topic_distribution(self, df: pd.DataFrame) -> go.Figure:
        try:
            if df.empty or 'rank' not in df.columns:
                return self._create_empty_figure(
                    "No topic data available / Нет данных о темах"
                )

            # Calculate topic distribution
            topic_counts = df['rank'].value_counts()
            
            if topic_counts.empty:
                return self._create_empty_figure(
                    "No valid topic data / Нет действительных данных о темах"
                )

            # Create bar chart
            fig = px.bar(
                x=topic_counts.index,
                y=topic_counts.values,
                labels={
                    'x': 'Topic / Тема',
                    'y': 'Number of Articles / Количество статей'
                },
                title='Article Distribution by Topic / Распределение статей по темам'
            )

            # Update layout
            fig.update_layout(
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
            return self._create_empty_figure(
                f"Error creating topic distribution / Ошибка создания распределения: {str(e)}"
            )

    def plot_trl_distribution(self, df: pd.DataFrame) -> go.Figure:
        try:
            if df.empty or 'TRL' not in df.columns:
                return self._create_empty_figure(
                    "No TRL data available / Нет данных УГТ"
                )
            
            # Remove null values and convert to integers
            trl_data = pd.to_numeric(df['TRL'], errors='coerce').dropna()
            
            if trl_data.empty:
                return self._create_empty_figure(
                    "No valid TRL data / Нет действительных данных УГТ"
                )
            
            # Convert to integers and get value counts
            trl_counts = trl_data.astype(int).value_counts().sort_index()
            
            # Create bar chart
            fig = px.bar(
                x=trl_counts.index,
                y=trl_counts.values,
                labels={
                    'x': 'TRL Level / Уровень УГТ',
                    'y': 'Number of Articles / Количество статей'
                },
                title='TRL Distribution / Распределение УГТ'
            )
            
            # Update layout
            fig.update_layout(
                height=400,
                showlegend=False,
                hovermode='x unified',
                xaxis_tickmode='linear',
                xaxis_tick0=1,
                xaxis_dtick=1
            )
            
            # Add hover template
            fig.update_traces(
                hovertemplate="<b>TRL:</b> %{x}<br><b>Articles:</b> %{y}<extra></extra>"
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(
                f"Error creating TRL distribution / Ошибка создания распределения УГТ: {str(e)}"
            )

    def plot_text_analysis(self, df: pd.DataFrame, column: str) -> go.Figure:
        try:
            if df.empty or column not in df.columns:
                return self._create_empty_figure(
                    f"No {column} data available / Нет данных {column}"
                )
            
            # Combine all text
            text = ' '.join(df[column].dropna().astype(str))
            if not text.strip():
                return self._create_empty_figure(
                    f"No valid {column} data / Нет действительных данных {column}"
                )
            
            # Tokenize and count words
            words = text.lower().split()
            word_freq = Counter(words).most_common(20)
            words, freqs = zip(*word_freq)
            
            # Create bar chart
            fig = px.bar(
                x=words,
                y=freqs,
                title=f"Most Common Words in {column} / Самые частые слова в {column}",
                labels={
                    'x': 'Word / Слово',
                    'y': 'Frequency / Частота'
                }
            )
            
            # Update layout
            fig.update_layout(
                height=400,
                showlegend=False,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(
                f"Error analyzing {column} / Ошибка анализа {column}: {str(e)}"
            )

    def plot_correlation_matrix(self, df: pd.DataFrame, columns: List[str]) -> go.Figure:
        try:
            if df.empty or not columns:
                return self._create_empty_figure(
                    "No data available for correlation / Нет данных для корреляции"
                )
            
            # Calculate correlation matrix
            corr_matrix = df[columns].corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                title="Correlation Matrix / Корреляционная матрица",
                labels={
                    'x': 'Variable / Переменная',
                    'y': 'Variable / Переменная',
                    'color': 'Correlation / Корреляция'
                },
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            
            # Update layout
            fig.update_layout(
                height=600,
                width=800
            )
            
            # Add correlation values as text
            for i in range(len(columns)):
                for j in range(len(columns)):
                    fig.add_annotation(
                        x=i,
                        y=j,
                        text=f"{corr_matrix.iloc[j, i]:.2f}",
                        showarrow=False,
                        font=dict(color='white' if abs(corr_matrix.iloc[j, i]) > 0.5 else 'black')
                    )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(
                f"Error creating correlation matrix / Ошибка создания корреляционной матрицы: {str(e)}"
            )

    def plot_sunburst(self, df: pd.DataFrame, hierarchy_columns: List[str]) -> go.Figure:
        try:
            if df.empty or not all(col in df.columns for col in hierarchy_columns):
                return self._create_empty_figure(
                    "No data available for sunburst / Нет данных для лучевой диаграммы"
                )
            
            # Prepare data for sunburst chart
            data = []
            for _, row in df.iterrows():
                current_path = []
                for col in hierarchy_columns:
                    if pd.notna(row[col]):
                        current_path.append(str(row[col]))
                        # Add each level of the hierarchy
                        data.append({
                            'id': '/'.join(current_path),
                            'parent': '/'.join(current_path[:-1]) if len(current_path) > 1 else '',
                            'labels': current_path[-1],
                            'level': len(current_path)
                        })
            
            # Convert to DataFrame for counting
            hierarchy_df = pd.DataFrame(data)
            value_counts = hierarchy_df.groupby('id').size().reset_index(name='count')
            
            # Create sunburst chart
            fig = go.Figure(go.Sunburst(
                ids=value_counts['id'],
                labels=[id.split('/')[-1] for id in value_counts['id']],
                parents=['/'.join(id.split('/')[:-1]) if '/' in id else '' for id in value_counts['id']],
                values=value_counts['count'],
                branchvalues='total',
                hovertemplate=(
                    '<b>%{label}</b><br>'
                    'Count: %{value}<br>'
                    '<extra></extra>'
                )
            ))
            
            # Update layout
            fig.update_layout(
                title="Hierarchical Data Visualization / Иерархическая визуализация данных",
                width=800,
                height=800,
                margin=dict(t=50, l=0, r=0, b=0)
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(
                f"Error creating sunburst chart / Ошибка создания лучевой диаграммы: {str(e)}"
            )
