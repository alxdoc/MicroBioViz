import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from typing import List, Tuple, Dict, Any
import numpy as np
from plotly.subplots import make_subplots
from scipy import stats

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

    def plot_technology_trends(self, df: pd.DataFrame) -> go.Figure:
        try:
            if df.empty or 'Technologies' not in df.columns or 'Year' not in df.columns:
                return self._create_empty_figure(
                    "No technology trend data available / Нет данных о трендах технологий"
                )
            
            # Get valid data
            valid_data = df[['Technologies', 'Year']].dropna()
            if valid_data.empty:
                return self._create_empty_figure(
                    "No valid technology trend data / Нет действительных данных о трендах"
                )
            
            # Process technology mentions by year
            tech_trends = {}
            years = sorted(valid_data['Year'].unique())
            
            for _, row in valid_data.iterrows():
                year = int(row['Year'])
                techs = row['Technologies'].split(';')
                if year not in tech_trends:
                    tech_trends[year] = {}
                for tech in techs:
                    if tech:  # Skip empty strings
                        tech_trends[year][tech] = tech_trends[year].get(tech, 0) + 1
            
            if not tech_trends:
                return self._create_empty_figure(
                    "No technology mentions found / Упоминания технологий не найдены"
                )
            
            # Create figure
            fig = go.Figure()
            
            # Add trace for each technology
            all_techs = set()
            for year_data in tech_trends.values():
                all_techs.update(year_data.keys())
            
            for tech in sorted(all_techs):
                yearly_counts = [tech_trends.get(year, {}).get(tech, 0) for year in years]
                fig.add_trace(
                    go.Scatter(
                        x=years,
                        y=yearly_counts,
                        name=tech,
                        mode='lines+markers',
                        hovertemplate=(
                            "<b>Year:</b> %{x}<br>"
                            "<b>Mentions:</b> %{y}<br>"
                            "<b>Technology:</b> " + tech +
                            "<extra></extra>"
                        )
                    )
                )
            
            # Update layout
            fig.update_layout(
                title="Technology Mentions Over Time / Упоминания технологий с течением времени",
                xaxis_title="Year / Год",
                yaxis_title="Number of Mentions / Количество упоминаний",
                hovermode='x unified',
                height=400,
                showlegend=True,
                legend_title="Technologies / Технологии"
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(
                f"Error creating technology trends / Ошибка создания трендов: {str(e)}"
            )
            
    def plot_word_cloud(self, word_frequencies: Dict[str, int]) -> go.Figure:
        try:
            if not word_frequencies:
                return self._create_empty_figure(
                    "No word frequency data available / Нет данных о частоте слов"
                )
            
            words = list(word_frequencies.keys())
            frequencies = list(word_frequencies.values())
            
            # Calculate positions in a circular layout
            n_words = len(words)
            angles = np.linspace(0, 2*np.pi, n_words)
            radii = np.sqrt(frequencies) / max(np.sqrt(frequencies)) * 0.8
            
            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
            
            # Create scatter plot with text
            fig = go.Figure()
            
            # Add words as text with size based on frequency
            max_freq = max(frequencies)
            for i, (word, freq) in enumerate(zip(words, frequencies)):
                size = 20 * (freq / max_freq) + 10  # Scale font size
                fig.add_trace(go.Scatter(
                    x=[x[i]],
                    y=[y[i]],
                    mode='text',
                    text=[word],
                    textfont=dict(
                        size=size,
                        color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]
                    ),
                    hoverinfo='text',
                    hovertext=f'{word}: {freq}'
                ))
            
            # Update layout
            fig.update_layout(
                title="Theme Word Cloud / Облако слов тем",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                hovermode='closest',
                width=800,
                height=600
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(
                f"Error creating word cloud / Ошибка создания облака слов: {str(e)}"
            )

    def plot_relationship(self, df: pd.DataFrame, x_var: str, y_var: str) -> go.Figure:
        try:
            if df.empty or x_var not in df.columns or y_var not in df.columns:
                return self._create_empty_figure(
                    "Missing required data / Отсутствуют необходимые данные"
                )
            
            # Get valid data pairs
            valid_data = df[[x_var, y_var]].dropna()
            if len(valid_data) < 2:
                return self._create_empty_figure(
                    "Insufficient data points / Недостаточно точек данных"
                )
            
            # Calculate correlation
            correlation, p_value = stats.pearsonr(valid_data[x_var], valid_data[y_var])
            
            # Create scatter plot with trend line
            fig = px.scatter(
                valid_data,
                x=x_var,
                y=y_var,
                trendline="ols",
                title=f"Relationship: {x_var} vs {y_var} / Зависимость: {x_var} против {y_var}"
            )
            
            # Add correlation information
            correlation_text = (
                f"Correlation / Корреляция: {correlation:.2f}<br>"
                f"P-value / P-значение: {p_value:.4f}"
            )
            
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98,
                text=correlation_text,
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
            
            # Update layout
            fig.update_layout(
                height=500,
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(
                f"Error creating relationship plot / Ошибка создания графика зависимости: {str(e)}"
            )

    def plot_data_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze and visualize relationships between variables in the dataset.
        Returns a dictionary containing figures and statistical information.
        """
        try:
            if df.empty:
                return {
                    'error': self._get_bilingual_message('empty_data'),
                    'figures': [],
                    'insights': []
                }

            results = {
                'figures': [],
                'insights': [],
                'error': None
            }

            # Analyze numerical relationships
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) >= 2:
                # Calculate correlation matrix
                corr_matrix = df[numerical_cols].corr()
                
                # Find significant correlations
                for i in range(len(numerical_cols)):
                    for j in range(i + 1, len(numerical_cols)):
                        col1, col2 = numerical_cols[i], numerical_cols[j]
                        corr = corr_matrix.iloc[i, j]
                        
                        if abs(corr) >= 0.3:  # Consider correlations >= 0.3
                            # Calculate statistical significance
                            valid_data = df[[col1, col2]].dropna()
                            if len(valid_data) >= 2:
                                correlation, p_value = stats.pearsonr(
                                    valid_data[col1], 
                                    valid_data[col2]
                                )
                                
                                # Create scatter plot with trend line
                                fig = px.scatter(
                                    valid_data,
                                    x=col1,
                                    y=col2,
                                    trendline="ols",
                                    title=f"{col1} vs {col2} / {col1} против {col2}"
                                )
                                
                                # Add statistical information
                                significance = "significant / значительный" if p_value < 0.05 else "not significant / не значительный"
                                correlation_strength = abs(correlation)
                                strength_text = (
                                    "strong / сильная" if correlation_strength >= 0.7 else
                                    "moderate / умеренная" if correlation_strength >= 0.5 else
                                    "weak / слабая"
                                )
                                
                                annotation_text = (
                                    f"Correlation: {correlation:.2f}\n"
                                    f"P-value: {p_value:.4f}\n"
                                    f"Relationship: {significance}\n"
                                    f"Strength: {strength_text}"
                                )
                                
                                fig.add_annotation(
                                    xref="paper",
                                    yref="paper",
                                    x=0.02,
                                    y=0.98,
                                    text=annotation_text,
                                    showarrow=False,
                                    bgcolor="white",
                                    bordercolor="black",
                                    borderwidth=1
                                )
                                
                                results['figures'].append(fig)
                                
                                # Add insight
                                direction = "positive / положительная" if correlation > 0 else "negative / отрицательная"
                                insight = (
                                    f"Found {strength_text} {direction} correlation between "
                                    f"{col1} and {col2} (r={correlation:.2f}, p={p_value:.4f})"
                                )
                                results['insights'].append(insight)

            # Analyze categorical relationships
            categorical_cols = df.select_dtypes(include=['object']).columns
            for i in range(len(categorical_cols)):
                for j in range(i + 1, len(categorical_cols)):
                    col1, col2 = categorical_cols[i], categorical_cols[j]
                    
                    # Create contingency table
                    contingency = pd.crosstab(df[col1], df[col2])
                    
                    # Perform chi-square test
                    if contingency.size > 1:  # Need at least 2 categories
                        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                        
                        # Create heatmap of relationships
                        fig = px.imshow(
                            contingency,
                            title=f"Relationship between {col1} and {col2} / "
                                  f"Связь между {col1} и {col2}"
                        )
                        
                        # Add statistical information
                        significance = "significant / значительный" if p_value < 0.05 else "not significant / не значительный"
                        annotation_text = (
                            f"Chi-square: {chi2:.2f}\n"
                            f"P-value: {p_value:.4f}\n"
                            f"Relationship: {significance}"
                        )
                        
                        fig.add_annotation(
                            xref="paper",
                            yref="paper",
                            x=0.02,
                            y=0.98,
                            text=annotation_text,
                            showarrow=False,
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=1
                        )
                        
                        results['figures'].append(fig)
                        
                        # Add insight
                        if p_value < 0.05:
                            insight = (
                                f"Found significant relationship between categories "
                                f"{col1} and {col2} (chi2={chi2:.2f}, p={p_value:.4f})"
                            )
                            results['insights'].append(insight)

            return results

        except Exception as e:
            return {
                'error': str(e),
                'figures': [],
                'insights': []
            }
