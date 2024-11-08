import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from typing import List, Tuple, Dict
import numpy as np
from plotly.subplots import make_subplots

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

    def plot_topic_distribution(self, df: pd.DataFrame) -> go.Figure:
        try:
            validation = self._validate_dataframe(df, ['rank'])
            if not validation['valid']:
                return self._create_empty_figure(validation['message'])
            
            # Get topic distribution
            topic_counts = df['rank'].value_counts()
            
            if topic_counts.empty:
                return self._create_empty_figure(
                    "No topic data available / Нет данных о темах"
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
                f"Error creating topic distribution / Ошибка создания распределения тем: {str(e)}"
            )

    def plot_trl_distribution(self, df: pd.DataFrame) -> go.Figure:
        try:
            validation = self._validate_dataframe(df, ['TRL'])
            if not validation['valid']:
                return self._create_empty_figure(validation['message'])
            
            # Remove null values and convert to integers
            trl_data = pd.to_numeric(df['TRL'], errors='coerce').dropna()
            
            if trl_data.empty:
                return self._create_empty_figure(
                    "No TRL data available / Нет данных УГТ"
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
            validation = self._validate_dataframe(df, ['Technologies', 'Year'])
            if not validation['valid']:
                return self._create_empty_figure(validation['message'])
            
            # Get valid data
            valid_data = df[['Technologies', 'Year']].dropna()
            if valid_data.empty:
                return self._create_empty_figure(
                    "No technology trends data available / Нет данных о трендах технологий"
                )
            
            # Split technologies and create yearly counts
            tech_by_year = {}
            for _, row in valid_data.iterrows():
                year = int(row['Year'])
                technologies = row['Technologies'].split(';')
                if year not in tech_by_year:
                    tech_by_year[year] = {}
                for tech in technologies:
                    if tech:  # Skip empty strings
                        tech_by_year[year][tech] = tech_by_year[year].get(tech, 0) + 1
            
            if not tech_by_year:
                return self._create_empty_figure(
                    "No valid technology trends data / Нет действительных данных о трендах технологий"
                )
            
            # Create traces for each technology
            fig = go.Figure()
            all_years = sorted(tech_by_year.keys())
            all_technologies = set()
            for year_data in tech_by_year.values():
                all_technologies.update(year_data.keys())
            
            for tech in sorted(all_technologies):
                yearly_counts = [tech_by_year[year].get(tech, 0) for year in all_years]
                fig.add_trace(
                    go.Scatter(
                        x=all_years,
                        y=yearly_counts,
                        name=tech,
                        mode='lines+markers'
                    )
                )
            
            # Update layout
            fig.update_layout(
                title="Technology Mentions Over Time / Упоминания технологий с течением времени",
                xaxis_title="Year / Год",
                yaxis_title="Number of Mentions / Количество упоминаний",
                height=400,
                hovermode='x unified',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(
                f"Error creating technology trends / Ошибка создания трендов технологий: {str(e)}"
            )

    def plot_numerical_distribution(self, df: pd.DataFrame, column: str) -> go.Figure:
        """Create boxplot and histogram for numerical data"""
        try:
            validation = self._validate_dataframe(df, [column])
            if not validation['valid']:
                return self._create_empty_figure(validation['message'])
            
            # Remove null values and convert to numeric
            valid_data = pd.to_numeric(df[column], errors='coerce').dropna()
            
            if valid_data.empty:
                return self._create_empty_figure(
                    f"No valid numerical data for {column} / "
                    f"Нет действительных числовых данных для {column}"
                )
            
            # Create subplot with boxplot and histogram
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    f"Distribution of {column} / Распределение {column}",
                    f"Histogram of {column} / Гистограмма {column}"
                )
            )
            
            # Add boxplot
            fig.add_trace(
                go.Box(y=valid_data, name=column),
                row=1, col=1
            )
            
            # Add histogram
            fig.add_trace(
                go.Histogram(x=valid_data, name=column),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                showlegend=False,
                title_text=f"Analysis of {column} / Анализ {column}"
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(
                f"Error in numerical distribution / Ошибка в числовом распределении: {str(e)}"
            )

    def plot_categorical_distribution(self, df: pd.DataFrame, column: str) -> go.Figure:
        """Create bar chart for categorical data"""
        try:
            validation = self._validate_dataframe(df, [column])
            if not validation['valid']:
                return self._create_empty_figure(validation['message'])
            
            # Remove empty values
            valid_data = df[column].dropna().replace('', np.nan).dropna()
            
            if valid_data.empty:
                return self._create_empty_figure(
                    f"No valid categorical data for {column} / "
                    f"Нет действительных категориальных данных для {column}"
                )
            
            value_counts = valid_data.value_counts()
            
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                labels={
                    'x': f"{column} / Категория",
                    'y': 'Count / Количество'
                },
                title=f"Distribution of {column} / Распределение {column}"
            )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(
                f"Error in categorical distribution / Ошибка в категориальном распределении: {str(e)}"
            )

    def plot_text_analysis(self, df: pd.DataFrame, column: str) -> go.Figure:
        """Create text analysis visualization"""
        try:
            validation = self._validate_dataframe(df, [column])
            if not validation['valid']:
                return self._create_empty_figure(validation['message'])
            
            # Get text lengths and word counts
            valid_texts = df[column].dropna().astype(str)
            
            if valid_texts.empty:
                return self._create_empty_figure(
                    f"No valid text data for {column} / "
                    f"Нет действительных текстовых данных для {column}"
                )
            
            text_lengths = valid_texts.str.len()
            word_counts = valid_texts.str.split().str.len()
            
            # Create subplot with text length and word count distributions
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    "Text Length Distribution / Распределение длины текста",
                    "Word Count Distribution / Распределение количества слов"
                )
            )
            
            fig.add_trace(
                go.Box(y=text_lengths, name="Text Length"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Box(y=word_counts, name="Word Count"),
                row=1, col=2
            )
            
            fig.update_layout(
                height=400,
                title_text=f"Text Analysis for {column} / Текстовый анализ для {column}",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(
                f"Error in text analysis / Ошибка в текстовом анализе: {str(e)}"
            )

    def plot_correlation_matrix(self, df: pd.DataFrame, columns: List[str]) -> go.Figure:
        """Create correlation matrix for numerical variables"""
        try:
            validation = self._validate_dataframe(df, columns)
            if not validation['valid']:
                return self._create_empty_figure(validation['message'])
            
            # Convert columns to numeric and drop non-numeric
            numeric_df = df[columns].apply(pd.to_numeric, errors='coerce')
            
            if numeric_df.empty:
                return self._create_empty_figure(
                    "No valid numerical data for correlation / "
                    "Нет действительных числовых данных для корреляции"
                )
            
            correlation_matrix = numeric_df.corr()
            
            fig = px.imshow(
                correlation_matrix,
                labels=dict(
                    color="Correlation / Корреляция"
                ),
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            
            fig.update_layout(
                title="Correlation Matrix / Корреляционная матрица",
                height=500,
                xaxis_title="Variables / Переменные",
                yaxis_title="Variables / Переменные"
            )
            
            # Add correlation values as annotations
            for i in range(len(correlation_matrix.columns)):
                for j in range(len(correlation_matrix.columns)):
                    fig.add_annotation(
                        x=i,
                        y=j,
                        text=f"{correlation_matrix.iloc[i, j]:.2f}",
                        showarrow=False,
                        font_size=10,
                        font_color='black'
                    )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(
                f"Error in correlation matrix / Ошибка в корреляционной матрице: {str(e)}"
            )

    def plot_relationship(self, df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
        """Create scatter plot showing relationship between two variables"""
        try:
            validation = self._validate_dataframe(df, [x_col, y_col])
            if not validation['valid']:
                return self._create_empty_figure(validation['message'])
            
            # Convert to numeric and drop invalid values
            x_data = pd.to_numeric(df[x_col], errors='coerce')
            y_data = pd.to_numeric(df[y_col], errors='coerce')
            
            valid_mask = x_data.notna() & y_data.notna()
            x_data = x_data[valid_mask]
            y_data = y_data[valid_mask]
            
            if len(x_data) == 0:
                return self._create_empty_figure(
                    "No valid data for relationship plot / "
                    "Нет действительных данных для графика зависимости"
                )
            
            fig = px.scatter(
                x=x_data,
                y=y_data,
                labels={
                    'x': f"{x_col}",
                    'y': f"{y_col}"
                },
                title=f"Relationship between {x_col} and {y_col} / "
                      f"Зависимость между {x_col} и {y_col}"
            )
            
            # Add trend line
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=np.poly1d(np.polyfit(x_data, y_data, 1))(x_data),
                    name="Trend Line / Линия тренда",
                    mode='lines',
                    line=dict(color='red', dash='dash')
                )
            )
            
            fig.update_layout(
                height=400,
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(
                f"Error in relationship plot / Ошибка в графике зависимости: {str(e)}"
            )
