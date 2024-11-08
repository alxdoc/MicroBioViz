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

    # ... [previous methods remain unchanged] ...

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

    # ... [rest of the class methods remain unchanged] ...
