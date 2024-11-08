# ... [previous imports remain unchanged] ...
from scipy import stats
from collections import defaultdict

class TextAnalyzer:
    # ... [previous methods remain unchanged] ...

    def analyze_text_patterns(self, text_column: str) -> Dict[str, Any]:
        """
        Detect patterns in text data using NLP techniques.
        """
        try:
            if text_column not in self.df.columns:
                return {
                    'error': f"Column {text_column} not found",
                    'patterns': {}
                }

            texts = self.df[text_column].dropna().astype(str)
            if texts.empty:
                return {
                    'error': "No valid text data",
                    'patterns': {}
                }

            # Analyze text length patterns
            lengths = texts.str.len()
            length_stats = {
                'mean': lengths.mean(),
                'std': lengths.std(),
                'min': lengths.min(),
                'max': lengths.max()
            }

            # Analyze word frequency patterns
            all_words = []
            for text in texts:
                words = self._preprocess_text(text)
                all_words.extend(words)

            word_freq = Counter(all_words)
            common_patterns = dict(word_freq.most_common(10))

            return {
                'error': None,
                'patterns': {
                    'length_statistics': length_stats,
                    'common_patterns': common_patterns
                }
            }

        except Exception as e:
            return {
                'error': str(e),
                'patterns': {}
            }

    def analyze_topic_correlations(self) -> Dict[str, Any]:
        """
        Analyze correlations between topics and other variables.
        """
        try:
            if 'rank' not in self.df.columns:
                return {
                    'error': "Topic column not found",
                    'correlations': {}
                }

            results = {
                'topic_trl_correlation': {},
                'topic_year_distribution': {},
                'topic_technology_association': {}
            }

            # Analyze topic-TRL relationships
            if 'TRL' in self.df.columns:
                topic_trl = self.df.groupby('rank')['TRL'].agg(['mean', 'std']).to_dict('index')
                results['topic_trl_correlation'] = topic_trl

            # Analyze topic distribution over years
            if 'Year' in self.df.columns:
                topic_year = pd.crosstab(
                    self.df['Year'],
                    self.df['rank'],
                    normalize='index'
                ).to_dict()
                results['topic_year_distribution'] = topic_year

            # Analyze topic-technology associations
            if 'Technologies' in self.df.columns:
                topic_tech = defaultdict(lambda: defaultdict(int))
                for _, row in self.df.iterrows():
                    if pd.notna(row['Technologies']):
                        topic = row['rank']
                        techs = row['Technologies'].split(';')
                        for tech in techs:
                            if tech:
                                topic_tech[topic][tech] += 1
                results['topic_technology_association'] = dict(topic_tech)

            return {
                'error': None,
                'correlations': results
            }

        except Exception as e:
            return {
                'error': str(e),
                'correlations': {}
            }

    def analyze_trl_progression(self) -> Dict[str, Any]:
        """
        Analyze TRL progression patterns over time and across topics.
        """
        try:
            if 'TRL' not in self.df.columns or 'Year' not in self.df.columns:
                return {
                    'error': "Required columns missing",
                    'progression': {}
                }

            results = {}

            # Calculate yearly TRL progression
            yearly_progression = self.df.groupby('Year')['TRL'].agg([
                'mean',
                'std',
                'count'
            ]).to_dict('index')
            results['yearly_progression'] = yearly_progression

            # Calculate TRL progression by topic
            topic_progression = self.df.groupby(['rank', 'Year'])['TRL'].mean().unstack().to_dict()
            results['topic_progression'] = topic_progression

            # Identify TRL trends
            valid_data = self.df[['Year', 'TRL']].dropna()
            if len(valid_data) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    valid_data['Year'],
                    valid_data['TRL']
                )
                results['trend_analysis'] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'std_err': std_err
                }

            return {
                'error': None,
                'progression': results
            }

        except Exception as e:
            return {
                'error': str(e),
                'progression': {}
            }

    # ... [rest of the class methods remain unchanged] ...
