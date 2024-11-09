[Previous content of text_analyzer.py up to line 381]

    def semantic_search(self, query: str, text_columns: List[str]) -> Dict[str, Any]:
        try:
            results = {}
            total_mentions = 0
            mentions_by_column = {}
            
            # Create TF-IDF vectorizer for semantic similarity
            vectorizer = TfidfVectorizer(stop_words=self.stop_words)
            
            for column in text_columns:
                if column not in self.df.columns:
                    continue
                    
                texts = self.df[column].fillna('').astype(str)
                if texts.empty:
                    continue
                    
                # Calculate TF-IDF matrices
                tfidf_matrix = vectorizer.fit_transform(texts)
                query_vector = vectorizer.transform([query])
                
                # Calculate similarity scores
                similarities = (query_vector * tfidf_matrix.T).toarray()[0]
                
                # Get matches above threshold
                threshold = 0.3
                matches = [(idx, score) for idx, score in enumerate(similarities) if score > threshold]
                
                if matches:
                    mentions_by_column[column] = {
                        'count': len(matches),
                        'mentions': [
                            {
                                'text': texts.iloc[idx],
                                'similarity': score,
                                'metadata': {
                                    key: self.df.iloc[idx][key] 
                                    for key in ['Title', 'Authors', 'Year'] 
                                    if key in self.df.columns
                                }
                            }
                            for idx, score in sorted(matches, key=lambda x: x[1], reverse=True)
                        ]
                    }
                    total_mentions += len(matches)
            
            return {
                'total_mentions': total_mentions,
                'mentions_by_column': mentions_by_column
            }
            
        except Exception as e:
            return {'error': str(e)}
