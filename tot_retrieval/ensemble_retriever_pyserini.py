"""
Ensemble Retriever for TOT Retrieval System
Combines multiple field-specific retrievers with learned weights
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import math
from .config import Config

@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    doc_id: str
    score: float
    field_scores: Dict[str, float]
    metadata: Dict[str, Any]

class BM25Retriever:
    """BM25 retriever for a single field"""
    
    def __init__(self, field_name: str, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever
        
        Args:
            field_name: Name of the field to index
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.field_name = field_name
        self.k1 = k1
        self.b = b
        self.documents = {}
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.idf_scores = {}
        self.vocab = set()
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """Build BM25 index from documents"""
        self.documents = {}
        doc_freq = defaultdict(int)
        total_length = 0
        
        for doc in documents:
            doc_id = doc['doc_id']
            field_value = doc.get(self.field_name, '')
            
            # Tokenize (simple whitespace tokenization)
            tokens = self._tokenize(field_value)
            self.documents[doc_id] = tokens
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            
            # Count document frequencies
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
            
            self.vocab.update(tokens)
        
        # Calculate average document length
        self.avg_doc_length = total_length / len(documents) if documents else 0
        
        # Calculate IDF scores
        num_docs = len(documents)
        for term, df in doc_freq.items():
            idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)
            self.idf_scores[term] = idf
    
    def retrieve(self, query: str) -> Dict[str, float]:
        """
        Retrieve documents for query
        
        Args:
            query: Query string
            
        Returns:
            Dictionary mapping doc_id to BM25 score
        """
        query_tokens = self._tokenize(query)
        scores = {}
        
        if not query_tokens:
            return scores
        
        for doc_id, doc_tokens in self.documents.items():
            score = self._calculate_bm25(query_tokens, doc_tokens, doc_id)
            scores[doc_id] = score
        
        return scores
    
    def _calculate_bm25(self, query_tokens: List[str], 
                       doc_tokens: List[str], doc_id: str) -> float:
        """Calculate BM25 score for a document"""
        score = 0.0
        doc_length = self.doc_lengths[doc_id]
        
        # Count term frequencies in document
        doc_term_freq = defaultdict(int)
        for token in doc_tokens:
            doc_term_freq[token] += 1
        
        for term in query_tokens:
            if term not in doc_term_freq:
                continue
            
            tf = doc_term_freq[term]
            idf = self.idf_scores.get(term, 0)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (lowercase and split)"""
        if not text or text == "N/A":
            return []
        return text.lower().split()

class EnsembleRetriever:
    """Ensemble retriever combining multiple field-specific retrievers"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble retriever
        
        Args:
            weights: Optional custom weights for each field
        """
        self.weights = weights or Config.DEFAULT_WEIGHTS.copy()
        self.retrievers = {}
        self.documents = []
        self.doc_metadata = {}
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """
        Build indices for all fields
        
        Args:
            documents: List of documents with all fields
        """
        print(f"Building indices for {len(documents)} documents...")
        self.documents = documents
        
        # Store metadata
        for doc in documents:
            self.doc_metadata[doc['doc_id']] = doc
        
        # Build retriever for each field
        fields = ['plot', 'title', 'author', 'genre', 'date', 'cover']
        
        for field in fields:
            print(f"  Building {field} index...")
            retriever = BM25Retriever(field, k1=Config.BM25_K1, b=Config.BM25_B)
            retriever.build_index(documents)
            self.retrievers[field] = retriever
        
        print("Index building complete!")
    
    def retrieve(self, decomposed_query: Dict[str, str], 
                top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve documents using ensemble approach
        
        Args:
            decomposed_query: Dictionary with field-specific queries
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult objects sorted by score
        """
        # Get scores from each retriever
        all_field_scores = {}
        
        for field, query in decomposed_query.items():
            if field in self.retrievers and query and query != "N/A":
                field_scores = self.retrievers[field].retrieve(query)
                all_field_scores[field] = self._normalize_scores(field_scores)
            else:
                all_field_scores[field] = {}
        
        # Aggregate scores
        aggregated_scores = self._aggregate_scores(all_field_scores)
        
        # Sort by score
        sorted_docs = sorted(aggregated_scores.items(), 
                           key=lambda x: x[1]['score'], 
                           reverse=True)
        
        # Create results
        results = []
        for doc_id, score_info in sorted_docs[:top_k]:
            result = RetrievalResult(
                doc_id=doc_id,
                score=score_info['score'],
                field_scores=score_info['field_scores'],
                metadata=self.doc_metadata.get(doc_id, {})
            )
            results.append(result)
        
        return results
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0, 1] range using min-max scaling"""
        if not scores:
            return {}
        
        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)
        
        if max_score == min_score:
            return {doc_id: 1.0 for doc_id in scores}
        
        normalized = {}
        for doc_id, score in scores.items():
            normalized[doc_id] = (score - min_score) / (max_score - min_score)
        
        return normalized
    
    def _aggregate_scores(self, all_field_scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict]:
        """Aggregate scores from all fields using weighted sum"""
        aggregated = defaultdict(lambda: {'score': 0.0, 'field_scores': {}})
        
        # Get all document IDs
        all_doc_ids = set()
        for field_scores in all_field_scores.values():
            all_doc_ids.update(field_scores.keys())
        
        # Calculate weighted scores
        for doc_id in all_doc_ids:
            total_score = 0.0
            field_contributions = {}
            
            for field, field_scores in all_field_scores.items():
                field_score = field_scores.get(doc_id, 0.0)
                weighted_score = field_score * self.weights.get(field, 0.0)
                total_score += weighted_score
                field_contributions[field] = field_score
            
            aggregated[doc_id]['score'] = total_score
            aggregated[doc_id]['field_scores'] = field_contributions
        
        return aggregated
    
    def optimize_weights(self, val_queries: List[Dict[str, str]], 
                        val_labels: List[str],
                        metric: str = "recall@5",
                        n_trials: int = 50) -> Dict[str, float]:
        """
        Optimize retriever weights using grid search
        
        Args:
            val_queries: List of decomposed validation queries
            val_labels: Ground truth document IDs
            metric: Metric to optimize (recall@5, mrr, etc.)
            n_trials: Number of random trials
            
        Returns:
            Optimized weights dictionary
        """
        print(f"Optimizing weights using {metric} on {len(val_queries)} queries...")
        
        best_weights = self.weights.copy()
        best_score = 0.0
        
        # Try different weight combinations
        for trial in range(n_trials):
            # Generate random weights
            weights = self._generate_random_weights()
            
            # Temporarily set weights
            original_weights = self.weights
            self.weights = weights
            
            # Evaluate
            score = self._evaluate_weights(val_queries, val_labels, metric)
            
            if score > best_score:
                best_score = score
                best_weights = weights.copy()
                print(f"  Trial {trial+1}: New best {metric} = {score:.4f}")
            
            # Restore original weights
            self.weights = original_weights
        
        # Set best weights
        self.weights = best_weights
        print(f"Optimization complete! Best {metric} = {best_score:.4f}")
        print(f"Optimized weights: {best_weights}")
        
        return best_weights
    
    def _generate_random_weights(self) -> Dict[str, float]:
        """Generate random normalized weights"""
        fields = ['plot', 'title', 'author', 'genre', 'date', 'cover']
        weights = {field: np.random.random() for field in fields}
        
        # Normalize
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _evaluate_weights(self, val_queries: List[Dict[str, str]], 
                         val_labels: List[str], metric: str) -> float:
        """Evaluate current weights on validation set"""
        scores = []
        
        for query, label in zip(val_queries, val_labels):
            results = self.retrieve(query, top_k=10)
            
            if metric == "recall@5":
                top_5_ids = [r.doc_id for r in results[:5]]
                score = 1.0 if label in top_5_ids else 0.0
            elif metric == "recall@10":
                top_10_ids = [r.doc_id for r in results[:10]]
                score = 1.0 if label in top_10_ids else 0.0
            elif metric == "mrr":
                score = 0.0
                for i, result in enumerate(results):
                    if result.doc_id == label:
                        score = 1.0 / (i + 1)
                        break
            else:
                score = 0.0
            
            scores.append(score)
        
        return np.mean(scores)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics"""
        stats = {
            'num_documents': len(self.documents),
            'weights': self.weights,
            'retrievers': {}
        }
        
        for field, retriever in self.retrievers.items():
            stats['retrievers'][field] = {
                'vocab_size': len(retriever.vocab),
                'avg_doc_length': retriever.avg_doc_length
            }
        
        return stats
