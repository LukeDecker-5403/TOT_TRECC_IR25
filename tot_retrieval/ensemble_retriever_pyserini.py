"""
Lucene-based Ensemble Retriever using Pyserini
Much faster and more scalable than pure Python BM25
"""

import json
import os
import tempfile
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from pathlib import Path

# Pyserini imports
from pyserini.index.lucene import IndexReader, LuceneIndexer
from pyserini.search.lucene import LuceneSearcher

from .config import Config

@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    doc_id: str
    score: float
    field_scores: Dict[str, float]
    metadata: Dict[str, Any]

class LuceneFieldRetriever:
    """Lucene-based retriever for a single field"""
    
    def __init__(self, field_name: str, index_dir: str):
        """
        Initialize Lucene field retriever
        
        Args:
            field_name: Name of the field to search
            index_dir: Directory containing the Lucene index
        """
        self.field_name = field_name
        self.index_dir = index_dir
        self.field_index_dir = os.path.join(index_dir, f"{field_name}_index")
        self.searcher = None
        
    def build_index(self, documents: List[Dict[str, Any]]):
        """
        Build Lucene index for this field
        
        Args:
            documents: List of documents
        """
        print(f"Building Lucene index for field: {self.field_name}")
        
        # Create temporary directory for documents
        temp_dir = tempfile.mkdtemp()
        json_dir = os.path.join(temp_dir, "documents")
        os.makedirs(json_dir, exist_ok=True)
        
        # Convert documents to Pyserini JSON format
        for i, doc in enumerate(documents):
            doc_id = doc['doc_id']
            field_content = doc.get(self.field_name, '')
            
            # Skip empty fields
            if not field_content or field_content == "N/A":
                field_content = " "  # Lucene needs some content
            
            # Create JSON document
            json_doc = {
                'id': doc_id,
                'contents': field_content,
                'metadata': json.dumps(doc)  # Store full document as metadata
            }
            
            # Write to file
            with open(os.path.join(json_dir, f'doc_{i}.json'), 'w') as f:
                json.dump(json_doc, f)
        
        # Build Lucene index
        os.makedirs(self.field_index_dir, exist_ok=True)
        
        # Use Pyserini's indexer
        indexer = LuceneIndexer(self.field_index_dir)
        
        # Index all JSON files
        for filename in os.listdir(json_dir):
            filepath = os.path.join(json_dir, filename)
            with open(filepath, 'r') as f:
                doc_json = json.load(f)
                indexer.add_doc_dict(doc_json)
        
        indexer.close()
        
        # Initialize searcher
        self.searcher = LuceneSearcher(self.field_index_dir)
        self.searcher.set_bm25(k1=Config.BM25_K1, b=Config.BM25_B)
        
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"  Index built: {self.field_index_dir}")
    
    def retrieve(self, query: str, k: int = 1000) -> Dict[str, float]:
        """
        Retrieve documents for query using Lucene
        
        Args:
            query: Query string
            k: Number of results to retrieve
            
        Returns:
            Dictionary mapping doc_id to score
        """
        if not self.searcher:
            self.searcher = LuceneSearcher(self.field_index_dir)
            self.searcher.set_bm25(k1=Config.BM25_K1, b=Config.BM25_B)
        
        if not query or query == "N/A":
            return {}
        
        try:
            # Search using Lucene
            hits = self.searcher.search(query, k=k)
            
            # Convert to score dictionary
            scores = {}
            for hit in hits:
                scores[hit.docid] = hit.score
            
            return scores
        
        except Exception as e:
            print(f"Error searching field {self.field_name}: {e}")
            return {}

class PyseriniEnsembleRetriever:
    """Ensemble retriever using Lucene/Pyserini"""
    
    def __init__(self, index_dir: str = "./lucene_indices", 
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize Pyserini ensemble retriever
        
        Args:
            index_dir: Directory to store Lucene indices
            weights: Optional custom weights for each field
        """
        self.index_dir = index_dir
        self.weights = weights or Config.DEFAULT_WEIGHTS.copy()
        self.retrievers = {}
        self.documents = []
        self.doc_metadata = {}
        
        # Create index directory
        os.makedirs(self.index_dir, exist_ok=True)
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """
        Build Lucene indices for all fields
        
        Args:
            documents: List of documents with all fields
        """
        print(f"Building Lucene indices for {len(documents)} documents...")
        self.documents = documents
        
        # Store metadata
        for doc in documents:
            self.doc_metadata[doc['doc_id']] = doc
        
        # Build retriever for each field
        fields = ['plot', 'title', 'author', 'genre', 'date', 'cover']
        
        for field in fields:
            print(f"\n=== Building index for field: {field} ===")
            retriever = LuceneFieldRetriever(field, self.index_dir)
            retriever.build_index(documents)
            self.retrievers[field] = retriever
        
        print("\nAll Lucene indices built successfully!")
    
    def load_index(self):
        """Load existing Lucene indices"""
        print("Loading existing Lucene indices...")
        
        fields = ['plot', 'title', 'author', 'genre', 'date', 'cover']
        
        for field in fields:
            field_index_dir = os.path.join(self.index_dir, f"{field}_index")
            
            if os.path.exists(field_index_dir):
                retriever = LuceneFieldRetriever(field, self.index_dir)
                retriever.searcher = LuceneSearcher(field_index_dir)
                retriever.searcher.set_bm25(k1=Config.BM25_K1, b=Config.BM25_B)
                self.retrievers[field] = retriever
                print(f"  Loaded {field} index")
            else:
                print(f"  Warning: {field} index not found at {field_index_dir}")
    
    def load_metadata_from_index(self):
        """
        Load metadata from indexed documents.
        Note: Metadata may not be stored in the index by default.
        This method attempts to extract it, but if it fails, metadata
        should be loaded from the data file instead.
        """
        if not self.retrievers:
            return
        
        # Use any available index to extract metadata
        # All indices should have the same metadata stored
        first_field = list(self.retrievers.keys())[0] if self.retrievers else None
        if not first_field:
            return
        
        retriever = self.retrievers[first_field]
        if not retriever.searcher:
            return
        
        try:
            # Get index reader to access stored fields
            from pyserini.index.lucene import IndexReader
            index_reader = IndexReader(retriever.field_index_dir)
            stats = index_reader.stats()
            num_docs = stats.get('documents', 0)
            
            if num_docs == 0:
                return
            
            print(f"  Attempting to extract metadata from {num_docs} indexed documents...")
            
            # Try to extract metadata - note that metadata may not be stored
            # in retrievable format by pyserini's default indexing
            extracted = 0
            for i in range(min(num_docs, 1000)):  # Check first 1000 documents
                try:
                    doc_id = f"doc_{i}"
                    doc = index_reader.doc(doc_id)
                    if doc is None:
                        continue
                    
                    # Try to get metadata field
                    metadata_str = doc.get('metadata')
                    if metadata_str:
                        try:
                            import json
                            metadata = json.loads(metadata_str)
                            actual_doc_id = doc.id()
                            self.doc_metadata[actual_doc_id] = metadata
                            extracted += 1
                        except (json.JSONDecodeError, TypeError):
                            pass
                except Exception:
                    continue
            
            if extracted == 0:
                        print(f"  Warning: Metadata not stored in index format (this is normal)")
                        print(f"     To get full metadata, provide --data_file when searching")
            else:
                        print(f"  Extracted metadata for {extracted} documents")
        except Exception as e:
            print(f"  Warning: Could not extract metadata from index: {e}")
            print(f"     To get full metadata, provide --data_file when searching")
    
    def retrieve(self, decomposed_query: Dict[str, str], 
                top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve documents using ensemble approach with Lucene
        
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
                print(f"  Searching {field}: '{query[:50]}...'")
                field_scores = self.retrievers[field].retrieve(query, k=1000)
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
    
    def retrieve_all(self, max_docs: int = 1000) -> List[RetrievalResult]:
        """
        Retrieve all documents (for constraint-only queries)
        
        Args:
            max_docs: Maximum number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects with uniform scores
        """
        from pyserini.index.lucene import IndexReader
        
        # Get all document IDs from the index
        all_doc_ids = set()
        
        # Use the first available retriever to access the index
        for field, retriever in self.retrievers.items():
            if retriever.searcher or os.path.exists(retriever.field_index_dir):
                try:
                    # Use IndexReader to get all document IDs
                    index_reader = IndexReader(retriever.field_index_dir)
                    stats = index_reader.stats()
                    total_docs = min(stats.get('documents', 0), max_docs)
                    
                    # Get all document IDs by iterating through the index
                    for i in range(total_docs):
                        try:
                            docid = f"doc_{i}"  # Assuming doc IDs are in this format
                            doc = index_reader.doc(docid)
                            if doc:
                                all_doc_ids.add(docid)
                        except:
                            # Try alternative docid format
                            try:
                                docids = index_reader.docids()
                                if i < len(docids):
                                    all_doc_ids.add(docids[i])
                            except:
                                pass
                    break  # Got what we need from one index
                except Exception as e:
                    # Fallback: use a very broad query
                    if retriever.searcher:
                        try:
                            # Try searching with a common word
                            hits = retriever.searcher.search("the", k=max_docs * 2)
                            for hit in hits:
                                all_doc_ids.add(hit.docid)
                            if all_doc_ids:
                                break
                        except:
                            pass
        
        # If we still don't have documents, try loading from metadata
        if not all_doc_ids and self.doc_metadata:
            all_doc_ids = set(self.doc_metadata.keys())[:max_docs]
        
        # Create results with uniform scores
        results = []
        for doc_id in list(all_doc_ids)[:max_docs]:
            result = RetrievalResult(
                doc_id=doc_id,
                score=1.0,  # Uniform score for constraint-only queries
                field_scores={},
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
        Optimize retriever weights using validation data
        
        Args:
            val_queries: List of decomposed validation queries
            val_labels: Ground truth document IDs
            metric: Metric to optimize
            n_trials: Number of random trials
            
        Returns:
            Optimized weights dictionary
        """
        print(f"Optimizing weights using {metric} on {len(val_queries)} queries...")
        
        best_weights = self.weights.copy()
        best_score = 0.0
        
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
        print(f"✅ Optimization complete! Best {metric} = {best_score:.4f}")
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
            'index_dir': self.index_dir,
            'retrievers': {}
        }
        
        for field in self.retrievers:
            field_index_dir = os.path.join(self.index_dir, f"{field}_index")
            if os.path.exists(field_index_dir):
                stats['retrievers'][field] = {
                    'index_path': field_index_dir,
                    'index_exists': True
                }
        
        return stats

# Alias for backward compatibility
EnsembleRetriever = PyseriniEnsembleRetriever
