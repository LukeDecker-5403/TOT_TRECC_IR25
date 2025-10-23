"""
Main interface for TOT Retrieval System
"""

import os
import json
from typing import List, Dict, Any, Optional
import argparse
from .query_decomposer import QueryDecomposer
from .ensemble_retriever import EnsembleRetriever
from .evaluation import TOTEvaluator
from .data_loader import DataLoader
from .config import Config

class TOTRetrievalSystem:
    """Main interface for TOT Retrieval System"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TOT Retrieval System
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.query_decomposer = None
        self.ensemble_retriever = None
        self.evaluator = None
        self.data_loader = DataLoader(Config.DATA_DIR)
        
        # Create cache directory
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
    
    def setup(self, documents: List[Dict[str, Any]]):
        """
        Setup the system with documents
        
        Args:
            documents: List of document dictionaries
        """
        print("Setting up TOT Retrieval System...")
        
        # Initialize query decomposer
        print("Initializing query decomposer...")
        self.query_decomposer = QueryDecomposer()
        
        # Initialize ensemble retriever
        print("Initializing ensemble retriever...")
        self.ensemble_retriever = EnsembleRetriever()
        
        # Build indices
        print("Building retrieval indices...")
        self.ensemble_retriever.build_index(documents)
        
        # Initialize evaluator
        self.evaluator = TOTEvaluator(self.ensemble_retriever, self.query_decomposer)
        
        print("System setup complete!")
    
    def search(self, query: str, mode: str = "extractive", top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents using a TOT query
        
        Args:
            query: Tip-of-the-tongue query
            mode: Decomposition mode ("extractive" or "predictive")
            top_k: Number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
        if not self.ensemble_retriever:
            raise ValueError("System not set up. Call setup() first.")
        
        print(f"Processing query: '{query}'")
        
        # Decompose query
        print("Decomposing query...")
        decomposed = self.query_decomposer.decompose(query, mode)
        decomposed_dict = decomposed.to_dict()
        
        print("Decomposed query:")
        for field, subquery in decomposed_dict.items():
            print(f"  {field}: {subquery}")
        
        # Retrieve documents
        print("Retrieving documents...")
        results = self.ensemble_retriever.retrieve(decomposed_dict, top_k)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_result = {
                "doc_id": result.doc_id,
                "score": result.score,
                "field_scores": result.field_scores,
                "metadata": result.metadata
            }
            formatted_results.append(formatted_result)
        
        print(f"Found {len(formatted_results)} results")
        return formatted_results
    
    def evaluate(self, test_queries: List[str], test_labels: List[str], 
                 mode: str = "extractive") -> Dict[str, Any]:
        """
        Evaluate the system on test data
        
        Args:
            test_queries: List of test queries
            test_labels: List of ground truth document IDs
            mode: Decomposition mode
            
        Returns:
            Evaluation results dictionary
        """
        if not self.evaluator:
            raise ValueError("System not set up. Call setup() first.")
        
        print("Evaluating system...")
        
        # Overall evaluation
        metrics = self.evaluator.evaluate(test_queries, test_labels, mode)
        
        # Ablation study
        ablation_results = self.evaluator.ablation_study(test_queries, test_labels, mode)
        
        # Per-field analysis
        field_metrics = self.evaluator.per_field_analysis(test_queries, test_labels, mode)
        
        # Generate report
        report = self.evaluator.generate_report(metrics, ablation_results, field_metrics)
        
        return {
            "metrics": metrics,
            "ablation_results": ablation_results,
            "field_metrics": field_metrics,
            "report": report
        }
    
    def optimize_weights(self, val_queries: List[str], val_labels: List[str], 
                        mode: str = "extractive") -> Dict[str, float]:
        """
        Optimize retriever weights using validation data
        
        Args:
            val_queries: List of validation queries
            val_labels: List of ground truth document IDs
            mode: Decomposition mode
            
        Returns:
            Optimized weights dictionary
        """
        if not self.ensemble_retriever:
            raise ValueError("System not set up. Call setup() first.")
        
        print("Optimizing retriever weights...")
        
        # Prepare validation data
        val_decomposed = []
        for query in val_queries:
            decomposed = self.query_decomposer.decompose(query, mode)
            val_decomposed.append(decomposed.to_dict())
        
        # Optimize weights
        optimized_weights = self.ensemble_retriever.optimize_weights(
            val_decomposed, val_labels, metric="recall@5"
        )
        
        print(f"Optimized weights: {optimized_weights}")
        return optimized_weights
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            "query_decomposer": self.query_decomposer.get_cache_stats() if self.query_decomposer else None,
            "ensemble_retriever": self.ensemble_retriever.get_stats() if self.ensemble_retriever else None
        }
        return stats

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description="TOT Retrieval System")
    parser.add_argument("--mode", choices=["demo", "evaluate", "optimize"], 
                       default="demo", help="Operation mode")
    parser.add_argument("--query", type=str, help="Query to search")
    parser.add_argument("--data_file", type=str, help="Data file to load")
    parser.add_argument("--decomposition_mode", choices=["extractive", "predictive"], 
                       default="extractive", help="Query decomposition mode")
    
    args = parser.parse_args()
    
    # Initialize system
    system = TOTRetrievalSystem()
    
    if args.mode == "demo":
        # Demo mode with synthetic data
        print("Running demo with synthetic data...")
        
        # Create synthetic dataset
        data_loader = DataLoader()
        documents = data_loader.create_synthetic_dataset(num_books=50)
        
        # Setup system
        system.setup(documents)
        
        # Demo queries
        demo_queries = [
            "A book about a wealthy man who throws parties",
            "Something about a lawyer defending a black man",
            "A dystopian book about surveillance",
            "A classic romance about Elizabeth and Mr. Darcy",
            "A coming-of-age story about a teenager"
        ]
        
        for query in demo_queries:
            print(f"\n{'='*50}")
            print(f"Query: {query}")
            print('='*50)
            
            results = system.search(query, mode=args.decomposition_mode, top_k=5)
            
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['metadata'].get('title', 'Unknown')} "
                      f"by {result['metadata'].get('author', 'Unknown')} "
                      f"(Score: {result['score']:.3f})")
    
    elif args.mode == "evaluate":
        # Evaluation mode
        if not args.data_file:
            print("Error: --data_file required for evaluation mode")
            return
        
        print(f"Loading data from {args.data_file}...")
        data_loader = DataLoader()
        documents = data_loader.load_dataset(args.data_file)
        
        # Setup system
        system.setup(documents)
        
        # Create test data (in practice, you'd load from separate test file)
        test_queries = [
            "A book about a wealthy man who throws parties",
            "Something about a lawyer defending a black man",
            "A dystopian book about surveillance"
        ]
        test_labels = ["book_0", "book_1", "book_2"]  # Ground truth IDs
        
        # Evaluate
        results = system.evaluate(test_queries, test_labels, args.decomposition_mode)
        print("\nEvaluation Results:")
        print(results["report"])
    
    elif args.mode == "optimize":
        # Weight optimization mode
        if not args.data_file:
            print("Error: --data_file required for optimization mode")
            return
        
        print(f"Loading data from {args.data_file}...")
        data_loader = DataLoader()
        documents = data_loader.load_dataset(args.data_file)
        
        # Setup system
        system.setup(documents)
        
        # Create validation data
        val_queries = [
            "A book about a wealthy man who throws parties",
            "Something about a lawyer defending a black man"
        ]
        val_labels = ["book_0", "book_1"]
        
        # Optimize weights
        optimized_weights = system.optimize_weights(val_queries, val_labels, args.decomposition_mode)
        print(f"Optimized weights: {optimized_weights}")

if __name__ == "__main__":
    main()
