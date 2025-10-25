"""
Main interface for TOT Retrieval System with Lucene/Pyserini
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional

# Import rule-based decomposer (no API key needed)
from query_decomposer_free import RuleBasedQueryDecomposer as QueryDecomposer

# Import Lucene-based retriever
from ensemble_retriever_pyserini import PyseriniEnsembleRetriever as EnsembleRetriever

from evaluation import TOTEvaluator
from data_loader import DataLoader
from config import Config

class TOTRetrievalSystem:
    """Main interface for TOT Retrieval System with Lucene"""
    
    def __init__(self, index_dir: str = "./lucene_indices"):
        """
        Initialize TOT Retrieval System
        
        Args:
            index_dir: Directory for Lucene indices
        """
        self.index_dir = index_dir
        self.query_decomposer = None
        self.ensemble_retriever = None
        self.evaluator = None
        self.data_loader = DataLoader(Config.DATA_DIR)
        
        # Create necessary directories
        Config.create_directories()
        os.makedirs(index_dir, exist_ok=True)
    
    def setup(self, documents: List[Dict[str, Any]], rebuild_index: bool = True):
        """
        Setup the system with documents
        
        Args:
            documents: List of document dictionaries
            rebuild_index: Whether to rebuild Lucene indices (set False to load existing)
        """
        print("=" * 60)
        print("Setting up TOT Retrieval System with Lucene")
        print("=" * 60)
        
        # Initialize query decomposer (rule-based, no API key needed)
        print("\n📋 Initializing rule-based query decomposer...")
        self.query_decomposer = QueryDecomposer()
        print("✅ Query decomposer ready (no API key needed)")
        
        # Initialize ensemble retriever
        print("\n🔍 Initializing Lucene-based ensemble retriever...")
        self.ensemble_retriever = EnsembleRetriever(index_dir=self.index_dir)
        
        # Build or load indices
        if rebuild_index:
            print(f"\n📊 Building Lucene indices for {len(documents)} documents...")
            print("This may take a few minutes...")
            self.ensemble_retriever.build_index(documents)
        else:
            print("\n📂 Loading existing Lucene indices...")
            self.ensemble_retriever.load_index()
            # Still need metadata
            for doc in documents:
                self.ensemble_retriever.doc_metadata[doc['doc_id']] = doc
        
        # Initialize evaluator
        self.evaluator = TOTEvaluator(self.ensemble_retriever, self.query_decomposer)
        
        print("\n" + "=" * 60)
        print("✅ System setup complete!")
        print("=" * 60)
    
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
        
        print("\n" + "=" * 60)
        print(f"🔍 Processing query: '{query}'")
        print("=" * 60)
        
        # Decompose query
        print("\n📋 Decomposing query...")
        decomposed = self.query_decomposer.decompose(query, mode)
        decomposed_dict = decomposed.to_dict()
        
        print("\n📝 Decomposed query:")
        for field, subquery in decomposed_dict.items():
            if subquery and subquery != "N/A":
                print(f"  • {field:8s}: {subquery}")
        
        # Retrieve documents using Lucene
        print(f"\n🔍 Retrieving top {top_k} documents from Lucene...")
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
        
        print(f"\n✅ Found {len(formatted_results)} results")
        return formatted_results
    
    def evaluate(self, test_queries: List[str], test_labels: List[str], 
                 mode: str = "extractive") -> Dict[str, Any]:
        """Evaluate the system on test data"""
        if not self.evaluator:
            raise ValueError("System not set up. Call setup() first.")
        
        print("\n" + "=" * 60)
        print("📊 Evaluating system...")
        print("=" * 60)
        
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
        """Optimize retriever weights using validation data"""
        if not self.ensemble_retriever:
            raise ValueError("System not set up. Call setup() first.")
        
        print("\n" + "=" * 60)
        print("⚙️  Optimizing retriever weights...")
        print("=" * 60)
        
        # Prepare validation data
        val_decomposed = []
        for query in val_queries:
            decomposed = self.query_decomposer.decompose(query, mode)
            val_decomposed.append(decomposed.to_dict())
        
        # Optimize weights
        optimized_weights = self.ensemble_retriever.optimize_weights(
            val_decomposed, val_labels, metric="recall@5"
        )
        
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
    parser = argparse.ArgumentParser(description="TOT Retrieval System with Lucene")
    parser.add_argument("--mode", choices=["demo", "evaluate", "optimize", "search"], 
                       default="demo", help="Operation mode")
    parser.add_argument("--query", type=str, help="Query to search (for search mode)")
    parser.add_argument("--data_file", type=str, help="Data file to load")
    parser.add_argument("--index_dir", type=str, default="./lucene_indices",
                       help="Directory for Lucene indices")
    parser.add_argument("--rebuild_index", action="store_true",
                       help="Rebuild Lucene indices (default: load existing)")
    parser.add_argument("--decomposition_mode", choices=["extractive", "predictive"], 
                       default="extractive", help="Query decomposition mode")
    
    args = parser.parse_args()
    
    # Initialize system
    system = TOTRetrievalSystem(index_dir=args.index_dir)
    
    if args.mode == "demo":
        # Demo mode with synthetic data
        print("🎭 Running demo with synthetic book dataset...")
        
        # Create synthetic dataset
        documents = system.data_loader.create_synthetic_dataset(num_books=6)
        
        # Setup system
        system.setup(documents, rebuild_index=True)
        
        # Demo queries
        demo_queries = [
            "A book about a wealthy man who throws parties",
            "Something about a lawyer defending a black man",
            "A dystopian book about surveillance",
            "A classic romance about Elizabeth and Mr. Darcy",
            "A coming-of-age story about a teenager"
        ]
        
        for query in demo_queries:
            results = system.search(query, mode=args.decomposition_mode, top_k=3)
            
            print("\n📚 Top Results:")
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                print(f"\n  {i}. {metadata.get('title', 'Unknown')}")
                print(f"     Author: {metadata.get('author', 'Unknown')}")
                print(f"     Score: {result['score']:.4f}")
                
                # Show top contributing fields
                field_scores = result['field_scores']
                sorted_fields = sorted(field_scores.items(), key=lambda x: x[1], reverse=True)
                print(f"     Top fields: {', '.join([f'{k}({v:.2f})' for k, v in sorted_fields[:3]])}")
    
    elif args.mode == "search":
        # Single query search mode
        if not args.query:
            print("❌ Error: --query required for search mode")
            return
        
        if not args.data_file:
            print("❌ Error: --data_file required for search mode")
            return
        
        print(f"📂 Loading data from {args.data_file}...")
        documents = system.data_loader.load_dataset(args.data_file)
        
        # Setup system (load existing index if available)
        system.setup(documents, rebuild_index=args.rebuild_index)
        
        # Search
        results = system.search(args.query, mode=args.decomposition_mode, top_k=10)
        
        print("\n📚 Search Results:")
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            print(f"\n{i}. {metadata.get('title', 'Unknown')} "
                  f"by {metadata.get('author', 'Unknown')}")
            print(f"   Score: {result['score']:.4f}")
    
    elif args.mode == "evaluate":
        # Evaluation mode
        if not args.data_file:
            print("❌ Error: --data_file required for evaluation mode")
            return
        
        print(f"📂 Loading data from {args.data_file}...")
        documents = system.data_loader.load_dataset(args.data_file)
        
        # Setup system
        system.setup(documents, rebuild_index=args.rebuild_index)
        
        # Load test queries (you'll need to create this file)
        test_queries_file = args.data_file.replace('.json', '_queries.json')
        if os.path.exists(test_queries_file):
            test_queries, test_labels = system.data_loader.load_queries_and_labels(test_queries_file)
        else:
            print(f"⚠️  Test queries file not found: {test_queries_file}")
            print("Creating sample queries from documents...")
            query_data = system.data_loader.create_sample_queries(documents, num_queries=10)
            test_queries = [q['query'] for q in query_data]
            test_labels = [q['doc_id'] for q in query_data]
        
        # Evaluate
        results = system.evaluate(test_queries, test_labels, args.decomposition_mode)
        print("\n" + results["report"])
    
    elif args.mode == "optimize":
        # Weight optimization mode
        if not args.data_file:
            print("❌ Error: --data_file required for optimization mode")
            return
        
        print(f"📂 Loading data from {args.data_file}...")
        documents = system.data_loader.load_dataset(args.data_file)
        
        # Setup system
        system.setup(documents, rebuild_index=args.rebuild_index)
        
        # Create validation data
        val_queries_file = args.data_file.replace('.json', '_val_queries.json')
        if os.path.exists(val_queries_file):
            
