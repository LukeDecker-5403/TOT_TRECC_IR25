"""
Main interface for TOT Retrieval System with Lucene/Pyserini
"""

import os
import json
import argparse
from typing import List, Dict, Any
import re
import sys

# Handle both relative and absolute imports
try:
    from .config import Config
    # free for xhorxhi gui test
    # from .query_decomposer_free import QueryDecomposer

    from .query_decomposer import QueryDecomposer
    from .ensemble_retriever_pyserini import PyseriniEnsembleRetriever
    from .evaluation import TOTEvaluator
    from .data_loader import DataLoader
except ImportError:
    # If running directly, add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tot_retrieval.config import Config
    from tot_retrieval.query_decomposer import QueryDecomposer
    from tot_retrieval.ensemble_retriever_pyserini import PyseriniEnsembleRetriever
    from tot_retrieval.evaluation import TOTEvaluator
    from tot_retrieval.data_loader import DataLoader

class TOTRetrievalSystem:
    """Main interface for TOT Retrieval System with Lucene"""

    def __init__(self, index_dir: str = None):
        """
        Initialize TOT Retrieval System

        Args:
            index_dir: Directory for Lucene indices (defaults to Config.LUCENE_INDEX_DIR)
        """
        self.index_dir = index_dir if index_dir is not None else str(Config.LUCENE_INDEX_DIR)
        self.query_decomposer = None
        self.ensemble_retriever = None
        self.evaluator = None
        self.data_loader = DataLoader(Config.DATA_DIR)

        # Create necessary directories
        Config.create_directories()
        os.makedirs(self.index_dir, exist_ok=True)

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

        # Initialize query decomposer
        print("\nInitializing GPT-based query decomposer...")
        self.query_decomposer = QueryDecomposer()
        print("GPT-based query decomposer ready (using OpenAI API)")

        # Initialize ensemble retriever
        print("\nInitializing Lucene-based ensemble retriever...")
        self.ensemble_retriever = PyseriniEnsembleRetriever(index_dir=self.index_dir)

        # Build or load indices
        if rebuild_index:
            print(f"\n📊 Building Lucene indices for {len(documents)} documents...")
            print("This may take a few minutes...")
            self.ensemble_retriever.build_index(documents)
        else:
            print("\nLoading existing Lucene indices...")
            self.ensemble_retriever.load_index()
            # Still need metadata
            for doc in documents:
                self.ensemble_retriever.doc_metadata[doc['doc_id']] = doc

        # Initialize evaluator
        self.evaluator = TOTEvaluator(self.ensemble_retriever, self.query_decomposer)

        print("\n" + "=" * 60)
        print("System setup complete!")
        print("=" * 60)

    def setup_from_indices_only(self):
        """
        Setup the system using only existing indices (no documents needed).
        Useful for querying when indices already exist.
        """
        print("=" * 60)
        print("Setting up TOT Retrieval System from existing indices")
        print("=" * 60)

        # Initialize query decomposer
        print("\nInitializing GPT-based query decomposer...")
        self.query_decomposer = QueryDecomposer()
        print("GPT-based query decomposer ready (using OpenAI API)")

        # Initialize ensemble retriever
        print("\nLoading existing Lucene indices...")
        self.ensemble_retriever = PyseriniEnsembleRetriever(index_dir=self.index_dir)
        self.ensemble_retriever.load_index()
        
        # Try to extract metadata from index if possible
        # For now, metadata may be empty but search will still work
        self.ensemble_retriever.load_metadata_from_index()

        print("\n" + "=" * 60)
        print("System ready for querying!")
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
        print(f"Processing query: '{query}'")
        print("=" * 60)

        # Decompose query
        print("\nDecomposing query...")
        decomposed = self.query_decomposer.decompose(query, mode)
        decomposed_dict = decomposed.to_dict()

        print("\nDecomposed query:")
        for field, subquery in decomposed_dict.items():
            if subquery and subquery != "N/A":
                print(f"  • {field:8s}: {subquery}")

        # Check if we have any searchable fields (non-N/A)
        has_searchable_fields = any(v and v != "N/A" for v in decomposed_dict.values())
        constraints = decomposed.get_constraints()
        
        # Handle invalid/no-signal queries
        if not has_searchable_fields and not constraints:
            print("Invalid query: no searchable terms or constraints provided")
            return []

        # Retrieve documents using Lucene
        if has_searchable_fields:
            # Normal retrieval with search terms
            retrieve_k = top_k * 3  # Get 3x more results for filtering
            print(f"\nRetrieving top {retrieve_k} documents from Lucene...")
            results = self.ensemble_retriever.retrieve(decomposed_dict, retrieve_k)
        else:
            # No searchable fields - only constraints, so retrieve all documents
            print(f"\nNo searchable fields found, retrieving all documents for constraint filtering...")
            # Retrieve a large number to filter from
            results = self.ensemble_retriever.retrieve_all(top_k * 10)

        # Apply structural constraints as post-filters
        if constraints:
            print(f"\nApplying structural constraints: {constraints}")
            filtered_results = self._apply_constraints(results, constraints)
        else:
            filtered_results = results

        # Format results
        formatted_results = []
        for result in filtered_results[:top_k]:
            formatted_result = {
                "doc_id": result.doc_id,
                "score": result.score,
                "field_scores": result.field_scores,
                "metadata": result.metadata
            }
            formatted_results.append(formatted_result)

        print(f"\nFound {len(formatted_results)} results")
        return formatted_results
    
    def _apply_constraints(self, results: List, constraints: Dict[str, Dict[str, Any]]) -> List:
        """
        Apply structural constraints to filter results
        
        Args:
            results: List of RetrievalResult objects
            constraints: Dictionary mapping field names to constraint dictionaries
            
        Returns:
            Filtered list of results that match constraints
        """
        filtered = []
        
        for result in results:
            metadata = result.metadata if result.metadata else {}
            matches = True
            
            # Title constraints
            if 'title' in constraints:
                title = metadata.get('title', '')
                title_constraints = constraints['title']
                
                if 'word_count' in title_constraints:
                    word_count = len(title.split()) if title else 0
                    if word_count != title_constraints['word_count']:
                        matches = False
                
                if 'min_words' in title_constraints:
                    word_count = len(title.split()) if title else 0
                    if word_count <= title_constraints['min_words']:
                        matches = False
                
                if 'max_words' in title_constraints:
                    word_count = len(title.split()) if title else 0
                    if word_count >= title_constraints['max_words']:
                        matches = False
            
            # Author constraints
            if 'author' in constraints and matches:
                author = metadata.get('author', '')
                author_constraints = constraints['author']
                
                if 'starts_with' in author_constraints:
                    # Check if author name starts with the specified letter
                    if author:
                        # Extract first character (handle formats like "Last, First" or "First Last")
                        first_char = None
                        if ',' in author:
                            # "Last, First" format - get first char of last name
                            last_name = author.split(',')[0].strip()
                            first_char = last_name[0].upper() if last_name else None
                        else:
                            # "First Last" format - get first char of first word
                            first_word = author.split()[0] if author.split() else ''
                            first_char = first_word[0].upper() if first_word else None
                        
                        if first_char != author_constraints['starts_with']:
                            matches = False
                    else:
                        matches = False
                
                if 'name_count' in author_constraints:
                    # Count names in author field (split by spaces and commas)
                    names = [n.strip() for n in author.replace(',', ' ').split() if n.strip()]
                    if len(names) != author_constraints['name_count']:
                        matches = False
            
            # Date constraints
            if 'date' in constraints and matches:
                date_str = metadata.get('date', '')
                date_constraints = constraints['date']
                
                # Try to extract year from date string
                year = None
                if date_str and date_str != "N/A":
                    import re
                    year_match = re.search(r'\b(\d{4})\b', str(date_str))
                    if year_match:
                        year = int(year_match.group(1))
                
                if 'max_year' in date_constraints:
                    if year is None or year > date_constraints['max_year']:
                        matches = False
                
                if 'min_year' in date_constraints:
                    if year is None or year < date_constraints['min_year']:
                        matches = False
            
            if matches:
                filtered.append(result)
        
        return filtered

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
                       default=None, help="Operation mode (defaults to 'search' if --query provided)")
    parser.add_argument("--query", type=str, help="Query to search (if provided, defaults mode to 'search')")
    parser.add_argument("--data_file", type=str, default=None,
                       help="Data file to load (defaults to data/gutenberg_subset.json if it exists)")
    parser.add_argument("--index_dir", type=str, default=None,
                       help=f"Directory for Lucene indices (defaults to {Config.LUCENE_INDEX_DIR})")
    parser.add_argument("--rebuild_index", action="store_true",
                       help="Rebuild Lucene indices (default: load existing)")
    parser.add_argument("--decomposition_mode", choices=["extractive", "predictive"],
                       default="extractive", help="Query decomposition mode")

    args = parser.parse_args()
    
    # Auto-detect mode: if query is provided and mode is not specified, default to search
    if args.query and args.mode is None:
        args.mode = "search"
    elif args.mode is None:
        args.mode = "demo"  # Default to demo if nothing specified

    # Initialize system
    system = TOTRetrievalSystem(index_dir=args.index_dir)

    if args.mode == "demo":
        # Demo mode using Gutenberg data
        print("🎭 Running demo with Gutenberg dataset...")

        # Full path to Gutenberg dataset
        gutenberg_file = str(Config.DATA_DIR / "gutenberg_subset.json")

        # Load Gutenberg dataset
        print(f"Loading Gutenberg dataset from {gutenberg_file}...")
        documents = system.data_loader.load_dataset(gutenberg_file)

        # Setup system
        system.setup(documents, rebuild_index=True)

        # Demo queries
        demo_queries = [
            "A story with a mysterious wealthy man",
            "A lawyer defends someone falsely accused",
            "A dystopian society under surveillance",
            "A romance set in classic society",
            "A coming-of-age narrative"
        ]

        for query in demo_queries:
            results = system.search(query, mode=args.decomposition_mode, top_k=3)
            print("\nTop Results:")
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                print(f"\n  {i}. {metadata.get('title', 'Unknown')}")
                print(f"     Author: {metadata.get('author', 'Unknown')}")
                print(f"     Score: {result['score']:.4f}")

                # Show top contributing fields (filter out 0.00 scores)
                field_scores = result['field_scores']
                sorted_fields = sorted(field_scores.items(), key=lambda x: x[1], reverse=True)
                # Filter out fields with zero or near-zero scores
                non_zero_fields = [(k, v) for k, v in sorted_fields if v > 0.001]
                if non_zero_fields:
                    top_fields_str = ', '.join([f'{k}({v:.2f})' for k, v in non_zero_fields[:3]])
                    print(f"     Top fields: {top_fields_str}")

    elif args.mode == "search":
        # Search mode: can use existing indices or load data file
        if args.query is None:
            print("❌ Error: --query required for search mode")
            return

        # Check if indices exist (use system's index_dir which has the default if None was provided)
        index_dir = system.index_dir
        index_exists = any(os.path.exists(os.path.join(index_dir, f"{field}_index")) 
                           for field in ['plot', 'title', 'author', 'genre', 'date', 'cover'])
        
        # Default to Gutenberg dataset if no data file specified
        if not args.data_file:
            default_data_file = str(Config.DATA_DIR / "gutenberg_subset.json")
            if os.path.exists(default_data_file):
                args.data_file = default_data_file
                print(f"Using default data file: {default_data_file}")
        
        if args.data_file:
            # Load data file and setup (use existing indices by default, rebuild only if requested)
            print(f"Loading data from {args.data_file}...")
            documents = system.data_loader.load_dataset(args.data_file)
            # Only rebuild if explicitly requested, otherwise use existing indices
            rebuild = args.rebuild_index
            system.setup(documents, rebuild_index=rebuild)
        elif index_exists:
            # Use existing indices without data file (fallback)
            print("Using existing Lucene indices...")
            print("   Note: Without --data_file, metadata (title/author) will be limited")
            system.setup_from_indices_only()
        else:
            print("❌ Error: No existing indices found and no --data_file provided")
            print("   Please provide --data_file to build indices, or ensure indices exist in", index_dir)
            return

        # Perform search
        results = system.search(args.query, mode=args.decomposition_mode, top_k=10)
        print("\nSearch Results:")
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            title = metadata.get('title', 'Unknown') if metadata else 'Unknown'
            author = metadata.get('author', 'Unknown') if metadata else 'Unknown'
            print(f"\n{i}. {title} by {author}")
            print(f"   Doc ID: {result['doc_id']}")
            print(f"   Score: {result['score']:.4f}")
            
            # Show top contributing fields (filter out 0.00 scores)
            if result.get('field_scores'):
                field_scores = result['field_scores']
                sorted_fields = sorted(field_scores.items(), key=lambda x: x[1], reverse=True)
                # Filter out fields with zero or near-zero scores
                non_zero_fields = [(k, v) for k, v in sorted_fields if v > 0.001]
                if non_zero_fields:
                    top_fields_str = ', '.join([f'{k}({v:.2f})' for k, v in non_zero_fields[:3]])
                    print(f"   Top fields: {top_fields_str}")

    elif args.mode == "evaluate":
        # For evaluate, require --data_file
        if not args.data_file:
            print("❌ Error: --data_file required for this mode")
            return

        print(f"Loading data from {args.data_file}...")
        documents = system.data_loader.load_dataset(args.data_file)

        # Setup system
        system.setup(documents, rebuild_index=args.rebuild_index)

        # Load test queries
        test_queries_file = args.data_file.replace('.json', '_queries.json')
        if os.path.exists(test_queries_file):
            test_queries, test_labels = system.data_loader.load_queries_and_labels(test_queries_file)
        else:
            print(f"Warning: Test queries file not found: {test_queries_file}")
            print("Creating sample queries from documents...")
            query_data = system.data_loader.create_sample_queries(documents, num_queries=10)
            test_queries = [q['query'] for q in query_data]
            test_labels = [q['doc_id'] for q in query_data]

        results = system.evaluate(test_queries, test_labels, args.decomposition_mode)
        print("\n" + results["report"])

    elif args.mode == "optimize":
        # For optimize, require --data_file
        if not args.data_file:
            print("❌ Error: --data_file required for this mode")
            return

        print(f"Loading data from {args.data_file}...")
        documents = system.data_loader.load_dataset(args.data_file)

        # Setup system
        system.setup(documents, rebuild_index=args.rebuild_index)

        val_queries_file = args.data_file.replace('.json', '_val_queries.json')
        if os.path.exists(val_queries_file):
            val_queries, val_labels = system.data_loader.load_queries_and_labels(val_queries_file)
        else:
            print(f"Warning: Validation queries file not found: {val_queries_file}")
            print("Creating sample queries from documents...")
            query_data = system.data_loader.create_sample_queries(documents, num_queries=10)
            val_queries = [q['query'] for q in query_data]
            val_labels = [q['doc_id'] for q in query_data]

        optimized_weights = system.optimize_weights(val_queries, val_labels, args.decomposition_mode)
        print(f"Optimized weights: {optimized_weights}")

if __name__ == "__main__":
    main()
