"""
Configuration module for TOT Retrieval System with Lucene/Pyserini support
"""

import os
from pathlib import Path

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not available, will use environment variables only

class Config:
    """Configuration settings for TOT Retrieval System"""
    
    # ============================================================
    # LLM Settings (Optional - only if using OpenAI decomposer)
    # ============================================================
    LLM_MODEL = "gpt-4"  # or "gpt-4" for better results
    # Try multiple environment variable names for API key
    # Check both uppercase and lowercase variants (common in .env files)
    OPENAI_API_KEY = (
        os.getenv("OPENAI_API_KEY") or 
        os.getenv("openai_api_key") or 
        os.getenv("openapi_api_key") or 
        os.getenv("OPENAPI_API_KEY")
    )
    
    # ============================================================
    # Directory Settings
    # ============================================================
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    CACHE_DIR = BASE_DIR / "cache"
    RESULTS_DIR = BASE_DIR / "results"
    LUCENE_INDEX_DIR = BASE_DIR / "lucene_indices"
    
    # ============================================================
    # Retrieval Settings - BM25 Parameters
    # ============================================================
    DEFAULT_TOP_K = 10
    BM25_K1 = 1.5  # BM25 term frequency saturation parameter
    BM25_B = 0.75  # BM25 length normalization parameter
    
    # ============================================================
    # Ensemble Settings - Field Weights
    # ============================================================
    # Default weights for each metadata field
    # These can be optimized using validation data
    DEFAULT_WEIGHTS = {
        "plot": 0.35,    # Story/content usually most informative
        "title": 0.25,   # Title is very specific when available
        "author": 0.15,  # Author helps narrow down significantly
        "genre": 0.10,   # Genre provides some signal
        "date": 0.05,    # Date/year less discriminative
        "cover": 0.10    # Cover description can be helpful
    }
    
    # ============================================================
    # Lucene/Pyserini Specific Settings
    # ============================================================
    # Maximum number of documents to retrieve per field before ensemble
    LUCENE_MAX_HITS_PER_FIELD = 1000
    
    # Lucene analyzer settings
    LUCENE_ANALYZER = "english"  # Options: "standard", "english", "whitespace"
    
    # Whether to use positional indexing (slower but supports phrase queries)
    LUCENE_POSITIONAL_INDEX = False
    
    # Lucene index settings
    LUCENE_RAM_BUFFER_SIZE = 2048  # MB - increase for faster indexing
    
    # ============================================================
    # Query Decomposition Settings
    # ============================================================
    # Which decomposer to use by default
    # Options: "rule_based" (free, no API), "llm" (requires OpenAI key)
    DEFAULT_DECOMPOSER = "llm"
    
    # Cache decomposed queries to avoid re-processing
    ENABLE_QUERY_CACHE = True
    QUERY_CACHE_FILE = CACHE_DIR / "decomposed_queries_cache.json"
    
    # ============================================================
    # Evaluation Settings
    # ============================================================
    METRICS = ["recall@1", "recall@5", "recall@10", "mrr", "precision@5", "precision@10"]
    
    # Default K values for evaluation
    EVAL_K_VALUES = [1, 5, 10, 20]
    
    # ============================================================
    # Data Processing Settings
    # ============================================================
    MAX_QUERY_LENGTH = 500
    MAX_DOCUMENT_LENGTH = 5000
    
    # Field name mappings for automatic normalization
    FIELD_ALIASES = {
        'plot': ['plot', 'summary', 'description', 'synopsis', 'text', 'content'],
        'title': ['title', 'name', 'heading'],
        'author': ['author', 'writer', 'creator', 'by'],
        'genre': ['genre', 'category', 'type', 'class'],
        'date': ['date', 'year', 'publication_year', 'published', 'pub_date'],
        'cover': ['cover', 'cover_description', 'visual', 'cover_art']
    }
    
    # ============================================================
    # Weight Optimization Settings
    # ============================================================
    # Number of random weight combinations to try during optimization
    WEIGHT_OPTIMIZATION_TRIALS = 50
    
    # Metric to optimize (options: "recall@5", "recall@10", "mrr")
    WEIGHT_OPTIMIZATION_METRIC = "recall@5"
    
    # ============================================================
    # Logging and Debugging
    # ============================================================
    VERBOSE = True  # Print detailed progress messages
    DEBUG = False   # Print debug information
    
    # ============================================================
    # Methods
    # ============================================================
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LUCENE_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        
        if cls.VERBOSE:
            print("Created/verified directories:")
            print(f"   Data: {cls.DATA_DIR}")
            print(f"   Cache: {cls.CACHE_DIR}")
            print(f"   Results: {cls.RESULTS_DIR}")
            print(f"   Lucene Indices: {cls.LUCENE_INDEX_DIR}")
    
    @classmethod
    def validate(cls, require_openai: bool = False):
        """
        Validate configuration
        
        Args:
            require_openai: If True, require OpenAI API key to be set
        """
        # Create directories
        cls.create_directories()
        
        # Check OpenAI key only if required
        if require_openai and not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not set. Please set it as an environment variable:\n"
                "export OPENAI_API_KEY='your-key-here'\n\n"
                "Or use the rule-based decomposer (no API key needed):\n"
                "Set Config.DEFAULT_DECOMPOSER = 'rule_based'"
            )
        
        # Validate Java for Lucene (required)
        if not cls._check_java():
            raise RuntimeError(
                "Java 11+ is required for Lucene/Pyserini but was not found.\n"
                "Install Java:\n"
                "  Ubuntu/Debian: sudo apt-get install openjdk-11-jdk\n"
                "  macOS: brew install openjdk@11\n"
                "  Windows: Download from https://adoptium.net/"
            )
        
        if cls.VERBOSE:
            print("Configuration validated successfully")
        
        return True
    
    @classmethod
    def _check_java(cls):
        """Check if Java is installed and accessible"""
        import subprocess
        try:
            result = subprocess.run(
                ['java', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    @classmethod
    def get_summary(cls):
        """Get a summary of current configuration"""
        summary = []
        summary.append("=" * 60)
        summary.append("TOT Retrieval System Configuration")
        summary.append("=" * 60)
        summary.append("")
        summary.append("Directories:")
        summary.append(f"  Data:          {cls.DATA_DIR}")
        summary.append(f"  Cache:         {cls.CACHE_DIR}")
        summary.append(f"  Lucene Index:  {cls.LUCENE_INDEX_DIR}")
        summary.append("")
        summary.append("BM25 Parameters:")
        summary.append(f"  k1: {cls.BM25_K1}")
        summary.append(f"  b:  {cls.BM25_B}")
        summary.append("")
        summary.append("Ensemble Weights:")
        for field, weight in cls.DEFAULT_WEIGHTS.items():
            summary.append(f"  {field:8s}: {weight:.2f}")
        summary.append("")
        summary.append("Query Decomposer:")
        summary.append(f"  Type: {cls.DEFAULT_DECOMPOSER}")
        summary.append(f"  Cache: {'Enabled' if cls.ENABLE_QUERY_CACHE else 'Disabled'}")
        summary.append("")
        summary.append("=" * 60)
        
        return "\n".join(summary)
    
    @classmethod
    def save_to_file(cls, filepath: str = None):
        """Save current configuration to a file"""
        if filepath is None:
            filepath = cls.RESULTS_DIR / "config_used.txt"
        
        with open(filepath, 'w') as f:
            f.write(cls.get_summary())
        
        if cls.VERBOSE:
            print(f"Configuration saved to: {filepath}")
    
    @classmethod
    def update_weights(cls, new_weights: dict):
        """Update ensemble weights"""
        cls.DEFAULT_WEIGHTS.update(new_weights)
        if cls.VERBOSE:
            print("Updated ensemble weights:")
            for field, weight in cls.DEFAULT_WEIGHTS.items():
                print(f"   {field:8s}: {weight:.4f}")

# ============================================================
# Auto-validate on import (with warnings, not errors)
# ============================================================
try:
    Config.create_directories()
    if not Config._check_java():
        print("WARNING: Java not found. Lucene/Pyserini will not work.")
        print("   Install Java 11+ to use this system.")
except Exception as e:
    print(f"WARNING: Configuration validation issue: {e}")
