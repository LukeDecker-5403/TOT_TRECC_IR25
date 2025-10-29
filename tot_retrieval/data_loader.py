"""
Data loading utilities for TOT Retrieval System
"""

import json
import pandas as pd
from typing import List, Dict, Any, Tuple
from pathlib import Path
import random

class DataLoader:
    """Loads and processes datasets for TOT retrieval"""

    def __init__(self, data_dir: str = "./data"):
        """
        Initialize data loader

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load dataset from file

        Args:
            filepath: Path to data file (JSON, JSONL, or CSV)

        Returns:
            List of document dictionaries
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # Load based on file extension
        if filepath.suffix == '.json':
            return self._load_json(filepath)
        elif filepath.suffix == '.jsonl':
            return self._load_jsonl(filepath)
        elif filepath.suffix == '.csv':
            return self._load_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def _load_json(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            return self._normalize_documents(data)
        elif isinstance(data, dict) and 'documents' in data:
            return self._normalize_documents(data['documents'])
        else:
            raise ValueError("JSON must be a list or dict with 'documents' key")

    def _load_jsonl(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load JSONL file (one JSON object per line)"""
        documents = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    documents.append(json.loads(line))
        return self._normalize_documents(documents)

    def _load_csv(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load CSV file"""
        df = pd.read_csv(filepath)
        documents = df.to_dict('records')
        return self._normalize_documents(documents)

    def _normalize_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize document format to expected schema

        Expected fields: doc_id, plot, title, author, genre, date, cover
        """
        normalized = []

        for i, doc in enumerate(documents):
            # Create doc_id if not present
            if 'doc_id' not in doc and 'id' not in doc:
                doc_id = f"doc_{i}"
            else:
                doc_id = str(doc.get('doc_id', doc.get('id', f"doc_{i}")))

            # Special handling for Gutenberg dataset
            if 'TEXT' in doc and 'METADATA' in doc:
                try:
                    metadata = json.loads(doc['METADATA'])
                except json.JSONDecodeError:
                    metadata = {}

                normalized_doc = {
                    'doc_id': doc_id,
                    'plot': doc.get('TEXT', ""),
                    'title': metadata.get('title', "N/A"),
                    'author': metadata.get('authors', "N/A"),
                    'genre': metadata.get('subjects', "N/A"),
                    'date': metadata.get('issued', "N/A"),
                    'cover': ""  # Gutenberg doesn't have cover info
                }
            else:
                # Default behavior for other datasets
                normalized_doc = {
                    'doc_id': doc_id,
                    'plot': self._get_field(doc, ['plot', 'summary', 'description', 'synopsis', 'text']),
                    'title': self._get_field(doc, ['title', 'name']),
                    'author': self._get_field(doc, ['author', 'writer', 'creator']),
                    'genre': self._get_field(doc, ['genre', 'category', 'type']),
                    'date': self._get_field(doc, ['date', 'year', 'publication_year', 'published']),
                    'cover': self._get_field(doc, ['cover', 'cover_description', 'visual'])
                }

            normalized.append(normalized_doc)

        return normalized

    def _get_field(self, doc: Dict[str, Any], field_names: List[str]) -> str:
        """Get field value with fallback to alternative field names"""
        for field_name in field_names:
            if field_name in doc and doc[field_name]:
                return str(doc[field_name])
        return ""

    def create_synthetic_dataset(self, num_books: int = 50) -> List[Dict[str, Any]]:
        """
        Create synthetic book dataset for testing

        Args:
            num_books: Number of synthetic books to generate

        Returns:
            List of synthetic book documents
        """
        # [unchanged from original code]
        ...

    def split_dataset(self, documents: List[Dict[str, Any]],
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """[unchanged from original code]"""
        ...

    def save_dataset(self, documents: List[Dict[str, Any]], filepath: str):
        """[unchanged from original code]"""
        ...

    def load_queries_and_labels(self, filepath: str) -> Tuple[List[str], List[str]]:
        """[unchanged from original code]"""
        ...

    def create_sample_queries(self, documents: List[Dict[str, Any]],
                            num_queries: int = 10) -> List[Dict[str, str]]:
        """[unchanged from original code]"""
        ...
