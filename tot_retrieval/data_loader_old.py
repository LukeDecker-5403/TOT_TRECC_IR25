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
        synthetic_books = [
            {
                'doc_id': 'book_0',
                'title': 'The Great Gatsby',
                'author': 'F. Scott Fitzgerald',
                'genre': 'Classic Literature',
                'date': '1925',
                'plot': 'A wealthy mysterious man named Jay Gatsby throws lavish parties in hopes of reuniting with his lost love Daisy Buchanan. Set in the Jazz Age.',
                'cover': 'Art deco design with eyes watching over a city'
            },
            {
                'doc_id': 'book_1',
                'title': 'To Kill a Mockingbird',
                'author': 'Harper Lee',
                'genre': 'Classic Literature',
                'date': '1960',
                'plot': 'A lawyer named Atticus Finch defends a black man falsely accused of rape in the Deep South, told through the eyes of his daughter Scout.',
                'cover': 'A tree with a knothole'
            },
            {
                'doc_id': 'book_2',
                'title': '1984',
                'author': 'George Orwell',
                'genre': 'Dystopian Fiction',
                'date': '1949',
                'plot': 'Winston Smith lives in a totalitarian society under constant surveillance by Big Brother. He rebels against the Party and falls in love.',
                'cover': 'An eye watching, dark colors'
            },
            {
                'doc_id': 'book_3',
                'title': 'Pride and Prejudice',
                'author': 'Jane Austen',
                'genre': 'Romance',
                'date': '1813',
                'plot': 'Elizabeth Bennet navigates society and relationships in Georgian England, initially disliking the proud Mr. Darcy before falling in love.',
                'cover': 'Regency era couple in formal dress'
            },
            {
                'doc_id': 'book_4',
                'title': 'The Catcher in the Rye',
                'author': 'J.D. Salinger',
                'genre': 'Coming-of-age',
                'date': '1951',
                'plot': 'Teenager Holden Caulfield is expelled from prep school and wanders New York City, struggling with alienation and teenage angst.',
                'cover': 'Red hunting cap, simple design'
            },
            {
                'doc_id': 'book_5',
                'title': 'Harry Potter and the Sorcerer\'s Stone',
                'author': 'J.K. Rowling',
                'genre': 'Fantasy',
                'date': '1997',
                'plot': 'An orphaned boy discovers he is a wizard and attends Hogwarts School of Witchcraft and Wizardry, where he makes friends and battles evil.',
                'cover': 'Boy with glasses, lightning bolt scar, wizard hat'
            }
        ]
        
        # Extend with variations if needed
        books = synthetic_books[:min(num_books, len(synthetic_books))]
        
        # If more books needed, create variations
        while len(books) < num_books:
            base_book = random.choice(synthetic_books)
            variation = base_book.copy()
            variation['doc_id'] = f"book_{len(books)}"
            books.append(variation)
        
        return books
    
    def split_dataset(self, documents: List[Dict[str, Any]], 
                     train_ratio: float = 0.7, 
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split dataset into train/validation/test sets
        
        Args:
            documents: List of documents
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            seed: Random seed
            
        Returns:
            Tuple of (train_docs, val_docs, test_docs)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        random.seed(seed)
        shuffled = documents.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_docs = shuffled[:train_end]
        val_docs = shuffled[train_end:val_end]
        test_docs = shuffled[val_end:]
        
        return train_docs, val_docs, test_docs
    
    def save_dataset(self, documents: List[Dict[str, Any]], filepath: str):
        """Save documents to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(documents)} documents to {filepath}")
    
    def load_queries_and_labels(self, filepath: str) -> Tuple[List[str], List[str]]:
        """
        Load queries and ground truth labels
        
        Expected format: JSON with list of {"query": "...", "doc_id": "..."}
        
        Args:
            filepath: Path to queries file
            
        Returns:
            Tuple of (queries, labels)
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        queries = [item['query'] for item in data]
        labels = [item['doc_id'] for item in data]
        
        return queries, labels
    
    def create_sample_queries(self, documents: List[Dict[str, Any]], 
                            num_queries: int = 10) -> List[Dict[str, str]]:
        """
        Create sample TOT-style queries from documents
        
        Args:
            documents: List of documents
            num_queries: Number of queries to generate
            
        Returns:
            List of query dictionaries with query and doc_id
        """
        queries = []
        
        # Templates for vague queries
        templates = [
            "Something about {plot_fragment}",
            "A book where {plot_fragment}",
            "I think it was by {author} or someone similar",
            "A {genre} book about {plot_fragment}",
            "Published around {date}, about {plot_fragment}",
            "The title had something like {title_word}",
            "The cover showed {cover_fragment}"
        ]
        
        for i in range(min(num_queries, len(documents))):
            doc = documents[i]
            
            # Extract fragments
            plot_words = doc['plot'].split()[:10]
            plot_fragment = ' '.join(plot_words) if plot_words else "something"
            title_word = doc['title'].split()[0] if doc['title'] else "something"
            
            # Pick random template
            template = random.choice(templates)
            
            query = template.format(
                plot_fragment=plot_fragment,
                author=doc.get('author', 'unknown'),
                genre=doc.get('genre', 'fiction'),
                date=doc.get('date', 'recently'),
                title_word=title_word,
                cover_fragment=doc.get('cover', 'something')
            )
            
            queries.append({
                'query': query,
                'doc_id': doc['doc_id']
            })
        
        return queries
