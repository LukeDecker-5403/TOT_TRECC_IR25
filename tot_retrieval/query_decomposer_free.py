"""
Rule-Based Query Decomposer (NO API KEY REQUIRED)
Uses pattern matching and heuristics instead of LLM
"""

import re
from typing import Dict, List
from dataclasses import dataclass
import json
import os

@dataclass
class DecomposedQuery:
    """Container for decomposed query results"""
    plot: str
    title: str
    author: str
    genre: str
    date: str
    cover: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format"""
        return {
            "plot": self.plot,
            "title": self.title,
            "author": self.author,
            "genre": self.genre,
            "date": self.date,
            "cover": self.cover
        }

class RuleBasedQueryDecomposer:
    """Rule-based query decomposer (no LLM required)"""
    
    def __init__(self):
        """Initialize the rule-based decomposer"""
        # Common author indicators
        self.author_patterns = [
            r'by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'author\s+(?:is\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'written by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        
        # Common title indicators
        self.title_patterns = [
            r'(?:book|novel|story)\s+(?:called|titled|named)\s+"([^"]+)"',
            r'"([^"]+)"',  # Anything in quotes
            r'title\s+(?:is\s+)?([A-Z][^,.!?]*)',
        ]
        
        # Date patterns
        self.date_patterns = [
            r'\b(19\d{2}|20\d{2})\b',  # Years
            r'in\s+the\s+(19\d{2}s|20\d{2}s)',  # Decades
            r'(?:published|written|came out)\s+(?:in\s+)?(\d{4})',
        ]
        
        # Genre keywords
        self.genre_keywords = {
            'romance': ['love', 'romance', 'romantic', 'relationship'],
            'mystery': ['mystery', 'detective', 'murder', 'crime', 'whodunit'],
            'fantasy': ['fantasy', 'magic', 'wizard', 'dragon', 'elf'],
            'sci-fi': ['science fiction', 'sci-fi', 'space', 'alien', 'future'],
            'horror': ['horror', 'scary', 'terror', 'haunted', 'ghost'],
            'thriller': ['thriller', 'suspense', 'tension'],
            'dystopian': ['dystopian', 'post-apocalyptic', 'totalitarian'],
            'historical': ['historical', 'period', 'war', 'ancient'],
            'biography': ['biography', 'memoir', 'life story'],
            'classic': ['classic', 'classical']
        }
        
        # Cover description keywords
        self.cover_keywords = [
            'cover', 'jacket', 'front', 'picture', 'image', 
            'illustration', 'artwork', 'design'
        ]
        
        # Cache
        self.cache = {}
        self.cache_file = "cache/decomposed_queries_rulebased.json"
        self._load_cache()
    
    def _load_cache(self):
        """Load cached decompositions"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except:
                self.cache = {}
    
    def _save_cache(self):
        """Save cache to file"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def decompose(self, query: str, mode: str = "extractive") -> DecomposedQuery:
        """
        Decompose query using rule-based approach
        
        Args:
            query: The original query
            mode: "extractive" or "predictive" (both use same rules here)
            
        Returns:
            DecomposedQuery object
        """
        # Check cache
        cache_key = f"{query}_{mode}"
        if cache_key in self.cache:
            return DecomposedQuery(**self.cache[cache_key])
        
        # Extract components
        author = self._extract_author(query)
        title = self._extract_title(query)
        date = self._extract_date(query)
        genre = self._extract_genre(query)
        cover = self._extract_cover(query)
        plot = self._extract_plot(query, author, title, date, genre, cover)
        
        decomposed = DecomposedQuery(
            plot=plot,
            title=title,
            author=author,
            genre=genre,
            date=date,
            cover=cover
        )
        
        # Cache result
        self.cache[cache_key] = decomposed.to_dict()
        self._save_cache()
        
        return decomposed
    
    def _extract_author(self, query: str) -> str:
        """Extract author name from query"""
        for pattern in self.author_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        return "N/A"
    
    def _extract_title(self, query: str) -> str:
        """Extract title from query"""
        for pattern in self.title_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        return "N/A"
    
    def _extract_date(self, query: str) -> str:
        """Extract publication date from query"""
        for pattern in self.date_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        return "N/A"
    
    def _extract_genre(self, query: str) -> str:
        """Extract genre from query"""
        query_lower = query.lower()
        
        # Check for genre keywords
        for genre, keywords in self.genre_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return genre
        
        return "N/A"
    
    def _extract_cover(self, query: str) -> str:
        """Extract cover description from query"""
        query_lower = query.lower()
        
        # Find sentences mentioning cover
        for keyword in self.cover_keywords:
            if keyword in query_lower:
                # Extract sentence containing the keyword
                sentences = re.split(r'[.!?]', query)
                for sentence in sentences:
                    if keyword in sentence.lower():
                        return sentence.strip()
        
        return "N/A"
    
    def _extract_plot(self, query: str, author: str, title: str, 
                     date: str, genre: str, cover: str) -> str:
        """Extract plot elements (everything not in other fields)"""
        # Remove already extracted information
        plot = query
        
        # Remove author mentions
        if author != "N/A":
            plot = re.sub(rf'\bby\s+{re.escape(author)}\b', '', plot, flags=re.IGNORECASE)
            plot = re.sub(rf'\bauthor\s+(?:is\s+)?{re.escape(author)}\b', '', plot, flags=re.IGNORECASE)
        
        # Remove title mentions
        if title != "N/A":
            plot = re.sub(rf'"{re.escape(title)}"', '', plot, flags=re.IGNORECASE)
            plot = re.sub(rf'\btitled\s+{re.escape(title)}\b', '', plot, flags=re.IGNORECASE)
        
        # Remove date mentions
        if date != "N/A":
            plot = re.sub(rf'\b{re.escape(date)}\b', '', plot)
        
        # Remove genre mentions
        if genre != "N/A":
            plot = re.sub(rf'\b{re.escape(genre)}\b', '', plot, flags=re.IGNORECASE)
        
        # Remove cover descriptions
        if cover != "N/A":
            plot = plot.replace(cover, '')
        
        # Clean up
        plot = re.sub(r'\s+', ' ', plot).strip()
        
        return plot if plot else query  # Return original if nothing left
    
    def batch_decompose(self, queries: List[str], mode: str = "extractive") -> List[DecomposedQuery]:
        """Decompose multiple queries"""
        return [self.decompose(q, mode) for q in queries]
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cached_queries": len(self.cache),
            "cache_file": self.cache_file
        }

# Alias for backward compatibility
QueryDecomposer = RuleBasedQueryDecomposer
