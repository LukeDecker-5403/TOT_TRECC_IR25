"""
Query Decomposer for TOT Retrieval System
Breaks down complex queries into field-specific subqueries using LLM
"""

import json
import openai
from typing import Dict, List, Optional
from dataclasses import dataclass
import os
from .config import Config

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

class QueryDecomposer:
    """LLM-based query decomposer for TOT retrieval"""
    
    def __init__(self, model: str = None, api_key: str = None):
        """
        Initialize the query decomposer
        
        Args:
            model: LLM model to use (default: from config)
            api_key: OpenAI API key (default: from environment)
        """
        self.model = model or Config.LLM_MODEL
        self.api_key = api_key or Config.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.api_key
        
        # Cache for storing decomposed queries
        self.cache = {}
        self.cache_file = os.path.join(Config.CACHE_DIR, "decomposed_queries.json")
        self._load_cache()
    
    def _load_cache(self):
        """Load decomposed queries from cache"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self.cache = {}
    
    def _save_cache(self):
        """Save decomposed queries to cache"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def decompose(self, query: str, mode: str = "extractive") -> DecomposedQuery:
        """
        Decompose a complex query into field-specific subqueries
        
        Args:
            query: The original complex query
            mode: "extractive" or "predictive"
            
        Returns:
            DecomposedQuery object with field-specific subqueries
        """
        # Check cache first
        cache_key = f"{query}_{mode}"
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            return DecomposedQuery(**cached_result)
        
        # Generate decomposition using LLM
        if mode == "extractive":
            decomposed = self._extractive_decomposition(query)
        elif mode == "predictive":
            decomposed = self._predictive_decomposition(query)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Cache the result
        self.cache[cache_key] = decomposed.to_dict()
        self._save_cache()
        
        return decomposed
    
    def _extractive_decomposition(self, query: str) -> DecomposedQuery:
        """Extract clues directly from the query text"""
        
        system_prompt = """You are an expert at analyzing vague book queries and extracting specific information for each metadata field.

For each field, extract relevant information from the query. If no information is available for a field, return "N/A".

Fields to extract:
- plot: Story elements, plot points, character names, settings
- title: Any title words, phrases, or partial titles mentioned
- author: Author names, initials, or any author-related clues
- genre: Book genre, category, or type
- date: Publication year, decade, or time period
- cover: Visual descriptions of the book cover

Return your response as a JSON object with these exact keys: plot, title, author, genre, date, cover"""

        user_prompt = f"Analyze this book query and extract information for each field:\n\nQuery: {query}"
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                result_dict = self._parse_fallback(result_text)
            
            return DecomposedQuery(
                plot=result_dict.get("plot", "N/A"),
                title=result_dict.get("title", "N/A"),
                author=result_dict.get("author", "N/A"),
                genre=result_dict.get("genre", "N/A"),
                date=result_dict.get("date", "N/A"),
                cover=result_dict.get("cover", "N/A")
            )
            
        except Exception as e:
            print(f"Error in extractive decomposition: {e}")
            return DecomposedQuery(
                plot="N/A", title="N/A", author="N/A", 
                genre="N/A", date="N/A", cover="N/A"
            )
    
    def _predictive_decomposition(self, query: str) -> DecomposedQuery:
        """Predict missing metadata based on context clues"""
        
        system_prompt = """You are an expert at analyzing vague book queries and predicting missing metadata based on context clues.

For each field, predict or infer information from the query. Use your knowledge to fill in missing details.

Fields to predict:
- plot: Story elements, plot points, character names, settings
- title: Likely title words or phrases based on plot description
- author: Likely author based on genre, time period, or writing style clues
- genre: Most likely genre based on plot elements
- date: Estimated publication year based on context (e.g., "read in high school" → year ≤ 2005)
- cover: Likely visual elements based on genre and plot

Return your response as a JSON object with these exact keys: plot, title, author, genre, date, cover"""

        user_prompt = f"Analyze this book query and predict missing metadata:\n\nQuery: {query}"
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Higher temperature for more creative predictions
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                result_dict = self._parse_fallback(result_text)
            
            return DecomposedQuery(
                plot=result_dict.get("plot", "N/A"),
                title=result_dict.get("title", "N/A"),
                author=result_dict.get("author", "N/A"),
                genre=result_dict.get("genre", "N/A"),
                date=result_dict.get("date", "N/A"),
                cover=result_dict.get("cover", "N/A")
            )
            
        except Exception as e:
            print(f"Error in predictive decomposition: {e}")
            return DecomposedQuery(
                plot="N/A", title="N/A", author="N/A", 
                genre="N/A", date="N/A", cover="N/A"
            )
    
    def _parse_fallback(self, text: str) -> Dict[str, str]:
        """Fallback parser for non-JSON responses"""
        result = {"plot": "N/A", "title": "N/A", "author": "N/A", 
                 "genre": "N/A", "date": "N/A", "cover": "N/A"}
        
        # Simple keyword-based extraction
        lines = text.split('\n')
        for line in lines:
            line = line.strip().lower()
            if 'plot' in line or 'story' in line:
                result['plot'] = line
            elif 'title' in line:
                result['title'] = line
            elif 'author' in line:
                result['author'] = line
            elif 'genre' in line:
                result['genre'] = line
            elif 'date' in line or 'year' in line:
                result['date'] = line
            elif 'cover' in line:
                result['cover'] = line
        
        return result
    
    def batch_decompose(self, queries: List[str], mode: str = "extractive") -> List[DecomposedQuery]:
        """Decompose multiple queries in batch"""
        results = []
        for query in queries:
            results.append(self.decompose(query, mode))
        return results
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cached_queries": len(self.cache),
            "cache_file": self.cache_file
        }
