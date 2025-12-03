"""
Query Decomposer for TOT Retrieval System
Breaks down complex queries into field-specific subqueries using LLM
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import os
from .config import Config

# Try to import OpenAI - support both old and new API
try:
    from openai import OpenAI
    NEW_API = True
except ImportError:
    try:
        import openai
        NEW_API = False
    except ImportError:
        raise ImportError("OpenAI package not installed. Install with: pip install openai")

@dataclass
class DecomposedQuery:
    """Container for decomposed query results"""
    plot: str
    title: str
    author: str
    genre: str
    date: str
    cover: str
    # Structural constraints (not searchable, but used for filtering)
    _constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
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
    
    def get_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Get structural constraints for post-filtering"""
        return self._constraints
    
    def set_constraint(self, field: str, constraint_type: str, value: Any):
        """Set a structural constraint (e.g., title word_count = 3)"""
        if field not in self._constraints:
            self._constraints[field] = {}
        self._constraints[field][constraint_type] = value

class QueryDecomposer:
    """LLM-based query decomposer for TOT retrieval"""
    
    def __init__(self, model: str = None, api_key: str = None):
        """
        Initialize the query decomposer
        
        Args:
            model: LLM model to use (default: from config)
            api_key: OpenAI API key (default: from environment or .env file)
        """
        self.model = model or Config.LLM_MODEL
        self.api_key = api_key or Config.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or add openapi_api_key to .env file."
            )
        
        # Initialize OpenAI client (support both old and new API)
        if NEW_API:
            self.client = OpenAI(api_key=self.api_key)
        else:
            openai.api_key = self.api_key
            self.client = None
        
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
        try:
            if mode == "extractive":
                decomposed = self._extractive_decomposition(query)
            elif mode == "predictive":
                decomposed = self._predictive_decomposition(query)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # Cache the result (excluding constraints for now)
            self.cache[cache_key] = decomposed.to_dict()
            self._save_cache()
            
            return decomposed
        except Exception as e:
            # Don't catch and suppress - let caller handle fallback
            raise
    
    def _extractive_decomposition(self, query: str) -> DecomposedQuery:
        """Extract clues directly from the query text"""
        
        system_prompt = """You are an expert at analyzing vague book queries and extracting specific SEARCHABLE terms for each metadata field.

IMPORTANT: Extract ACTUAL SEARCHABLE TEXT, not descriptions of constraints. For structural constraints (like "title with 3 words" or "author starts with B"), DO NOT put the constraint description in the field. If there's no actual searchable text for a field (only constraints), put "N/A". The extracted text will be used for keyword/full-text search, so it must be actual words or phrases that might appear in documents.

For each field:
- plot: Extract actual story content, plot elements, character names, or settings mentioned (searchable text)
- title: Extract actual title words, phrases, or partial titles (NOT descriptions like "3 words long")
- author: Extract ACTUAL author names, surnames, or initials (NOT descriptions like "starts with B"). If only an initial/letter is mentioned (e.g., "starts with B" or "name begins with B"), extract just that letter (e.g., "B") - but note that single letter searches may not work well.
- genre: Extract the actual genre name (e.g., "fantasy", "romance", "mystery")
- date: Extract actual years, decades, or specific dates (e.g., "2015", "1990s", NOT descriptions like "before 2015"). For "before X" extract just the year (e.g., "2015"), for "after X" extract the year, for ranges extract both years.
- cover: Extract visual elements or descriptions that might appear in text

If a constraint is mentioned (like "starts with B" or "before 2015"), extract the actual searchable value (like "B" for author initial, or the year for date ranges).

If no information is available for a field, return "N/A".

Return your response as a JSON object with these exact keys: plot, title, author, genre, date, cover"""

        user_prompt = f"Analyze this book query and extract information for each field:\n\nQuery: {query}"
        
        try:
            # Use new or old API depending on what's available
            if NEW_API and self.client:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                result_text = response.choices[0].message.content.strip()
            else:
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
            
            # Post-process to improve searchability and extract constraints
            result_dict, constraints = self._post_process_decomposition(result_dict, query)
            
            return DecomposedQuery(
                plot=result_dict.get("plot", "N/A"),
                title=result_dict.get("title", "N/A"),
                author=result_dict.get("author", "N/A"),
                genre=result_dict.get("genre", "N/A"),
                date=result_dict.get("date", "N/A"),
                cover=result_dict.get("cover", "N/A"),
                _constraints=constraints
            )
            
        except Exception as e:
            error_msg = str(e)
            # Re-raise exception so caller can handle fallback
            raise RuntimeError(f"GPT decomposition failed: {error_msg}") from e
    
    def _predictive_decomposition(self, query: str) -> DecomposedQuery:
        """Predict missing metadata based on context clues"""
        
        system_prompt = """You are an expert at analyzing vague book queries and predicting missing metadata based on context clues.

For each field, predict or infer SEARCHABLE information from the query. Use your knowledge to fill in missing details, but provide ACTUAL SEARCHABLE TEXT, not constraint descriptions.

Fields to predict:
- plot: Story elements, plot points, character names, settings (actual searchable text)
- title: Likely title words or phrases based on plot description (actual title terms)
- author: Likely author names, surnames, or initials based on genre, time period, or writing style clues (actual author text)
- genre: Most likely genre name based on plot elements (actual genre term like "fantasy")
- date: Estimated publication year or date based on context (actual year or date range like "2005" or "1990-2010", not descriptions)
- cover: Likely visual elements based on genre and plot (actual descriptive terms)

IMPORTANT: Extract actual searchable keywords/text, not descriptions of constraints.

Return your response as a JSON object with these exact keys: plot, title, author, genre, date, cover"""

        user_prompt = f"Analyze this book query and predict missing metadata:\n\nQuery: {query}"
        
        try:
            # Use new or old API depending on what's available
            if NEW_API and self.client:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,  # Higher temperature for more creative predictions
                    max_tokens=500
                )
                result_text = response.choices[0].message.content.strip()
            else:
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
            
            # Post-process to improve searchability and extract constraints
            result_dict, constraints = self._post_process_decomposition(result_dict, query)
            
            return DecomposedQuery(
                plot=result_dict.get("plot", "N/A"),
                title=result_dict.get("title", "N/A"),
                author=result_dict.get("author", "N/A"),
                genre=result_dict.get("genre", "N/A"),
                date=result_dict.get("date", "N/A"),
                cover=result_dict.get("cover", "N/A"),
                _constraints=constraints
            )
            
        except Exception as e:
            print(f"Error in predictive decomposition: {e}")
            # Re-raise exception so caller can handle fallback
            raise RuntimeError(f"GPT decomposition failed: {e}")
    
    def _post_process_decomposition(self, result_dict: Dict[str, str], original_query: str = "") -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
        """
        Post-process decomposition results to improve searchability and extract constraints.
        Handles special cases like "starts with X" patterns and structural constraints.
        
        Returns:
            (result_dict, constraints_dict) where constraints_dict contains structural filters
        """
        import re
        constraints = {}
        
        # Extract structural constraints from original query
        # Title constraints
        title_constraints = self._extract_title_constraints(original_query)
        if title_constraints:
            constraints['title'] = title_constraints
        
        # Author constraints
        author_constraints = self._extract_author_constraints(original_query)
        if author_constraints:
            constraints['author'] = author_constraints
        
        # Date constraints
        date_constraints = self._extract_date_constraints(original_query)
        if date_constraints:
            constraints['date'] = date_constraints
        
        # Handle author field - if it's a single letter, use prefix search
        author = result_dict.get("author", "N/A")
        if author and author != "N/A" and len(author.strip()) == 1:
            # Single letter - use prefix wildcard for "starts with" matching
            result_dict["author"] = f"{author.strip()}*"
        
        # Handle date field - extract year from "before X" or "after X"
        date = result_dict.get("date", "N/A")
        if date and date != "N/A":
            # Try to extract year from descriptions
            before_match = re.search(r'before\s+(\d{4})', date, re.IGNORECASE)
            after_match = re.search(r'after\s+(\d{4})', date, re.IGNORECASE)
            if before_match:
                result_dict["date"] = before_match.group(1)
                # Also set constraint for filtering
                if 'date' not in constraints:
                    constraints['date'] = {}
                constraints['date']['max_year'] = int(before_match.group(1))
            elif after_match:
                result_dict["date"] = after_match.group(1)
                if 'date' not in constraints:
                    constraints['date'] = {}
                constraints['date']['min_year'] = int(after_match.group(1))
            elif not re.match(r'^\d{4}(-\d{4})?$', date.strip()):
                # Not a simple year format, try to extract year
                year_match = re.search(r'\b(\d{4})\b', date)
                if year_match:
                    result_dict["date"] = year_match.group(1)
        
        # Remove non-searchable constraint descriptions from search fields
        # These will be handled via post-filtering instead
        for field in ['title', 'author']:
            field_value = result_dict.get(field, "N/A")
            if field_value and field_value != "N/A":
                # Check if it's a constraint description (not searchable text)
                if self._is_constraint_description(field_value):
                    # Keep constraint but clear search field
                    if field not in constraints:
                        constraints[field] = {}
                    # Extract the constraint info
                    constraint_info = self._parse_constraint_description(field, field_value)
                    if constraint_info:
                        constraints[field].update(constraint_info)
                    # Clear searchable text if it's only a constraint
                    if not self._has_searchable_content(field_value):
                        result_dict[field] = "N/A"
        
        return result_dict, constraints
    
    def _extract_title_constraints(self, query: str) -> Dict[str, Any]:
        """Extract structural constraints about title (word count, length, etc.)"""
        constraints = {}
        import re
        
        # Word count: "title with 3 words", "title is 3 words", "3-word title"
        word_count_match = re.search(r'title.*?(\d+)\s*word', query, re.IGNORECASE)
        if not word_count_match:
            word_count_match = re.search(r'(\d+)[-\s]word.*?title', query, re.IGNORECASE)
        if word_count_match:
            constraints['word_count'] = int(word_count_match.group(1))
        
        # Length: "title shorter than 10 words", "title longer than 5 words"
        shorter_match = re.search(r'title.*?shorter.*?than\s+(\d+)', query, re.IGNORECASE)
        if shorter_match:
            constraints['max_words'] = int(shorter_match.group(1))
        
        longer_match = re.search(r'title.*?longer.*?than\s+(\d+)', query, re.IGNORECASE)
        if longer_match:
            constraints['min_words'] = int(longer_match.group(1))
        
        return constraints if constraints else None
    
    def _extract_author_constraints(self, query: str) -> Dict[str, Any]:
        """Extract structural constraints about author"""
        constraints = {}
        import re
        
        # Name count: "author with 2 names", "2-name author"
        name_count_match = re.search(r'author.*?(\d+)\s*name', query, re.IGNORECASE)
        if not name_count_match:
            name_count_match = re.search(r'(\d+)[-\s]name.*?author', query, re.IGNORECASE)
        if name_count_match:
            constraints['name_count'] = int(name_count_match.group(1))
        
        return constraints if constraints else None
    
    def _extract_date_constraints(self, query: str) -> Dict[str, Any]:
        """Extract date constraints"""
        constraints = {}
        import re
        
        # Before/after constraints already handled in main post-processing
        # Add range constraints here if needed
        range_match = re.search(r'between\s+(\d{4})\s+and\s+(\d{4})', query, re.IGNORECASE)
        if range_match:
            constraints['min_year'] = int(range_match.group(1))
            constraints['max_year'] = int(range_match.group(2))
        
        return constraints if constraints else None
    
    def _is_constraint_description(self, text: str) -> bool:
        """Check if text is a constraint description rather than searchable content"""
        constraint_patterns = [
            r'\d+\s*word',
            r'starts?\s+with',
            r'begins?\s+with',
            r'shorter\s+than',
            r'longer\s+than',
            r'\d+\s*name',
            r'\d+\s*characters?',
        ]
        import re
        text_lower = text.lower()
        for pattern in constraint_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def _has_searchable_content(self, text: str) -> bool:
        """Check if text has actual searchable content beyond constraints"""
        # Remove constraint patterns and see if anything remains
        import re
        cleaned = text
        constraint_patterns = [
            r'\d+\s*word',
            r'starts?\s+with',
            r'begins?\s+with',
            r'shorter\s+than',
            r'longer\s+than',
        ]
        for pattern in constraint_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Check if meaningful content remains (at least 3 chars of non-constraint text)
        cleaned = cleaned.strip()
        if len(cleaned) >= 3:
            return True
        return False
    
    def _parse_constraint_description(self, field: str, text: str) -> Dict[str, Any]:
        """Parse constraint description to extract constraint values"""
        constraints = {}
        import re
        
        if field == 'title':
            # Extract word count from "title with 3 words"
            word_match = re.search(r'(\d+)\s*word', text, re.IGNORECASE)
            if word_match:
                constraints['word_count'] = int(word_match.group(1))
        
        if field == 'author':
            # Extract initial from "starts with B"
            starts_match = re.search(r'starts?\s+with\s+["\']?([A-Za-z])', text, re.IGNORECASE)
            if starts_match:
                constraints['starts_with'] = starts_match.group(1).upper()
        
        return constraints
    
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
