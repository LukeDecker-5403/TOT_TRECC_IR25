"""
Ensemble Retriever for TOT System
Combines multiple field-based retrievers (plot, title, author, etc.)
using weighted score fusion.
"""

import os
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any
from dataclasses import dataclass
from .config import Config

@dataclass
class RetrievalResult:
    doc_id: str
    score: float
    field_scores: Dict[str, float]
    metadata: Dict[str, Any]

class EnsembleRetriever:
    """Weighted ensemble retriever using FAISS + SentenceTransformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.field_weights = {
            "plot": 0.4,
            "title": 0.2,
            "author": 0.15,
            "genre": 0.15,
            "date": 0.05,
            "cover": 0.05,
        }
        self.indices = {}
        self.embeddings = {}
        self.documents = None

    def build_index(self, documents: List[Dict[str, Any]]):
        """Build FAISS indices for each field"""
        print("Building FAISS indices...")
        self.documents = documents

        for field in self.field_weights.keys():
            print(f"Encoding {field}...")
            texts = [doc.get(field, "") for doc in documents]
            emb = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            index = faiss.IndexFlatIP(emb.shape[1])
            faiss.normalize_L2(emb)
            index.add(emb)
            self.indices[field] = index
            self.embeddings[field] = emb

        print("All indices built!")

    def retrieve(self, decomposed_query: Dict[str, str], top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve top documents for a decomposed query"""
        if not self.indices:
            raise ValueError("No indices built. Call build_index() first.")

        field_scores = {}
        n_docs = len(self.documents)

        # Collect per-field scores
        for field, query_text in decomposed_query.items():
            if field not in self.indices or not query_text or query_text == "N/A":
                continue
            q_emb = self.model.encode([query_text], convert_to_numpy=True)
            faiss.normalize_L2(q_emb)
            D, I = self.indices[field].search(q_emb, n_docs)
            field_scores[field] = (D[0], I[0])

        # Combine scores using weighted sum
        combined_scores = np.zeros(n_docs)
        for field, (scores, ids) in field_scores.items():
            weight = self.field_weights.get(field, 0)
            combined_scores[ids] += weight * scores

        # Sort and get top_k
        top_indices = np.argsort(-combined_scores)[:top_k]
        results = [
            RetrievalResult(
                doc_id=self.documents[i]["id"],
                score=float(combined_scores[i]),
                field_scores={f: float(field_scores[f][0][np.where(field_scores[f][1]==i)][0])
                              for f in field_scores if i in field_scores[f][1]},
                metadata=self.documents[i]
            )
            for i in top_indices
        ]
        return results

    def optimize_weights(self, val_decomposed, val_labels, metric="recall@5"):
        """Stub: Optimize weights via grid search or gradient-free tuning"""
        print("⚙️ Weight optimization not implemented yet (stub).")
        return self.field_weights

    def get_stats(self):
        """Return retriever stats"""
        return {
            "num_docs": len(self.documents) if self.documents else 0,
            "num_fields": len(self.indices),
            "model": self.model.get_sentence_embedding_dimension(),
        }
