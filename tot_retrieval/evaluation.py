"""
Evaluation module for TOT Retrieval System
Implements metrics and ablation studies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from .ensemble_retriever_pyserini import PyseriniEnsembleRetriever, RetrievalResult
from .config import Config

# Import decomposer based on config
if Config.DEFAULT_DECOMPOSER == "rule_based":
    from .query_decomposer_free import RuleBasedQueryDecomposer as QueryDecomposer
else:
    from .query_decomposer import QueryDecomposer

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    precision_at_5: float
    precision_at_10: float

class TOTEvaluator:
    """Evaluator for TOT Retrieval system"""
    
    def __init__(self, ensemble_retriever: PyseriniEnsembleRetriever, 
                 query_decomposer: QueryDecomposer):
        """
        Initialize evaluator
        
        Args:
            ensemble_retriever: Trained ensemble retriever
            query_decomposer: Query decomposer instance
        """
        self.ensemble_retriever = ensemble_retriever
        self.query_decomposer = query_decomposer
    
    def evaluate(self, test_queries: List[str], test_labels: List[str], 
                 mode: str = "extractive") -> EvaluationMetrics:
        """
        Evaluate the system on test data
        
        Args:
            test_queries: List of test queries
            test_labels: List of ground truth document IDs
            mode: Decomposition mode ("extractive" or "predictive")
            
        Returns:
            EvaluationMetrics object
        """
        print(f"Evaluating on {len(test_queries)} test queries...")
        
        all_recall_at_1 = []
        all_recall_at_5 = []
        all_recall_at_10 = []
        all_mrr = []
        all_precision_at_5 = []
        all_precision_at_10 = []
        
        for i, (query, label) in enumerate(zip(test_queries, test_labels)):
            # Decompose query
            decomposed = self.query_decomposer.decompose(query, mode)
            decomposed_dict = decomposed.to_dict()
            
            # Retrieve documents
            results = self.ensemble_retriever.retrieve(decomposed_dict, top_k=10)
            
            # Calculate metrics
            recall_at_1 = self._calculate_recall_at_k(results, label, 1)
            recall_at_5 = self._calculate_recall_at_k(results, label, 5)
            recall_at_10 = self._calculate_recall_at_k(results, label, 10)
            mrr = self._calculate_mrr(results, label)
            precision_at_5 = self._calculate_precision_at_k(results, label, 5)
            precision_at_10 = self._calculate_precision_at_k(results, label, 10)
            
            all_recall_at_1.append(recall_at_1)
            all_recall_at_5.append(recall_at_5)
            all_recall_at_10.append(recall_at_10)
            all_mrr.append(mrr)
            all_precision_at_5.append(precision_at_5)
            all_precision_at_10.append(precision_at_10)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(test_queries)} queries...")
        
        return EvaluationMetrics(
            recall_at_1=np.mean(all_recall_at_1),
            recall_at_5=np.mean(all_recall_at_5),
            recall_at_10=np.mean(all_recall_at_10),
            mrr=np.mean(all_mrr),
            precision_at_5=np.mean(all_precision_at_5),
            precision_at_10=np.mean(all_precision_at_10)
        )
    
    def ablation_study(self, test_queries: List[str], test_labels: List[str],
                       mode: str = "extractive") -> Dict[str, EvaluationMetrics]:
        """
        Perform ablation study by removing one retriever at a time
        
        Args:
            test_queries: List of test queries
            test_labels: List of ground truth document IDs
            mode: Decomposition mode
            
        Returns:
            Dictionary mapping field to metrics when that field is removed
        """
        print("Performing ablation study...")
        
        ablation_results = {}
        original_weights = self.ensemble_retriever.weights.copy()
        
        # Test removing each field
        for field in ['plot', 'title', 'author', 'genre', 'date', 'cover']:
            print(f"Testing without {field} retriever...")
            
            # Set weight to 0 for this field
            modified_weights = original_weights.copy()
            modified_weights[field] = 0.0
            
            # Normalize remaining weights
            total_weight = sum(modified_weights.values())
            if total_weight > 0:
                modified_weights = {k: v/total_weight for k, v in modified_weights.items()}
            
            # Temporarily update weights
            self.ensemble_retriever.weights = modified_weights
            
            # Evaluate
            metrics = self.evaluate(test_queries, test_labels, mode)
            ablation_results[f"without_{field}"] = metrics
        
        # Restore original weights
        self.ensemble_retriever.weights = original_weights
        
        return ablation_results
    
    def per_field_analysis(self, test_queries: List[str], test_labels: List[str],
                          mode: str = "extractive") -> Dict[str, Dict[str, float]]:
        """
        Analyze performance of individual retrievers
        
        Args:
            test_queries: List of test queries
            test_labels: List of ground truth document IDs
            mode: Decomposition mode
            
        Returns:
            Dictionary with per-field performance metrics
        """
        print("Analyzing per-field performance...")
        
        field_metrics = {}
        
        for field in ['plot', 'title', 'author', 'genre', 'date', 'cover']:
            print(f"Analyzing {field} retriever...")
            
            field_recall_at_5 = []
            field_mrr = []
            
            for query, label in zip(test_queries, test_labels):
                # Decompose query
                decomposed = self.query_decomposer.decompose(query, mode)
                decomposed_dict = decomposed.to_dict()
                
                # Get field-specific query
                field_query = decomposed_dict.get(field, "")
                if field_query and field_query != "N/A":
                    # Retrieve using only this field
                    scores = self.ensemble_retriever.retrievers[field].retrieve(field_query)
                    
                    # Sort by score
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    top_5_ids = [doc_id for doc_id, _ in sorted_scores[:5]]
                    top_10_ids = [doc_id for doc_id, _ in sorted_scores[:10]]
                    
                    # Calculate metrics
                    recall_at_5 = 1.0 if label in top_5_ids else 0.0
                    mrr = self._calculate_mrr_from_scores(scores, label)
                    
                    field_recall_at_5.append(recall_at_5)
                    field_mrr.append(mrr)
            
            field_metrics[field] = {
                'recall_at_5': np.mean(field_recall_at_5) if field_recall_at_5 else 0.0,
                'mrr': np.mean(field_mrr) if field_mrr else 0.0,
                'num_queries': len(field_recall_at_5)
            }
        
        return field_metrics
    
    def _calculate_recall_at_k(self, results: List[RetrievalResult], 
                              label: str, k: int) -> float:
        """Calculate recall@k"""
        top_k_ids = [r.doc_id for r in results[:k]]
        return 1.0 if label in top_k_ids else 0.0
    
    def _calculate_mrr(self, results: List[RetrievalResult], label: str) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, result in enumerate(results):
            if result.doc_id == label:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_mrr_from_scores(self, scores: Dict[str, float], label: str) -> float:
        """Calculate MRR from score dictionary"""
        if label not in scores:
            return 0.0
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for i, (doc_id, _) in enumerate(sorted_scores):
            if doc_id == label:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_precision_at_k(self, results: List[RetrievalResult], 
                                label: str, k: int) -> float:
        """Calculate precision@k"""
        top_k_ids = [r.doc_id for r in results[:k]]
        return 1.0 if label in top_k_ids else 0.0
    
    def plot_ablation_results(self, ablation_results: Dict[str, EvaluationMetrics], 
                            save_path: Optional[str] = None):
        """Plot ablation study results"""
        fields = [k.replace('without_', '') for k in ablation_results.keys()]
        recall_scores = [ablation_results[k].recall_at_5 for k in ablation_results.keys()]
        mrr_scores = [ablation_results[k].mrr for k in ablation_results.keys()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Recall@5 plot
        ax1.bar(fields, recall_scores, color='skyblue')
        ax1.set_title('Recall@5 by Removed Field')
        ax1.set_ylabel('Recall@5')
        ax1.set_xlabel('Removed Field')
        ax1.tick_params(axis='x', rotation=45)
        
        # MRR plot
        ax2.bar(fields, mrr_scores, color='lightcoral')
        ax2.set_title('MRR by Removed Field')
        ax2.set_ylabel('MRR')
        ax2.set_xlabel('Removed Field')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_per_field_analysis(self, field_metrics: Dict[str, Dict[str, float]], 
                               save_path: Optional[str] = None):
        """Plot per-field analysis results"""
        fields = list(field_metrics.keys())
        recall_scores = [field_metrics[f]['recall_at_5'] for f in fields]
        mrr_scores = [field_metrics[f]['mrr'] for f in fields]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Recall@5 plot
        ax1.bar(fields, recall_scores, color='lightgreen')
        ax1.set_title('Individual Retriever Recall@5')
        ax1.set_ylabel('Recall@5')
        ax1.set_xlabel('Field')
        ax1.tick_params(axis='x', rotation=45)
        
        # MRR plot
        ax2.bar(fields, mrr_scores, color='orange')
        ax2.set_title('Individual Retriever MRR')
        ax2.set_ylabel('MRR')
        ax2.set_xlabel('Field')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, metrics: EvaluationMetrics, 
                       ablation_results: Dict[str, EvaluationMetrics],
                       field_metrics: Dict[str, Dict[str, float]]) -> str:
        """Generate comprehensive evaluation report"""
        report = []
        report.append("=" * 50)
        report.append("TOT RETRIEVAL SYSTEM EVALUATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL PERFORMANCE:")
        report.append(f"  Recall@1:  {metrics.recall_at_1:.4f}")
        report.append(f"  Recall@5:  {metrics.recall_at_5:.4f}")
        report.append(f"  Recall@10: {metrics.recall_at_10:.4f}")
        report.append(f"  MRR:       {metrics.mrr:.4f}")
        report.append(f"  Precision@5:  {metrics.precision_at_5:.4f}")
        report.append(f"  Precision@10: {metrics.precision_at_10:.4f}")
        report.append("")
        
        # Ablation study
        report.append("ABLATION STUDY (Recall@5 when field removed):")
        for field, result in ablation_results.items():
            field_name = field.replace('without_', '')
            report.append(f"  Without {field_name}: {result.recall_at_5:.4f}")
        report.append("")
        
        # Per-field analysis
        report.append("INDIVIDUAL RETRIEVER PERFORMANCE:")
        for field, metrics_dict in field_metrics.items():
            report.append(f"  {field.capitalize()}:")
            report.append(f"    Recall@5: {metrics_dict['recall_at_5']:.4f}")
            report.append(f"    MRR:     {metrics_dict['mrr']:.4f}")
            report.append(f"    Queries: {metrics_dict['num_queries']}")
        report.append("")
        
        return "\n".join(report)
