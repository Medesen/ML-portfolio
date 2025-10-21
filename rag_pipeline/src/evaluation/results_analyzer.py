"""Statistical analysis and reporting for evaluation results."""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime

from ..utils.logger import get_logger


class ResultsAnalyzer:
    """
    Analyzes evaluation results and generates reports.
    
    Provides statistical analysis, comparison tables, and insights.
    """
    
    def __init__(self, logger_name: str = "results_analyzer"):
        """
        Initialize results analyzer.
        
        Args:
            logger_name: Logger name
        """
        self.logger = get_logger(logger_name)
    
    def load_results(self, results_path: Path) -> Dict[str, Any]:
        """
        Load evaluation results from JSON file.
        
        Args:
            results_path: Path to results JSON
            
        Returns:
            Results dictionary
        """
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        self.logger.info(f"Loaded results from: {results_path}")
        return results
    
    def generate_summary_table(
        self,
        results: Dict[str, Any],
        metrics: Optional[List[str]] = None
    ) -> str:
        """
        Generate a markdown summary table comparing strategies.
        
        Args:
            results: Evaluation results
            metrics: List of metrics to include (None for defaults)
            
        Returns:
            Markdown table string
        """
        if metrics is None:
            metrics = ["recall@5", "recall@10", "recall@20", "mrr", "ndcg@10", "topic_coverage"]
        
        results_by_strategy = results.get("results_by_strategy", {})
        
        if not results_by_strategy:
            return "No results available."
        
        # Build table header
        table = "| Metric | " + " | ".join(results_by_strategy.keys()) + " |\n"
        table += "|" + "---|" * (len(results_by_strategy) + 1) + "\n"
        
        # Add rows for each metric
        for metric in metrics:
            row = f"| **{metric}** |"
            
            for strategy, strategy_results in results_by_strategy.items():
                value = strategy_results.get("aggregated", {}).get("retrieval_metrics", {}).get(metric)
                
                if value is not None:
                    row += f" {value:.4f} |"
                else:
                    row += " N/A |"
            
            table += row + "\n"
        
        return table
    
    def generate_comparison_report(
        self,
        results: Dict[str, Any]
    ) -> str:
        """
        Generate a comprehensive comparison report in markdown.
        
        Args:
            results: Evaluation results
            
        Returns:
            Markdown report string
        """
        report = []
        
        # Header
        report.append("# RAG Pipeline Evaluation Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        metadata = results.get("metadata", {})
        report.append(f"\n**Evaluation Date:** {metadata.get('timestamp', 'N/A')}")
        report.append(f"**Number of Questions:** {metadata.get('num_questions', 'N/A')}")
        report.append(f"**Strategies Evaluated:** {', '.join(metadata.get('strategies_evaluated', []))}")
        report.append(f"**Total Time:** {metadata.get('total_time', 0):.2f}s")
        
        # Overall Performance Summary
        report.append("\n## Overall Performance Summary")
        report.append("\n" + self.generate_summary_table(results))
        
        # Best Performers
        comparison = results.get("comparison", {})
        if comparison:
            report.append("\n## Best Performing Strategies by Metric")
            report.append("")
            
            for metric, comp_data in comparison.items():
                if "best_strategy" in comp_data:
                    best = comp_data["best_strategy"]
                    value = comp_data["best_value"]
                    report.append(f"- **{metric}**: {best} ({value:.4f})")
        
        # Performance by Category
        report.append("\n## Performance by Question Category")
        results_by_strategy = results.get("results_by_strategy", {})
        
        if results_by_strategy:
            first_strategy = next(iter(results_by_strategy.values()))
            categories = first_strategy.get("aggregated", {}).get("by_category", {})
            
            if categories:
                for category in categories.keys():
                    report.append(f"\n### {category.title()}")
                    cat_table = self._generate_category_table(results_by_strategy, category)
                    report.append(cat_table)
        
        # Performance by Difficulty
        report.append("\n## Performance by Question Difficulty")
        
        if results_by_strategy:
            first_strategy = next(iter(results_by_strategy.values()))
            difficulties = first_strategy.get("aggregated", {}).get("by_difficulty", {})
            
            if difficulties:
                for difficulty in ["easy", "medium", "hard"]:
                    if difficulty in difficulties:
                        report.append(f"\n### {difficulty.title()}")
                        diff_table = self._generate_difficulty_table(results_by_strategy, difficulty)
                        report.append(diff_table)
        
        # Insights and Recommendations
        report.append("\n## Key Insights")
        insights = self._generate_insights(results)
        for insight in insights:
            report.append(f"- {insight}")
        
        # Limitations
        report.append("\n## Limitations")
        report.append("- LLM-based judging may have biases and inconsistencies")
        report.append("- Test set size (35 questions) may not cover all edge cases")
        report.append("- Relevant doc IDs are manually curated and may be incomplete")
        report.append("- Answer quality scores are subjective assessments")
        
        return "\n".join(report)
    
    def _generate_category_table(
        self,
        results_by_strategy: Dict[str, Any],
        category: str
    ) -> str:
        """Generate comparison table for a specific category."""
        
        metrics = ["recall@10", "mrr", "ndcg@10"]
        
        table = "| Metric | " + " | ".join(results_by_strategy.keys()) + " |\n"
        table += "|" + "---|" * (len(results_by_strategy) + 1) + "\n"
        
        for metric in metrics:
            row = f"| {metric} |"
            for strategy, strategy_results in results_by_strategy.items():
                cat_metrics = strategy_results.get("aggregated", {}).get("by_category", {}).get(category, {})
                value = cat_metrics.get(metric)
                
                if value is not None:
                    row += f" {value:.4f} |"
                else:
                    row += " N/A |"
            
            table += row + "\n"
        
        return table
    
    def _generate_difficulty_table(
        self,
        results_by_strategy: Dict[str, Any],
        difficulty: str
    ) -> str:
        """Generate comparison table for a specific difficulty."""
        
        metrics = ["recall@10", "mrr", "ndcg@10"]
        
        table = "| Metric | " + " | ".join(results_by_strategy.keys()) + " |\n"
        table += "|" + "---|" * (len(results_by_strategy) + 1) + "\n"
        
        for metric in metrics:
            row = f"| {metric} |"
            for strategy, strategy_results in results_by_strategy.items():
                diff_metrics = strategy_results.get("aggregated", {}).get("by_difficulty", {}).get(difficulty, {})
                value = diff_metrics.get(metric)
                
                if value is not None:
                    row += f" {value:.4f} |"
                else:
                    row += " N/A |"
            
            table += row + "\n"
        
        return table
    
    def _generate_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate key insights from results."""
        
        insights = []
        
        comparison = results.get("comparison", {})
        
        # Overall best strategy
        if "recall@10" in comparison:
            best = comparison["recall@10"]["best_strategy"]
            value = comparison["recall@10"]["best_value"]
            insights.append(
                f"**{best.title()}** strategy achieves best Recall@10 ({value:.4f}), "
                "indicating it retrieves the most relevant documents"
            )
        
        if "mrr" in comparison:
            best = comparison["mrr"]["best_strategy"]
            value = comparison["mrr"]["best_value"]
            insights.append(
                f"**{best.title()}** strategy has best MRR ({value:.4f}), "
                "meaning it ranks relevant documents highest on average"
            )
        
        # Check for consistency across metrics
        best_strategies = set()
        for metric, comp_data in comparison.items():
            if "best_strategy" in comp_data:
                best_strategies.add(comp_data["best_strategy"])
        
        if len(best_strategies) == 1:
            winner = best_strategies.pop()
            insights.append(
                f"**{winner.title()}** strategy is the clear winner, "
                "performing best across all metrics"
            )
        elif len(best_strategies) > 1:
            insights.append(
                f"Performance varies by metric, suggesting different strategies "
                f"excel in different aspects: {', '.join(best_strategies)}"
            )
        
        return insights
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        Print evaluation summary to console.
        
        Args:
            results: Evaluation results
        """
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        
        metadata = results.get("metadata", {})
        print(f"\nQuestions evaluated: {metadata.get('num_questions', 'N/A')}")
        print(f"Strategies: {', '.join(metadata.get('strategies_evaluated', []))}")
        print(f"Total time: {metadata.get('total_time', 0):.2f}s")
        
        print("\n" + self.generate_summary_table(results))
        
        # Best performers
        comparison = results.get("comparison", {})
        if comparison:
            print("\nBEST PERFORMERS:")
            for metric, comp_data in comparison.items():
                if "best_strategy" in comp_data:
                    best = comp_data["best_strategy"]
                    value = comp_data["best_value"]
                    print(f"  {metric}: {best} ({value:.4f})")
        
        print("\n" + "=" * 70)

