#!/usr/bin/env python3
"""
Standalone evaluation script for RAG pipeline.

This script can be run independently to evaluate all strategies
and generate a comprehensive comparison report.

Usage:
    python scripts/run_evaluation.py [--quick] [--output results.json]
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.evaluation import RAGEvaluator, ResultsAnalyzer


def main():
    """Main evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive RAG pipeline evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick evaluation: skip answer judging, use fewer questions"
    )
    
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate"
    )
    
    parser.add_argument(
        "--strategy",
        choices=["fixed", "semantic", "hierarchical"],
        default=None,
        help="Evaluate specific strategy only"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for results JSON"
    )
    
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating markdown report"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = RAGEvaluator(config)
    analyzer = ResultsAnalyzer()
    
    # Determine settings
    judge_answers = not args.quick
    max_questions = args.max_questions
    if args.quick and max_questions is None:
        max_questions = 10  # Quick mode: only 10 questions
    
    print("\n" + "=" * 70)
    print("RAG PIPELINE EVALUATION")
    print("=" * 70)
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Max questions: {max_questions or 'All (35)'}")
    print(f"Judging answers: {judge_answers}")
    print(f"Strategy: {args.strategy or 'All'}")
    print("=" * 70 + "\n")
    
    # Run evaluation
    print("Running evaluation (this may take several minutes)...\n")
    
    try:
        results = evaluator.run_evaluation(
            strategy=args.strategy,
            max_questions=max_questions,
            judge_answers=judge_answers
        )
        
        # Print summary
        print("\n")
        analyzer.print_summary(results)
        
        # Save results
        if args.output:
            output_path = args.output
        else:
            results_dir = config.base_path / "data/evaluation/results"
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = "quick" if args.quick else "full"
            output_path = results_dir / f"evaluation_{mode}_{timestamp}.json"
        
        evaluator.save_results(results, output_path)
        print(f"\n✅ Results saved to: {output_path}")
        
        # Generate report
        if not args.no_report:
            report_path = output_path.with_suffix('.md')
            report_content = analyzer.generate_comparison_report(results)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"✅ Report saved to: {report_path}")
        
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

