"""
Evaluation Entry Script.

Usage:
    # Evaluate a single method
    python scripts/evaluate.py --results_dir results/ --annotation_file data/annotations/test.json --output report.json

    # Compare multiple methods
    python scripts/evaluate.py --compare no_preprocess=results/baseline fixed_pipeline=results/fixed our_method=results/ours --annotation_file data/annotations/test.json --output comparison.json

    # Analyze learning curve
    python scripts/evaluate.py --learning_curve results/ --output learning_curve.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.evaluator import Evaluator


def parse_compare_args(compare_list):
    """Parse --compare argument into {method_name: results_dir} dict.

    Format: method_name=results_dir
    """
    result = {}
    for item in compare_list:
        if "=" not in item:
            print(f"Warning: Invalid compare format '{item}', expected 'name=path'")
            continue
        name, path = item.split("=", 1)
        result[name.strip()] = path.strip()
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate detection results")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory containing results.json for single method evaluation")
    parser.add_argument("--annotation_file", type=str, required=True,
                        help="Path to COCO format annotation JSON file")
    parser.add_argument("--output", type=str, default="evaluation_report.json",
                        help="Output path for evaluation report (JSON)")
    parser.add_argument("--compare", nargs="+", default=None,
                        help="Compare multiple methods: name1=path1 name2=path2 ...")
    parser.add_argument("--learning_curve", type=str, default=None,
                        help="Results directory for learning curve analysis")
    args = parser.parse_args()

    results = {}

    # Single method evaluation
    if args.results_dir:
        print(f"Evaluating results from: {args.results_dir}")
        evaluation = Evaluator.evaluate_results(args.results_dir, args.annotation_file)
        results["evaluation"] = evaluation

        print(f"\n=== Evaluation Results ===")
        print(f"  mAP@0.5:      {evaluation.get('mAP@0.5', 0.0):.4f}")
        print(f"  mAP@0.5:0.95: {evaluation.get('mAP@0.5:0.95', 0.0):.4f}")
        print(f"  Recall:        {evaluation.get('recall', 0.0):.4f}")
        print(f"  Precision:     {evaluation.get('precision', 0.0):.4f}")
        print(f"  F1:            {evaluation.get('f1', 0.0):.4f}")
        print(f"  Preprocess gain (mAP): {evaluation.get('preprocessing_gain_mAP', 0.0):+.4f}")

        qp = evaluation.get("quality_group_performance", {})
        if qp:
            print(f"\n  Quality Group Performance:")
            for group in ["low", "medium", "high"]:
                info = qp.get(group, {})
                print(f"    {group}: count={info.get('count', 0)}, "
                      f"mean_mAP={info.get('mean_mAP', 0.0):.4f}")

    # Method comparison
    if args.compare:
        results_dirs = parse_compare_args(args.compare)
        print(f"\nComparing methods: {list(results_dirs.keys())}")
        comparison = Evaluator.compare_methods(results_dirs, args.annotation_file)
        results["comparison"] = comparison

        # Print comparison table
        comp_info = comparison.get("comparison", {})
        metrics = comp_info.get("metrics", {})
        methods = comp_info.get("methods", [])

        if methods and metrics:
            print(f"\n=== Method Comparison ===")
            header = f"{'Metric':<20}" + "".join(f"{m:<15}" for m in methods) + "Best"
            print(header)
            print("-" * len(header))
            for metric_name, values in metrics.items():
                row = f"{metric_name:<20}"
                for method in methods:
                    row += f"{values.get(method, 0.0):<15.4f}"
                row += values.get("best", "N/A")
                print(row)

    # Learning curve
    if args.learning_curve:
        print(f"\nAnalyzing learning curve from: {args.learning_curve}")
        curve = Evaluator.plot_learning_curve(args.learning_curve)
        results["learning_curve"] = curve

        if curve.get("num_images"):
            print(f"\n=== Learning Curve ===")
            for n, d, r in zip(
                curve["num_images"],
                curve.get("mAP_delta", []),
                curve.get("improved_ratio", []),
            ):
                print(f"  n={n:>5}: mAP_delta={d:+.4f}, improved_ratio={r:.3f}")

    # Generate report
    if results:
        Evaluator.generate_report(results, args.output)
        print(f"\nReport saved to: {args.output}")
    else:
        print("No evaluation performed. Use --results_dir, --compare, or --learning_curve.")


if __name__ == "__main__":
    main()
