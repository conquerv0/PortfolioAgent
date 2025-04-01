#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run both Fixed Income and Forex evaluators.
"""

import os
import argparse
from fi_evaluator import FixedIncomeEvaluator
from fx_evaluator import ForexEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run evaluations for prediction models')
    parser.add_argument('--eval-type', type=str, choices=['fi', 'fx', 'all'], default='all',
                        help='Type of evaluation to run (fi=Fixed Income, fx=Forex, all=Both)')
    parser.add_argument('--fi-pred', type=str, default='data/fi_weekly_predictions.csv',
                        help='Path to Fixed Income predictions file')
    parser.add_argument('--fi-data', type=str, default='data/fi_combined_features_weekly.csv',
                        help='Path to Fixed Income actual data file')
    # parser.add_argument('--fx-pred', type=str, default='data/fx_weekly_predictions.csv',
    #                     help='Path to Forex predictions file')
    # parser.add_argument('--fx-data', type=str, default='data/fx_combined_features_weekly.csv',
    #                     help='Path to Forex actual data file')
    parser.add_argument('--output-dir', type=str, default='data/evaluation',
                        help='Base directory for saving evaluation results')
    
    return parser.parse_args()


def run_fixed_income_evaluation(args):
    """Run evaluation for Fixed Income predictions."""
    print("\n" + "="*50)
    print("RUNNING FIXED INCOME EVALUATION")
    print("="*50)
    
    # Create output directory
    fi_output_dir = os.path.join(args.output_dir, 'fixed_income')
    os.makedirs(fi_output_dir, exist_ok=True)
    
    # Initialize and run evaluator
    fi_evaluator = FixedIncomeEvaluator(
        predictions_file=args.fi_pred,
        actual_data_file=args.fi_data,
        output_dir=fi_output_dir
    )
    
    # Run evaluation
    fi_evaluator.evaluate()
    
    print("\nFixed Income evaluation complete!")
    print(f"Results saved to: {fi_output_dir}")


def run_forex_evaluation(args):
    """Run evaluation for Forex predictions."""
    print("\n" + "="*50)
    print("RUNNING FOREX EVALUATION")
    print("="*50)
    
    # Create output directory
    fx_output_dir = os.path.join(args.output_dir, 'forex')
    os.makedirs(fx_output_dir, exist_ok=True)
    
    # Initialize and run evaluator
    fx_evaluator = ForexEvaluator(
        predictions_file=args.fx_pred,
        actual_data_file=args.fx_data,
        output_dir=fx_output_dir
    )
    
    # Run evaluation
    fx_evaluator.evaluate()
    
    print("\nForex evaluation complete!")
    print(f"Results saved to: {fx_output_dir}")


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_args()
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluations based on the specified type
    if args.eval_type in ['fi', 'all']:
        run_fixed_income_evaluation(args)
    
    # if args.eval_type in ['fx', 'all']:
    #     run_forex_evaluation(args)
    
    print("\nAll evaluations completed successfully!")


if __name__ == "__main__":
    main() 