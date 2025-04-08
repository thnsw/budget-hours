#!/usr/bin/env python3
import os
import sys
import argparse
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from data_extraction import extract_data_for_classification
from llm_classifier import classify_batch
from output_generator import generate_csv_output, generate_summary_report

def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Billable Hours Classifier - LLM-based classification of time entries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract data from SQL Server without classification"
    )
    
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help="Only classify data (requires input file with --input)"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input CSV file path (for --classify-only mode)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path"
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate a summary report"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Compare predicted approvals with actual approvals and generate evaluation metrics"
    )
    
    parser.add_argument(
        "--summary-output",
        type=str,
        help="Path for the summary report output"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of records to process"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for API rate limiting"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making actual API calls (for testing)"
    )
    
    return parser

def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments"""
    if args.classify_only and not args.input:
        raise ValueError("--classify-only requires --input to specify the input file")
    
    if args.input and not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

def load_csv_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a CSV file"""
    import pandas as pd
    try:
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    except Exception as e:
        raise Exception(f"Error loading CSV data: {str(e)}")

def mock_classify_batch(data: List[Dict[str, Any]], batch_size: int = 100) -> List[Dict[str, Any]]:
    """Mock classification function for dry runs"""
    print("DRY RUN: Mocking LLM classification")
    results = []
    
    for entry in data:
        # Simulate processing time
        time.sleep(0.1)
        
        mock_result = entry.copy()
        # Simple mock classification logic - client projects are billable
        is_billable = 'client' in entry.get('project_name', '').lower()
        
        mock_result.update({
            "is_billable_predicted": is_billable,
            "classification_confidence": 0.9 if is_billable else 0.8,
            "classification_reasoning": "DRY RUN - mock classification"
        })
        
        results.append(mock_result)
        
    return results

def run_pipeline(args: argparse.Namespace) -> None:
    """Run the main pipeline based on command line arguments"""
    try:
        if args.extract_only:
            print("Extracting data from SQL Server...")
            data = extract_data_for_classification()
            if args.output:
                generate_csv_output(data, args.output)
            return
        
        if args.classify_only:
            if not args.input:
                raise ValueError("--input is required when using --classify-only")
            print(f"Loading data from {args.input}...")
            data = load_csv_data(args.input)
        else:
            print("Extracting data from SQL Server...")
            data = extract_data_for_classification()
        
        print("Classifying hours...")
        classified_data = classify_batch(data)
        
        if args.output:
            print(f"Generating CSV output to {args.output}...")
            generate_csv_output(classified_data, args.output)
        
        if args.summary:
            print("Generating summary report...")
            generate_summary_report(classified_data, evaluate=args.evaluate)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    
    try:
        validate_args(args)
        run_pipeline(args)
    except Exception as e:
        print(f"Error: {str(e)}")
        parser.print_help()
        sys.exit(1) 