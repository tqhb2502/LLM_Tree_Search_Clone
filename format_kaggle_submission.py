#!/usr/bin/env python3
"""
Format Game24 evaluation results for Kaggle submission.
This script processes the evaluation results and creates a submission file.
"""

import json
import pandas as pd
import os
from pathlib import Path
import argparse
import numpy as np
from typing import Dict, List, Any


def load_evaluation_results(results_dir: str) -> Dict[str, Any]:
    """Load evaluation results from different methods."""
    results = {}
    
    methods = ['mcts_results', 'cot_sc_results', 'cot_greedy_results']
    
    for method in methods:
        method_dir = Path(results_dir) / method
        if method_dir.exists():
            # Look for result files
            for file_path in method_dir.glob("*.json"):
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        results[method] = data
                        print(f"Loaded results from {method}: {file_path}")
                        break
                    except json.JSONDecodeError:
                        continue
            
            # Also check for jsonl files
            if method not in results:
                for file_path in method_dir.glob("*.jsonl"):
                    try:
                        data = []
                        with open(file_path, 'r') as f:
                            for line in f:
                                data.append(json.loads(line.strip()))
                        results[method] = data
                        print(f"Loaded results from {method}: {file_path}")
                        break
                    except json.JSONDecodeError:
                        continue
    
    return results


def extract_problem_solutions(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract problem-solution pairs from evaluation results."""
    problem_solutions = []
    
    # Process each method's results
    for method, data in results.items():
        if isinstance(data, list):
            for item in data:
                if 'question' in item:
                    problem = item['question'].strip()
                    # Extract the best solution
                    solution = extract_best_solution(item, method)
                    
                    problem_solutions.append({
                        'problem': problem,
                        'solution': solution,
                        'method': method,
                        'confidence': get_solution_confidence(item, method)
                    })
        elif isinstance data, dict):
            # Handle different result formats
            if 'results' in data:
                for item in data['results']:
                    if 'question' in item:
                        problem = item['question'].strip()
                        solution = extract_best_solution(item, method)
                        
                        problem_solutions.append({
                            'problem': problem,
                            'solution': solution,
                            'method': method,
                            'confidence': get_solution_confidence(item, method)
                        })
    
    return problem_solutions


def extract_best_solution(item: Dict[str, Any], method: str) -> str:
    """Extract the best solution from an evaluation item."""
    
    # Try different keys where solutions might be stored
    solution_keys = ['answer', 'prediction', 'output', 'best_answer', 'solution']
    
    for key in solution_keys:
        if key in item:
            if isinstance(item[key], list) and len(item[key]) > 0:
                return str(item[key][0])
            elif isinstance(item[key], str):
                return item[key]
    
    # If no direct solution found, try to construct from outputs
    if 'outputs' in item and isinstance(item['outputs'], list) and len(item['outputs']) > 0:
        return str(item['outputs'][0])
    
    # Default fallback
    return "No solution found"


def get_solution_confidence(item: Dict[str, Any], method: str) -> float:
    """Get confidence score for a solution."""
    
    # Try different keys where confidence might be stored
    confidence_keys = ['confidence', 'score', 'value', 'probability']
    
    for key in confidence_keys:
        if key in item:
            if isinstance(item[key], (int, float)):
                return float(item[key])
            elif isinstance(item[key], list) and len(item[key]) > 0:
                return float(item[key][0])
    
    # Default confidence based on method
    method_defaults = {
        'mcts_results': 0.8,
        'cot_sc_results': 0.7,
        'cot_greedy_results': 0.6
    }
    
    return method_defaults.get(method, 0.5)


def aggregate_solutions(problem_solutions: List[Dict[str, Any]]) -> Dict[str, str]:
    """Aggregate solutions for problems that appear multiple times."""
    
    problem_map = {}
    
    for item in problem_solutions:
        problem = item['problem']
        solution = item['solution']
        confidence = item['confidence']
        
        if problem not in problem_map:
            problem_map[problem] = []
        
        problem_map[problem].append({
            'solution': solution,
            'confidence': confidence,
            'method': item['method']
        })
    
    # Select best solution for each problem
    final_solutions = {}
    
    for problem, solutions in problem_map.items():
        # Sort by confidence and select the best
        best_solution = max(solutions, key=lambda x: x['confidence'])
        final_solutions[problem] = best_solution['solution']
    
    return final_solutions


def load_game24_test_data() -> pd.DataFrame:
    """Load the Game24 test dataset."""
    csv_path = Path("tsllm/envs/game24/24.csv")
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    else:
        print("Warning: Game24 CSV file not found. Creating dummy data.")
        # Create some example problems for demonstration
        problems = [
            "1 1 4 6",
            "1 1 11 11", 
            "1 1 3 8",
            "1 1 1 8",
            "6 6 6 6"
        ]
        return pd.DataFrame({'Puzzles': problems})


def create_kaggle_submission(final_solutions: Dict[str, str], output_path: str):
    """Create Kaggle submission file."""
    
    # Load test data to get the correct format
    test_df = load_game24_test_data()
    
    submission_data = []
    
    for idx, row in test_df.iterrows():
        if 'Puzzles' in row:
            problem = str(row['Puzzles']).strip()
            solution = final_solutions.get(problem, "No solution found")
            
            submission_data.append({
                'id': idx,
                'problem': problem,
                'solution': solution
            })
        else:
            # Handle different column names
            problem = str(row.iloc[1] if len(row) > 1 else row.iloc[0]).strip()
            solution = final_solutions.get(problem, "No solution found")
            
            submission_data.append({
                'id': idx,
                'problem': problem,
                'solution': solution
            })
    
    # Create submission DataFrame
    submission_df = pd.DataFrame(submission_data)
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Kaggle submission saved to: {output_path}")
    
    # Print statistics
    solved_count = len([s for s in submission_df['solution'] if s != "No solution found"])
    total_count = len(submission_df)
    print(f"Solutions found: {solved_count}/{total_count} ({solved_count/total_count*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Format Game24 evaluation results for Kaggle submission")
    parser.add_argument("--results_dir", type=str, default="results", 
                       help="Directory containing evaluation results")
    parser.add_argument("--output", type=str, default="kaggle_submission/game24_submission.csv",
                       help="Output file for Kaggle submission")
    
    args = parser.parse_args()
    
    print("Loading evaluation results...")
    results = load_evaluation_results(args.results_dir)
    
    if not results:
        print("No evaluation results found. Please run the evaluation first.")
        return
    
    print("Extracting problem-solution pairs...")
    problem_solutions = extract_problem_solutions(results)
    
    print(f"Found {len(problem_solutions)} problem-solution pairs")
    
    print("Aggregating solutions...")
    final_solutions = aggregate_solutions(problem_solutions)
    
    print(f"Final solutions for {len(final_solutions)} unique problems")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("Creating Kaggle submission file...")
    create_kaggle_submission(final_solutions, args.output)
    
    print("Done!")


if __name__ == "__main__":
    main()