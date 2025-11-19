#!/usr/bin/env python3
"""
Quick validation script to test the Game24 environment and models.
Run this after setup to ensure everything is working correctly.
"""

import sys
import os
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} (NOT FOUND)")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists and print status."""
    if Path(dirpath).is_dir():
        print(f"✓ {description}: {dirpath}")
        return True
    else:
        print(f"✗ {description}: {dirpath} (NOT FOUND)")
        return False

def main():
    print("Game24 Evaluation Setup Validation")
    print("=" * 50)
    
    all_good = True
    
    # Check required directories
    all_good &= check_directory_exists("models/policy/llama2-7b-game24-policy-hf", "Policy Model Directory")
    all_good &= check_directory_exists("models/value/llama2-7b-game24-value", "Value Model Directory") 
    all_good &= check_directory_exists("models/ct2_cache/llama2-7b-game24-policy-ct2", "CT2 Cache Directory")
    
    # Check key model files
    all_good &= check_file_exists("models/policy/llama2-7b-game24-policy-hf/config.json", "Policy Config")
    all_good &= check_file_exists("models/value/llama2-7b-game24-value/config.json", "Value Config")
    all_good &= check_file_exists("models/ct2_cache/llama2-7b-game24-policy-ct2/config.json", "CT2 Config")
    
    # Check scripts
    all_good &= check_file_exists("setup_game24_evaluation.sh", "Setup Script")
    all_good &= check_file_exists("evaluate_game24.sh", "Evaluation Script")
    all_good &= check_file_exists("format_kaggle_submission.py", "Submission Formatter")
    
    # Check data files
    all_good &= check_file_exists("tsllm/envs/game24/24.csv", "Game24 Test Dataset")
    all_good &= check_file_exists("tsllm/envs/game24/env.py", "Game24 Environment")
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("✓ All checks passed! You're ready to run evaluation.")
        print("\nNext steps:")
        print("1. Run: ./evaluate_game24.sh")
        print("2. After evaluation completes, run: python format_kaggle_submission.py")
    else:
        print("✗ Some files are missing. Please run ./setup_game24_evaluation.sh first.")
        return 1
    
    # Test import of key modules
    print("\nTesting Python imports...")
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed")
        all_good = False
    
    try:
        import ctranslate2
        print(f"✓ CTranslate2: {ctranslate2.__version__}")
    except ImportError:
        print("✗ CTranslate2 not installed")
        all_good = False
    
    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not installed")
        all_good = False
    
    try:
        from tsllm.envs.game24.env import Game24Env
        print("✓ Game24 environment can be imported")
    except ImportError as e:
        print(f"✗ Cannot import Game24 environment: {e}")
        all_good = False
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("✓ All validation checks passed!")
        
        # Show a quick test of the Game24 environment
        try:
            from tsllm.envs.game24.env import Game24Env, judge_correct
            
            # Test the judge function
            test_problem = "1 1 4 6"
            test_answer = "(6 - 4) * (1 + 1)"
            is_correct = judge_correct(test_problem, None, test_answer)
            
            print(f"\nGame24 Environment Test:")
            print(f"Problem: {test_problem}")
            print(f"Answer: {test_answer}")
            print(f"Correct: {is_correct}")
            
            if is_correct:
                print("✓ Game24 environment is working correctly!")
            else:
                print("✗ Game24 environment test failed")
                
        except Exception as e:
            print(f"✗ Error testing Game24 environment: {e}")
    else:
        print("✗ Validation failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())