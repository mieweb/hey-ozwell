#!/usr/bin/env python3
print("inside train_all.py")
"""
Batch training script for all Hey Ozwell wake phrases.
Trains all four wake-word models in sequence.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Wake phrases to train
WAKE_PHRASES = ['hey-ozwell', 'im-done', 'go-ozwell', 'ozwell-go']

# Training configuration
TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'positive_samples': 500,
    'negative_samples': 2000,
    'augment_factor': 3
}


def run_command(cmd, cwd=None):
    """Run a shell command and return success status"""
    print(f">>> Running command: {' '.join(cmd)} in cwd={cwd}")
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        logger.info("Command completed successfully")
        print(">>> Command succeeded")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        print(f">>> Command failed: {e}")
        print(f">>> Error output: {e.stderr}")
        return False


def prepare_data_for_phrase(phrase):
    """Prepare training data for a specific phrase"""
    print(">>> In prepare_data_for_phrase()")
    logger.info(f"Preparing data for phrase: {phrase}")
    
    cmd = [
        'python', 'prepare_data.py',
        '--phrase', phrase,
        '--positive-samples', str(TRAINING_CONFIG['positive_samples']),
        '--negative-samples', str(TRAINING_CONFIG['negative_samples']),
        '--augment',
        '--augment-factor', str(TRAINING_CONFIG['augment_factor'])
    ]
    
    return run_command(cmd, cwd='tools')


def train_model_for_phrase(phrase):
    """Train model for a specific phrase"""
    print(">>> In train_model_for_phrase()")
    logger.info(f"Training model for phrase: {phrase}")
    
    output_path = f"../exports/{phrase}.onnx"
    
    cmd = [
        'python', 'train.py',
        '--phrase', phrase,
        '--output', output_path,
        '--epochs', str(TRAINING_CONFIG['epochs']),
        '--batch-size', str(TRAINING_CONFIG['batch_size']),
        '--learning-rate', str(TRAINING_CONFIG['learning_rate'])
    ]
    
    return run_command(cmd, cwd='tools')


def evaluate_model_for_phrase(phrase):
    """Evaluate trained model for a specific phrase"""
    print(">>> In evaluate_model_for_phrase()")
    logger.info(f"Evaluating model for phrase: {phrase}")
    
    model_path = f"../exports/{phrase}.onnx"
    test_data_path = f"../data/{phrase}"
    
    cmd = [
        'python', 'evaluate.py',
        '--model', model_path,
        '--test-data', test_data_path,
        '--phrase', phrase,
        '--fp-test'
    ]
    
    return run_command(cmd, cwd='testing')


def main():
    """Main training pipeline"""
    logger.info("Starting batch training for all Hey Ozwell wake phrases")
    
    # Create exports directory
    exports_dir = Path('../exports')
    exports_dir.mkdir(parents=True, exist_ok=True)
    
    # Track results
    results = {
        'data_preparation': {},
        'training': {},
        'evaluation': {}
    }
    
    for phrase in WAKE_PHRASES:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing phrase: {phrase}")
        logger.info(f"{'='*50}")
        
        # Step 1: Prepare data
        success = prepare_data_for_phrase(phrase)
        results['data_preparation'][phrase] = success
        print(f"Completed data prep for: {phrase}")
        
        if not success:
            logger.error(f"Data preparation failed for {phrase}, skipping training")
            continue
        
        # Step 2: Train model
        success = train_model_for_phrase(phrase)
        results['training'][phrase] = success
        print(f"Completed training for: {phrase}")
        
        if not success:
            logger.error(f"Training failed for {phrase}, skipping evaluation")
            continue
        
        # Step 3: Evaluate model
        success = evaluate_model_for_phrase(phrase)
        results['evaluation'][phrase] = success
        print(f"Completed evaluation for: {phrase}")
        
        if not success:
            logger.warning(f"Evaluation failed for {phrase}")
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*50}")
    
    for phrase in WAKE_PHRASES:
        logger.info(f"\n{phrase}:")
        logger.info(f"  Data preparation: {'✓' if results['data_preparation'].get(phrase) else '✗'}")
        logger.info(f"  Training: {'✓' if results['training'].get(phrase) else '✗'}")
        logger.info(f"  Evaluation: {'✓' if results['evaluation'].get(phrase) else '✗'}")
    
    # Check if all models were trained successfully
    successful_models = [phrase for phrase in WAKE_PHRASES if results['training'].get(phrase)]
    
    if len(successful_models) == len(WAKE_PHRASES):
        logger.info("\n🎉 All models trained successfully!")
        logger.info("Models are ready for deployment to the JavaScript SDK.")
        logger.info("Copy the .onnx files from exports/ to ../prod/js/models/")
    else:
        failed_models = [phrase for phrase in WAKE_PHRASES if not results['training'].get(phrase)]
        logger.warning(f"\n⚠️  Training failed for: {failed_models}")
        logger.info(f"Successfully trained: {successful_models}")


if __name__ == '__main__':
    print("inside train_all.py")
    main()