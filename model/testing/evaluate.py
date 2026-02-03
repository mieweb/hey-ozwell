#!/usr/bin/env python3
"""
Evaluation script for Hey Ozwell wake-word models.
Tests accuracy, false positive rate, and latency metrics.
"""

import os
import simplejson as json
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates wake-word detection models"""
    
    def __init__(self, model_path: str, sample_rate: int = 16000, n_mels: int = 80):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_duration = 3.0
        self.max_samples = int(self.max_duration * sample_rate)
        
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        logger.info(f"Loaded ONNX model: {model_path}")
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """Preprocess audio file for inference"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Pad or truncate to fixed length
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        else:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))
        
        # Convert to mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=512,
            n_fft=2048
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        
        # Add batch dimension
        return log_mel.reshape(1, 1, *log_mel.shape).astype(np.float32)
    
    def predict(self, features: np.ndarray, return_confidence: bool = True) -> Tuple[int, float]:
        """Run inference on preprocessed features"""
        start_time = time.time()
        
        outputs = self.session.run([self.output_name], {self.input_name: features})
        logits = outputs[0][0]  # Remove batch dimension
        
        inference_time = time.time() - start_time
        
        # Convert to probabilities
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        prediction = np.argmax(probabilities)
        confidence = probabilities[prediction]
        
        if return_confidence:
            return prediction, confidence, inference_time
        else:
            return prediction, inference_time
    
    def evaluate_test_set(self, manifest: dict, phrase: str) -> Dict:
        """Evaluate model on test dataset"""
        
    
        df = pd.DataFrame(manifest['test']['positive_samples'] + manifest['test']['negative_samples'])
        true_labels = df['label'].to_list()
        pred_tuple = df['file'].map(self.preprocess_audio).map(self.predict)
        predictions, confidences, inference_times = pred_tuple.str[0].to_list(), pred_tuple.str[1].to_list(), pred_tuple.str[2].to_list()

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # False positive rate (important for wake-word detection)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Average inference time
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        
        results = {
            'phrase': phrase,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': fpr,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'avg_inference_time_ms': avg_inference_time,
            'total_samples': len(predictions),
            'confusion_matrix': cm.tolist()
        }
        
        return results
    
    def test_false_positive_rate(self, manifest: dict, duration_hours: float = 1.0) -> float:
        """Test false positive rate over extended audio"""
        logger.info(f"Testing false positive rate over {duration_hours} hours of audio")
        
        audio_files = pd.DataFrame(manifest['test']['negative_samples'])['file'].to_list()
        
        if not audio_files:
            logger.warning("No negative audio files found")
            return 0.0
        
        total_detections = 0
        total_audio_duration = 0
        
        # Process files until we reach target duration
        for audio_file in audio_files:
            if total_audio_duration >= duration_hours * 3600:
                break
            
            # Get audio duration
            info = sf.info(audio_file)
            audio_duration = info.frames / info.samplerate
            total_audio_duration += audio_duration
            
            # Test in sliding windows
            window_size = 3.0  # seconds
            hop_size = 1.0     # seconds
            
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            num_windows = int((len(audio) / self.sample_rate - window_size) / hop_size) + 1
            
            for i in range(num_windows):
                start_sample = int(i * hop_size * self.sample_rate)
                end_sample = start_sample + int(window_size * self.sample_rate)
                
                if end_sample > len(audio):
                    break
                
                window_audio = audio[start_sample:end_sample]
                
                # Pad if necessary
                if len(window_audio) < self.max_samples:
                    window_audio = np.pad(window_audio, (0, self.max_samples - len(window_audio)))
                
                # Convert to mel-spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=window_audio, 
                    sr=self.sample_rate,
                    n_mels=self.n_mels,
                    hop_length=512,
                    n_fft=2048
                )
                log_mel = librosa.power_to_db(mel_spec, ref=np.max)
                log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
                features = log_mel.reshape(1, 1, *log_mel.shape).astype(np.float32)
                
                # Get prediction
                pred, conf, _ = self.predict(features)
                
                if pred == 1:  # Wake word detected
                    total_detections += 1
        
        # Calculate false positives per hour
        fp_per_hour = total_detections / (total_audio_duration / 3600)
        
        logger.info(f"False positive rate: {fp_per_hour:.2f} per hour")
        logger.info(f"Total audio processed: {total_audio_duration/3600:.2f} hours")
        logger.info(f"Total false positives: {total_detections}")
        
        return fp_per_hour
    
    def generate_report(self, results: Dict, output_dir: str):
        """Generate evaluation report with visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results as JSON
        results_file = output_path / f"{results['phrase']}_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create confusion matrix visualization
        plt.figure(figsize=(8, 6))
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {results["phrase"]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path / f"{results['phrase']}_confusion_matrix.png", dpi=300)
        plt.close()
        
        # Generate text report
        report_file = output_path / f"{results['phrase']}_evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(f"Evaluation Report for {results['phrase']}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Overall Performance:\n")
            f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall: {results['recall']:.4f}\n")
            f.write(f"  F1 Score: {results['f1_score']:.4f}\n\n")
            
            f.write("Detection Details:\n")
            f.write(f"  True Positives: {results['true_positives']}\n")
            f.write(f"  True Negatives: {results['true_negatives']}\n")
            f.write(f"  False Positives: {results['false_positives']}\n")
            f.write(f"  False Negatives: {results['false_negatives']}\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"  False Positive Rate: {results['false_positive_rate']:.4f}\n")
            f.write(f"  Average Inference Time: {results['avg_inference_time_ms']:.2f} ms\n")
            f.write(f"  Total Test Samples: {results['total_samples']}\n")
        
        logger.info(f"Evaluation report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Hey Ozwell wake-word model')
    parser.add_argument('--model', required=True,
                       help='Path to ONNX model file')
    parser.add_argument('--test-data', required=True,
                       help='Path to test data directory')
    parser.add_argument('--phrase', required=True,
                       choices=['hey-ozwell', "ozwell-i'm-done", 'go-ozwell', 'ozwell-go'],
                       help='Wake phrase being evaluated')
    parser.add_argument('--output-dir', default='../results',
                       help='Directory to save evaluation results')
    parser.add_argument('--fp-test', action='store_true',
                       help='Run extended false positive rate test')
    parser.add_argument('--fp-duration', type=float, default=1.0,
                       help='Duration in hours for false positive test')
    
    args = parser.parse_args()
    os.makedirs('../logs/testing', exist_ok=True)
    handler = logging.FileHandler(f'../logs/testing/{args.phrase}.log', 'w')
    logger.addHandler(handler)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model)

    test_dir = Path(args.test_data)
    with open(test_dir.parent / 'training_manifest.json', 'r') as f:
        manifest = json.load(f)
    
    # Run main evaluation
    results = evaluator.evaluate_test_set(manifest, args.phrase)
    
    # Print results
    logger.info("Evaluation Results:")
    logger.info(f"  Accuracy: {results['accuracy']:.4f}")
    logger.info(f"  Precision: {results['precision']:.4f}")
    logger.info(f"  Recall: {results['recall']:.4f}")
    logger.info(f"  F1 Score: {results['f1_score']:.4f}")
    logger.info(f"  False Positive Rate: {results['false_positive_rate']:.4f}")
    logger.info(f"  Avg Inference Time: {results['avg_inference_time_ms']:.2f} ms")
    
    # Run extended false positive test if requested
    if args.fp_test:
        fp_rate = evaluator.test_false_positive_rate(manifest, args.fp_duration)
        results['extended_fp_rate_per_hour'] = fp_rate
    
    # Generate report
    evaluator.generate_report(results, args.output_dir)
    
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
