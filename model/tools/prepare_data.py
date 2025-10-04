#!/usr/bin/env python3
"""
Data preparation script for Hey Ozwell wake-word models.
Collects and prepares training datasets for the four wake phrases.
"""

import os
import argparse
import json
import random
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import List, Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreparer:
    """Prepares training data for wake-word detection models"""
    
    WAKE_PHRASES = {
        'hey-ozwell': ['hey ozwell', 'hey oswell', 'hay ozwell'],
        'im-done': ['ozwell im done', 'ozwell i am done', 'oswell im done', 'ozwell done'],
        'go-ozwell': ['go ozwell', 'go oswell'],
        'ozwell-go': ['ozwell go', 'oswell go']
    }
    
    def __init__(self, data_dir: str = '../data', sample_rate: int = 16000):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        
        # Create directory structure
        for phrase in self.WAKE_PHRASES.keys():
            phrase_dir = self.data_dir / phrase
            for split in ['positive', 'negative', 'test']:
                (phrase_dir / split).mkdir(parents=True, exist_ok=True)
    
    def collect_samples(self, phrase: str, positive_count: int = 500, negative_count: int = 2000):
        """
        Collect training samples for a specific phrase
        
        Args:
            phrase: Target wake phrase
            positive_count: Number of positive samples to collect
            negative_count: Number of negative samples to collect
        """
        logger.info(f"Collecting samples for phrase: {phrase}")
        
        if phrase not in self.WAKE_PHRASES:
            raise ValueError(f"Unknown phrase: {phrase}")
        
        # For now, create placeholder files since we don't have real audio data
        # In production, this would integrate with recording tools or existing datasets
        
        phrase_dir = self.data_dir / phrase
        
        # Generate positive samples (placeholder)
        logger.info(f"Generating {positive_count} positive samples...")
        for i in range(positive_count):
            audio_data = self._generate_placeholder_audio(phrase, positive=True)
            filename = phrase_dir / 'positive' / f'{phrase}_{i:04d}.wav'
            sf.write(filename, audio_data, self.sample_rate)
        
        # Generate negative samples (placeholder)
        logger.info(f"Generating {negative_count} negative samples...")
        for i in range(negative_count):
            audio_data = self._generate_placeholder_audio(phrase, positive=False)
            filename = phrase_dir / 'negative' / f'negative_{i:04d}.wav'
            sf.write(filename, audio_data, self.sample_rate)
        
        # Split some samples for testing
        test_count = min(50, positive_count // 10)
        logger.info(f"Creating {test_count} test samples...")
        for i in range(test_count):
            audio_data = self._generate_placeholder_audio(phrase, positive=True)
            filename = phrase_dir / 'test' / f'test_{i:04d}.wav'
            sf.write(filename, audio_data, self.sample_rate)
        
        # Create metadata file
        metadata = {
            'phrase': phrase,
            'variations': self.WAKE_PHRASES[phrase],
            'positive_count': positive_count,
            'negative_count': negative_count,
            'test_count': test_count,
            'sample_rate': self.sample_rate,
            'duration_range': [0.5, 3.0],  # seconds
        }
        
        metadata_file = phrase_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Data collection complete for {phrase}")
    
    def _generate_placeholder_audio(self, phrase: str, positive: bool = True, duration: float = None) -> np.ndarray:
        """
        Generate placeholder audio data for demonstration
        In production, this would be replaced with real audio recordings
        """
        if duration is None:
            duration = random.uniform(0.5, 3.0)
        
        samples = int(duration * self.sample_rate)
        
        if positive:
            # Generate speech-like signal with some structure
            t = np.linspace(0, duration, samples)
            
            # Create formant-like frequencies for speech simulation
            f1 = 300 + 200 * np.sin(2 * np.pi * 2 * t)  # First formant
            f2 = 1200 + 800 * np.sin(2 * np.pi * 3 * t)  # Second formant
            
            signal = (0.3 * np.sin(2 * np.pi * f1 * t) + 
                     0.2 * np.sin(2 * np.pi * f2 * t))
            
            # Add envelope to simulate speech timing
            envelope = np.exp(-3 * (t - duration/2)**2 / duration**2)
            signal *= envelope
            
        else:
            # Generate noise or non-speech audio
            if random.random() < 0.5:
                # White noise
                signal = 0.1 * np.random.randn(samples)
            else:
                # Tonal noise (not speech-like)
                t = np.linspace(0, duration, samples)
                freq = random.uniform(100, 8000)
                signal = 0.2 * np.sin(2 * np.pi * freq * t)
        
        # Add some background noise
        noise = 0.01 * np.random.randn(samples)
        signal += noise
        
        # Ensure signal is in valid range
        signal = np.clip(signal, -0.95, 0.95)
        
        return signal.astype(np.float32)
    
    def augment_data(self, phrase: str, augmentation_factor: int = 3):
        """
        Apply data augmentation to existing samples
        
        Args:
            phrase: Target phrase to augment
            augmentation_factor: How many augmented versions to create per original
        """
        logger.info(f"Augmenting data for phrase: {phrase}")
        
        phrase_dir = self.data_dir / phrase
        positive_dir = phrase_dir / 'positive'
        
        # Get list of original files
        original_files = list(positive_dir.glob('*.wav'))
        
        for original_file in original_files:
            audio, sr = librosa.load(original_file, sr=self.sample_rate)
            
            for aug_idx in range(augmentation_factor):
                # Apply random augmentations
                augmented = self._apply_augmentation(audio)
                
                # Save augmented file
                aug_filename = positive_dir / f'{original_file.stem}_aug_{aug_idx}.wav'
                sf.write(aug_filename, augmented, self.sample_rate)
        
        logger.info(f"Augmentation complete. Created {len(original_files) * augmentation_factor} additional samples")
    
    def _apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """Apply random augmentation to audio sample"""
        
        # Speed/pitch variation
        if random.random() < 0.7:
            speed_factor = random.uniform(0.8, 1.2)
            audio = librosa.effects.time_stretch(audio, rate=speed_factor)
        
        # Pitch shifting
        if random.random() < 0.5:
            pitch_shift = random.uniform(-2, 2)  # semitones
            audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=pitch_shift)
        
        # Add noise
        if random.random() < 0.6:
            noise_level = random.uniform(0.001, 0.01)
            noise = noise_level * np.random.randn(len(audio))
            audio = audio + noise
        
        # Normalize
        audio = audio / max(abs(audio.max()), abs(audio.min()), 1e-7)
        
        return audio.astype(np.float32)
    
    def create_training_manifest(self, phrase: str):
        """Create training manifest file for the phrase"""
        phrase_dir = self.data_dir / phrase
        
        manifest = {
            'positive_samples': [],
            'negative_samples': [],
            'test_samples': []
        }
        
        # Collect positive samples
        positive_dir = phrase_dir / 'positive'
        for audio_file in positive_dir.glob('*.wav'):
            manifest['positive_samples'].append({
                'file': str(audio_file.relative_to(self.data_dir)),
                'label': 1,
                'phrase': phrase
            })
        
        # Collect negative samples
        negative_dir = phrase_dir / 'negative'
        for audio_file in negative_dir.glob('*.wav'):
            manifest['negative_samples'].append({
                'file': str(audio_file.relative_to(self.data_dir)),
                'label': 0,
                'phrase': 'negative'
            })
        
        # Collect test samples
        test_dir = phrase_dir / 'test'
        for audio_file in test_dir.glob('*.wav'):
            manifest['test_samples'].append({
                'file': str(audio_file.relative_to(self.data_dir)),
                'label': 1,
                'phrase': phrase
            })
        
        # Save manifest
        manifest_file = phrase_dir / 'training_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Training manifest created: {manifest_file}")
        logger.info(f"  Positive: {len(manifest['positive_samples'])}")
        logger.info(f"  Negative: {len(manifest['negative_samples'])}")
        logger.info(f"  Test: {len(manifest['test_samples'])}")


def main():
    parser = argparse.ArgumentParser(description='Prepare training data for Hey Ozwell wake-word models')
    parser.add_argument('--phrase', required=True, choices=['hey-ozwell', 'im-done', 'go-ozwell', 'ozwell-go'],
                       help='Wake phrase to prepare data for')
    parser.add_argument('--positive-samples', type=int, default=500,
                       help='Number of positive samples to generate')
    parser.add_argument('--negative-samples', type=int, default=2000,
                       help='Number of negative samples to generate')
    parser.add_argument('--augment', action='store_true',
                       help='Apply data augmentation after collection')
    parser.add_argument('--augment-factor', type=int, default=3,
                       help='Augmentation factor (number of variations per sample)')
    parser.add_argument('--data-dir', default='../data',
                       help='Directory to store training data')
    
    args = parser.parse_args()
    
    # Initialize data preparer
    preparer = DataPreparer(data_dir=args.data_dir)
    
    # Collect samples
    preparer.collect_samples(
        phrase=args.phrase,
        positive_count=args.positive_samples,
        negative_count=args.negative_samples
    )
    
    # Apply augmentation if requested
    if args.augment:
        preparer.augment_data(args.phrase, args.augment_factor)
    
    # Create training manifest
    preparer.create_training_manifest(args.phrase)
    
    logger.info("Data preparation complete!")


if __name__ == '__main__':
    main()