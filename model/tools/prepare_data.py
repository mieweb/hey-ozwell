#!/usr/bin/env python3
"""
Data preparation script for Hey Ozwell wake-word models.
Collects and prepares training datasets for the four wake phrases.
"""

import os
import argparse
import simplejson as json
import random
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from elevenlabs import ElevenLabs
from dotenv import load_dotenv
import pandas as pd
import re
import string

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreparer:
    """Prepares training data for wake-word detection models"""
    
    WAKE_PHRASES = {
        'hey-ozwell': ['hey ozwell', 'hey oswell', 'hay ozwell'],
        "ozwell-i'm-done": ['ozwell im done', 'ozwell i am done', 'oswell im done', 'ozwell done'],
        'go-ozwell': ['go ozwell', 'go oswell'],
        'ozwell-go': ['ozwell go', 'oswell go']
    }

    NEGATIVE_PHRASES = {
        'hey-ozwell': ['hey oswald', 'hey amal', 'hey paul', 'nay ozwell', 'he is well'],
        "ozwell-i'm-done": ["oswald i'm done", "ozwell i'm not done", 'ozwell is fun', "oh swell i'm done"],
        'go-ozwell': ['go oswald', 'no ozwell', 'so ozwell', 'show ozwell', 'go amal', 'go call', 'gauze well', 'go with the flow'],
        'ozwell-go': ['oswald go', 'ozwell no', 'ozwell show', 'ozwell dont', 'ozwell knows', 'is it slow', 'was mellow', "oh we'll know", "'cause we'll know"]
    }
    
    def __init__(self, data_dir: str = '../data', wake_phrases_csv: str | None = None, negative_phrases_csv: str | None = None, sample_rate: int = 16000):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        
        # Create directory structure
        for phrase in self.WAKE_PHRASES.keys():
            phrase_dir = self.data_dir / phrase
            for split in ['train/positive', 'train/negative', 'test/positive', 'test/negative']:
                (phrase_dir / split).mkdir(parents=True, exist_ok=True)

        # If passed a path to a csv file, add the positive variants to WAKE_PHRASES
        if wake_phrases_csv:
            df = pd.read_csv(wake_phrases_csv)
            for group_name, group in df.groupby('wake_phrase'):
                key = group_name.replace(' ', '-')
                positive_phrases = self.WAKE_PHRASES.get(key, [])
                positive_phrases.extend(group['positive_phrase'].to_list())
                positive_phrases = list(set(positive_phrases))
                self.WAKE_PHRASES[key] = positive_phrases

        # If passed a path to a csv file, add the negative pharse to NEGATIVE_PHRASES
        if negative_phrases_csv:
            df = pd.read_csv(negative_phrases_csv)
            for group_name, group in df.groupby('wake_phrase'):
                key = group_name.replace(' ', '-')
                negative_phrases = self.NEGATIVE_PHRASES.get(key, [])
                negative_phrases.extend(group['negative_phrase'].to_list())
                negative_phrases = list(set(negative_phrases))
                self.NEGATIVE_PHRASES[key] = negative_phrases

        # Initialize ElevenLabs client
        self.client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))
        voices_response = self.client.voices.get_all()
        self.voices = voices_response.voices
    

    def collect_samples(self, phrase: str, samples: dict, reset_idx: bool = False):
        """
        Collect training samples for a specific phrase
        
        Args:
            phrase: Target wake phrase
            samples: Dictionary where keys are a Tuple (split, sample_type) and values are the number of samples to generate
            reset_idx: if True will begin naming files starting from 1 after the maximum existing idx, otherwise starts from 0
        """

        # assert samples.keys()
        logger.info(f"Collecting samples for phrase: {phrase}")
        if phrase not in self.WAKE_PHRASES:
            raise ValueError(f"Unknown phrase: {phrase}")
                
        phrase_dir = self.data_dir / phrase
        manifest = self.get_manifest(phrase)

        for (split, sample_type), num_samples in samples.items():
            label = int(sample_type == "positive")
            logger.info(f"Generating {num_samples} {sample_type} samples ({split})...")
            start_idx = 0 if (reset_idx or (len(os.listdir(phrase_dir / split / sample_type)) == 0)) else max(list(map(lambda x: int(re.match(r'(\d+)_.*\.wav', x)[1]), os.listdir(phrase_dir / split / sample_type)))) + 1
            for i in range(start_idx, start_idx + num_samples):
                try:
                    selected_phrase = self.select_phrase(phrase, label==1)
                    voice = random.choice(self.voices)
                    audio = self.client.text_to_speech.convert(
                        voice_id=voice.voice_id,
                        model_id="eleven_multilingual_v2",
                        text=selected_phrase
                    )
                    filename = phrase_dir / split / sample_type / f'{i:04d}_{re.sub(r' +', '_', re.sub(r' - .*', '', voice.name)).strip('_')}.wav'
                    # save audio to file
                    with open(filename, "wb") as f:
                        for chunk in audio:
                            if chunk:
                                f.write(chunk)
                    manifest[split][f"{sample_type}_samples"].append({"file": str(filename), "label": label, "phrase": selected_phrase, "augmented": None, "voice": voice.name})
                except:
                    logger.error(f'Problem rose during generation of {filename} ({split})...saving manifest')
                    self.save_manifest(phrase, manifest)
                    return

        self.save_manifest(phrase, manifest)
        logger.info(f"Data collection complete for {phrase}")

    
    def augment_data(self, phrase: str, augmentation_factor: int = 3):
        """
        Apply data augmentation to existing samples
        
        Args:
            phrase: Target phrase to augment
            augmentation_factor: How many augmented versions to create per original
        """
        logger.info(f"Augmenting data for phrase: {phrase}")
                
        manifest = self.get_manifest(phrase)

        for split in ['train', 'test']:
            for sample_type in ['positive', 'negative']:
                df = pd.DataFrame(manifest[split][f'{sample_type}_samples'])
                if df.empty:
                    continue
                orig = df[df['augmented'].isnull()]
                orig['existing_augs'] = orig['file'].map(lambda x: (df['augmented'] == x).sum())
                orig = orig[orig['existing_augs'].lt(augmentation_factor)]
                if orig.empty:
                    continue
                audio = orig['file'].map(lambda x: librosa.load(Path(x), sr=self.sample_rate)[0])
                audio.name = 'audio'
                for aug_id in range(min(orig['existing_augs']), augmentation_factor):
                    try:
                        filter = orig['existing_augs'].le(aug_id)
                        if not filter.any():
                            continue
                        aug_audio = audio.loc[filter].map(self._apply_augmentation)
                        aug_df = orig[filter].drop(columns=['existing_augs']).copy()
                        aug_df['augmented'] = aug_df['file']
                        aug_df['file'] = aug_df['file'].str.replace('.wav', f'_aug_{aug_id}.wav')

                        # Save augmented audio
                        pd.concat((aug_df['file'], aug_audio), axis=1).apply(lambda row: sf.write(row['file'], row['audio'], self.sample_rate), axis=1)
                        manifest[split][f'{sample_type}_samples'].extend(aug_df.to_dict(orient='records'))
                    except:
                        logger.error(f'Problem augmeting {aug_id/augmentation_factor} {sample_type} samples for {split}...saving_manifest')
                        self.save_manifest(phrase, manifest)
        self.save_manifest(phrase, manifest)
        logger.info(f"Augmentation complete.")
    
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
    
    def get_sample_counts(self, phrase: str):
        manifest = self.get_manifest(phrase)
        counts = {}
        for split in  ['train', 'test'] :
            for sample_type in ['positive', 'negative']:
                df = pd.DataFrame(manifest[split][f'{sample_type}_samples'])
                counts[(split, sample_type, 'orig')] = len(df[df['augmented'].isnull()])  if len(df) > 0 else 0
                counts[(split, sample_type, 'aug')] = len(df[~df['augmented'].isnull()]) if len(df) > 0 else 0
        return counts
    
    def get_manifest(self, phrase_key: str):
        """
        Get training manifest for a specific phrase

        phrase_key: phrase to get manifest for
        """
        manifest_path = self.data_dir / phrase_key / "training_manifest.json"
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as file:
                manifest = json.load(file)
        else:
            manifest = {'train': {'positive_samples': [], 'negative_samples': []}, 'test': {'positive_samples': [], 'negative_samples': []}}
        return manifest
    
    def save_manifest(self, phrase_key: str, manifest: dict):
        """
        Save training manifest for a specific phrase

        phrase_key: phrase to save manifest to
        """
        manifest_path = self.data_dir / phrase_key / "training_manifest.json"
        with open(manifest_path, 'w') as file:
            json.dump(manifest, file, indent=4, ignore_nan=True)
    
    def select_phrase(self, phrase_key: str, positive: bool):
        if positive:
            return phrase_key.replace('-', ' ')
        else:
            return random.choice(self.NEGATIVE_PHRASES[phrase_key])


def main():
    parser = argparse.ArgumentParser(description='Prepare training data for Hey Ozwell wake-word models')
    parser.add_argument('--phrase', required=True, choices=['hey-ozwell', "ozwell-i'm-done", 'go-ozwell', 'ozwell-go'],
                       help='Wake phrase to prepare data for')
    parser.add_argument('--positive-samples', type=int, default=500,
                       help='Number of positive samples to generate')
    parser.add_argument('--negative-samples', type=int, default=2000,
                       help='Number of negative samples to generate')
    parser.add_argument('--test-split-factor', type=float, default=0.2, help='Fraction of samples to split for testing')
    parser.add_argument('--augment', action='store_true',
                       help='Apply data augmentation after collection')
    parser.add_argument('--augment-factor', type=int, default=3,
                       help='Augmentation factor (number of variations per sample)')
    parser.add_argument('--data-dir', default='../data',
                       help='Directory to store training data')
    
    
    args = parser.parse_args()

    os.makedirs('../logs/data_prep', exist_ok=True)
    handler = logging.FileHandler(f'../logs/data_prep/{args.phrase}.log', 'w')
    logger.addHandler(handler)
    
    # Initialize data preparer
    preparer = DataPreparer(data_dir=args.data_dir, negative_phrases_csv='../negative_phrases.csv')
    existing_sample_counts = preparer.get_sample_counts(args.phrase)

    assert args.test_split_factor >= 0.0 and args.test_split_factor <= 1.0, logger.error('Test split factor must be in range 0.0 and 1.0')
    generate_samples = {}
    for split in ['train', 'test']:
        split_factor = (1.0 - args.test_split_factor) if split == 'train' else args.test_split_factor
        for sample_type in ['positive', 'negative']:
            num_samples = args.positive_samples if (sample_type == 'positive') else args.negative_samples
            generate_samples[split, sample_type] = max(0, round(num_samples * split_factor) - existing_sample_counts[split, sample_type, 'orig'])

    # Collect samples
    preparer.collect_samples(
        phrase=args.phrase,
        samples=generate_samples,
        reset_idx=False
    )
    
    # Apply augmentation if requested
    if args.augment:
        preparer.augment_data(args.phrase, args.augment_factor)
    
    logger.info("Data preparation complete!")
    logger.info(f"File counts: {preparer.get_sample_counts(args.phrase)}")


if __name__ == '__main__':
    main()
    

