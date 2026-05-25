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


class DataPreparer:
    """Prepares training data for wake-word detection models"""
    def __init__(self, phrase: str, negative_phrases: list, data_dir: str = '../data', sample_rate: int = 16000):
        self.phrase = phrase.lower()
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.phrase_dir = self.data_dir / self.phrase.replace(' ','-')
        self.negative_phrases = negative_phrases
        
        # Create directory structure
        for split in ['train/positive', 'train/negative', 'test/positive', 'test/negative']:
            (self.phrase_dir / split).mkdir(parents=True, exist_ok=True)

        # Initialize ElevenLabs client
        self.client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))
        voices_response = self.client.voices.get_all()
        self.voices = voices_response.voices
        self.manifest = self.get_manifest()
    

    def collect_samples(self, split: str, sample_type: str, samples: int, reset_idx: bool = False):
        """
        Collect training samples for a specific phrase
        
        Args:
            split: which split to collect samples for (['train', 'test'])
            sample_type: which sample_type to generate samples for (['positive', 'negative'])
            samples: number of samples to generate
            reset_idx: if True will begin naming files starting from 1 after the maximum existing idx, otherwise starts from 0
        """

        # assert samples.keys()
        logger.info(f"Collecting {sample_type} samples for split: {split}")
        assert split in ['train', 'test']
        assert sample_type in ['positive', 'negative']
        
        logger.info(f"Generating {samples} samples...")
        start_idx = 0 if (reset_idx or (len(self.manifest[split][f'{sample_type}_samples']) == 0)) else pd.DataFrame((self.manifest[split][f'{sample_type}_samples']))['file'].map(lambda x: int(re.match(r'(\d+)_.*\.wav', x)[1])).max() + 1
        text = self.phrase.replace('-', ' ')
        label = int(sample_type == 'positive')
        for i in range(start_idx, start_idx + samples):
            if sample_type == 'negative':
                text = random.choice(self.negative_phrases)
            voice = random.choice(self.voices)
            filename = f'{i:04d}_{re.sub(r' +', '_', re.sub(r' - .*', '', voice.name)).strip('_')}.wav'

            try:
                audio = self.client.text_to_speech.convert(
                    voice_id=voice.voice_id,
                    model_id="eleven_multilingual_v2",
                    text=text
                )

                # save audio to file
                with open(self.phrase_dir / split / sample_type / filename, "wb") as f:
                    for chunk in audio:
                        if chunk:
                            f.write(chunk)
            except:
                logger.error(f'Problem occured during {sample_type} sample collection for split {split}')
                logger.error(f'Issue arose during generation of sample: {filename}')
                logger.error('Saving manifest...')
                self.save_manifest()
                logger.error('Manifest saved!')
                return
            self.manifest[split][f"{sample_type}_samples"].append({"file": filename, "label": label, "text": text, "voice": voice.name})
        self.save_manifest()
        logger.info(f"Done collecting {sample_type} samples for {split} split")

    
    def augment_data(self, split: str, sample_type: str, augmentation_factor: int = 3):
        """
        Apply data augmentation to existing samples
        
        Args:
            split: which split to augment samples for (['train', 'test'])
            sample_type: which sample_type to augment samples for (['positive', 'negative'])
            augmentation_factor: How many augmented versions to create per original
        """
        logger.info(f"Augmenting {sample_type} data for split: {split}")
                

        df = pd.DataFrame(self.manifest[split][f'{sample_type}_samples'])
        if df.empty:
            logger.info(f"No {sample_type} samples to augment for split: {split}.")
            return
        orig = df[~df['file'].str.contains('_aug_')]
        orig['existing_augs'] = orig['file'].map(lambda x: (df['file'].str.contains(x.replace('.wav', '_aug_'))).sum())
        orig = orig[orig['existing_augs'].lt(augmentation_factor)]
        if orig.empty:
            return
        audio = orig['file'].map(lambda x: librosa.load(self.phrase_dir / split / sample_type / x, sr=self.sample_rate)[0])
        audio.name = 'audio'
        for aug_id in range(min(orig['existing_augs']), augmentation_factor):
            filter = orig['existing_augs'].le(aug_id)
            if not filter.any():
                continue
            aug_df = orig[filter].drop(columns=['existing_augs']).copy()
            aug_df['file'] = aug_df['file'].str.replace('.wav', f'_aug_{aug_id}.wav')
            try:
                # Apply augmentation
                aug_df['audio'] = audio.loc[filter].map(self._apply_augmentation)
                # Save augmented audio
                aug_df.apply(lambda row: sf.write(self.phrase_dir / split / sample_type / row['file'], row['audio'], self.sample_rate), axis=1)
            except:
                logger.error(f'Problem augmeting {aug_id/augmentation_factor} {sample_type} samples for {split}...saving_manifest')
                self.save_manifest()
            self.manifest[split][f'{sample_type}_samples'].extend(aug_df.drop(columns=['audio']).to_dict(orient='records'))
        self.save_manifest()
        logger.info(f"Augmentation complete.")
    
    def get_count(self, split: str, sample_type: str, key: str = 'total'):

        df = pd.DataFrame(self.manifest[split][f'{sample_type}_samples'])
        if key == 'total':
            return len(df)
        if key == 'orig':
            return  (~df['file'].str.contains('_aug_')).sum() if len(df) > 0 else 0
        if key == 'aug':
            return (df['file'].str.contains('_aug_')).sum() if len(df) > 0 else 0
    
    def get_manifest(self):
        """
        Get training manifest for a specific phrase

        phrase_key: phrase to get manifest for
        """
        manifest_path = self.phrase_dir / "training_manifest.json"
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as file:
                manifest = json.load(file)
        else:
            manifest = {'train': {'positive_samples': [], 'negative_samples': []}, 'test': {'positive_samples': [], 'negative_samples': []}}
        return manifest
    
    def save_manifest(self):
        """
        Save training manifest for a specific phrase

        phrase_key: phrase to save manifest to
        """
        manifest_path = self.phrase_dir / "training_manifest.json"
        with open(manifest_path, 'w') as file:
            json.dump(self.manifest, file, indent=4, ignore_nan=True)
    

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
    assert args.test_split_factor >= 0.0 and args.test_split_factor <= 1.0, logger.error('Test split factor must be in range 0.0 and 1.0')


    os.makedirs('../logs/data_prep', exist_ok=True)
    handler = logging.FileHandler(f'../logs/data_prep/{args.phrase}.log', 'w')
    logger.addHandler(handler)
    
    df = pd.read_csv('../negative_phrases.csv')
    negative_phrases = list(set(df['negative_phrase'][df['wake_phrase'].str.replace(' ', '-') == args.phrase].to_list() + NEGATIVE_PHRASES[args.phrase]))

    # Initialize data preparer
    preparer = DataPreparer(phrase=args.phrase, data_dir=args.data_dir, negative_phrases=negative_phrases)
    for split in ['train', 'test']:
        split_factor = (1.0 - args.test_split_factor) if split == 'train' else args.test_split_factor
        logger.info(f'Preparing data for {split} split...')
        for sample_type in ['positive', 'negative']:
            target_sample_count = round(args.positive_samples * split_factor) if (sample_type == 'positive') else round(args.negative_samples * split_factor)
            existing_sample_count = preparer.get_count(split, sample_type, 'orig')
            samples = max(0, target_sample_count = existing_sample_count)
            logger.info(f'{existing_sample_count} existing {sample_type} samples for {split} split')
            logger.info(f'{samples} samples needed to reach target of {target_sample_count}')
            # Collect samples
            preparer.collect_samples(
                split=split,
                sample_type=sample_type,
                samples=samples,
                reset_idx=False
            )
            
    # # Augment samples
    # if args.augment:
    #     existing_aug_count = preparer.get_count('train', 'positive', 'aug')
    #     preparer.augment_data('train', 'positive', args.augment_factor)
    #     logger.info(f'{existing_aug_count - preparer.get_count('train', 'positive', 'aug')} augmented positive samples created')

    #     existing_aug_count = preparer.get_count('train', 'negative', 'aug')
    #     preparer.augment_data('train', 'negative', args.augment_factor)
    #     logger.info(f'{existing_aug_count - preparer.get_count('train','negative', 'aug')} augmented negative samples created')
     
    logger.info("Data preparation complete!")
    logger.info("File counts:")
    logger.info(f"\ttrain:")
    logger.info(f"\t\tpositive: total={preparer.get_count('train', 'positive', 'total')} (orig={preparer.get_count('train', 'positive', 'orig')}, aug={preparer.get_count('train', 'positive', 'aug')})")
    logger.info(f"\t\tnegative: total={preparer.get_count('train', 'negative', 'total')} (orig={preparer.get_count('train', 'negative', 'orig')}, aug={preparer.get_count('train', 'negative', 'aug')})")
    logger.info(f"\ttest:")
    logger.info(f"\t\tpositive: total={preparer.get_count('test', 'positive', 'total')} (orig={preparer.get_count('test', 'positive', 'orig')}, aug={preparer.get_count('test', 'positive', 'aug')})")
    logger.info(f"\t\tnegative: total={preparer.get_count('test', 'negative', 'total')} (orig={preparer.get_count('test', 'negative', 'orig')}, aug={preparer.get_count('test', 'negative', 'aug')})")


if __name__ == '__main__':
    main()
