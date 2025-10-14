import os
import random
import argparse

from elevenlabs import ElevenLabs
import numpy as np
from pathlib import Path
import librosa
import soundfile as sf
import logging
from dotenv import load_dotenv

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
    
    def __init__(self, data_dir: str = '../data', sample_rate: int = 16000):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        
        # Create directory structure
        for phrase in self.WAKE_PHRASES.keys():
            phrase_dir = self.data_dir / phrase
            for split in ['positive', 'negative', 'test']:
                (phrase_dir / split).mkdir(parents=True, exist_ok=True)
            (phrase_dir / 'test' / 'positive').mkdir(parents=True, exist_ok=True)
            (phrase_dir / 'test' / 'negative').mkdir(parents=True, exist_ok=True)

    def collect_samples(self, samples: int = 30):
            # Initialize ElevenLabs client
        client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))
        voices_response = client.voices.get_all()
        voices = voices_response.voices

        # Define target phrases
        phrases = ["Hey Ozwell", "Ozwell I'm done", "Go Ozwell", "Ozwell go"]

        # Generate samples
        for phrase in phrases:
            # Convert phrase to match the WAKE_PHRASES key format
            if phrase == "Ozwell I'm done":
                phrase_key = "ozwell-i'm-done"
            else:
                phrase_key = phrase.lower().replace(' ', '-')
            phrase_dir = self.data_dir / phrase_key / 'positive'
            phrase_dir.mkdir(parents=True, exist_ok=True)

            for i in range(samples):
                voice = random.choice(voices)
                print(f"Generating sample {i+1}/{samples} for phrase '{phrase}' using voice '{voice.name}'...")
                audio = client.text_to_speech.convert(
                    voice_id=voice.voice_id,
                    model_id="eleven_multilingual_v2",
                    text=phrase
                )
                file_path = phrase_dir / f"{i:03d}_{voice.name}.wav"
                with open(file_path, "wb") as f:
                    for chunk in audio:
                        if chunk:
                            f.write(chunk)
        

    def augment_data(self, phrase: str, augmentation_factor: int = 3):
        """
        Apply data augmentation to existing samples
        
        Args:
            phrase: Target phrase to augment
            augmentation_factor: How many augmented versions to create per original
        """
        logger.info(f"Augmenting data for phrase: {phrase}")
        
        phrase_dir = self.data_dir / phrase
        print(phrase_dir)
        positive_dir = phrase_dir / 'positive'
        
        # Get list of original files (exclude already augmented files)
        all_files = list(positive_dir.rglob('*.wav'))
        print(all_files)
        original_files = [f for f in all_files if '_aug_' not in f.name]
        
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

def main():
    # Parse command-line arguments
    load_dotenv()  # Load environment variables from .env file
    parser = argparse.ArgumentParser(description="Generate Ozwell wake word audio samples using ElevenLabs API.")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples per phrase (default: 30)")
    parser.add_argument('--augment', action='store_true',
                       help='Apply data augmentation after collection')
    parser.add_argument('--augment-factor', type=int, default=3,
                       help='Augmentation factor (number of variations per sample)')
    parser.add_argument('--data-dir', default='../data',
                       help='Directory to store training data')

    args = parser.parse_args()
    samples = args.samples

        # Initialize data preparer
    preparer = DataPreparer(data_dir=args.data_dir)
    
    # Collect samples
    preparer.collect_samples(
        samples=samples
    )


    if args.augment:
        # Augment data for all phrases that were collected
        for phrase_key in preparer.WAKE_PHRASES.keys():
            preparer.augment_data(phrase_key, args.augment_factor)

if __name__ == "__main__":
    main()