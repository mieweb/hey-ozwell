#!/usr/bin/env python3
"""
Simple test to verify the model training pipeline components.
Tests basic functionality without requiring full datasets.
"""

import sys
import os
from pathlib import Path

# Add tools directory to path
tools_dir = Path(__file__).parent / 'tools'
sys.path.append(str(tools_dir))

# Import our modules
from prepare_data import DataPreparer
import tempfile
import json

def test_data_preparation():
    """Test the data preparation pipeline"""
    print("Testing data preparation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize data preparer
        preparer = DataPreparer(data_dir=temp_dir, sample_rate=16000)
        
        # Test with small dataset
        phrase = 'hey-ozwell'
        preparer.collect_samples(phrase, positive_count=5, negative_count=10)
        
        # Check if files were created
        phrase_dir = Path(temp_dir) / phrase
        positive_files = list((phrase_dir / 'positive').glob('*.wav'))
        negative_files = list((phrase_dir / 'negative').glob('*.wav'))
        
        assert len(positive_files) == 5, f"Expected 5 positive files, got {len(positive_files)}"
        assert len(negative_files) == 10, f"Expected 10 negative files, got {len(negative_files)}"
        
        # Check metadata
        metadata_file = phrase_dir / 'metadata.json'
        assert metadata_file.exists(), "Metadata file not created"
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        assert metadata['phrase'] == phrase
        assert metadata['positive_count'] == 5
        assert metadata['negative_count'] == 10
        
        print("✓ Data preparation test passed")

def test_audio_processing():
    """Test basic audio processing functions"""
    print("Testing audio processing...")
    
    try:
        import numpy as np
        import librosa
        
        # Generate test audio
        duration = 2.0
        sample_rate = 16000
        samples = int(duration * sample_rate)
        
        # Simple sine wave
        t = np.linspace(0, duration, samples)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        # Test mel-spectrogram conversion
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sample_rate,
            n_mels=80,
            hop_length=512,
            n_fft=2048
        )
        
        assert mel_spec.shape[0] == 80, f"Expected 80 mel bands, got {mel_spec.shape[0]}"
        
        print("✓ Audio processing test passed")
        
    except ImportError as e:
        print(f"⚠ Audio processing test skipped (missing dependencies: {e})")

def test_javascript_sdk():
    """Test JavaScript SDK basic structure"""
    print("Testing JavaScript SDK structure...")
    
    # Check if all main files exist
    js_dir = Path(__file__).parent / 'prod' / 'js'
    
    required_files = [
        'src/index.js',
        'src/WakeListener.js',
        'src/ModelManager.js',
        'src/RingBufferRecorder.js',
        'src/AudioProcessor.js',
        'package.json',
        'examples/basic/index.html'
    ]
    
    for file_path in required_files:
        full_path = js_dir / file_path
        assert full_path.exists(), f"Required file missing: {file_path}"
    
    print("✓ JavaScript SDK structure test passed")

def main():
    """Run all tests"""
    print("Running Hey Ozwell component tests...\n")
    
    try:
        test_data_preparation()
        test_audio_processing()
        test_javascript_sdk()
        
        print("\n🎉 All tests passed!")
        print("\nNext steps:")
        print("1. Install Python dependencies: pip install -r model/requirements.txt")
        print("2. Install JS dependencies: cd prod/js && npm install")
        print("3. Train models: cd model/tools && python train_all.py")
        print("4. Test browser SDK: cd prod/js/examples/basic && python -m http.server 8000")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()