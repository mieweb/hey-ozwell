#!/usr/bin/env python3
"""
Training script for Hey Ozwell wake-word models.
Based on the Hey Buddy framework for wake-word detection.
"""

import os
import simplejson as json
import random
import argparse
from pathlib import Path
from typing import Tuple, Optional
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import onnxruntime
import onnx
import librosa
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """Dataset for audio wake-word detection"""
    
    def __init__(self, samples: list, data_dir: str, sample_rate: int = 16000, 
                 max_duration: float = 3.0, n_mels: int = 80, augment: bool = False, augment_prob: float = .5):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(max_duration * sample_rate)
        self.n_mels = n_mels
        self.samples = samples
        self.augment = augment
        if self.augment:
            assert augment_prob <= 1.0 and augment_prob >= 0.0, f"augment_prob must be between 0.0 and 1.0 when augment is True"
            self.augment_prob = augment_prob
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        full_path = self.data_dir / file_path

        # Load audio
        audio, sr = librosa.load(full_path, sr=self.sample_rate)

        # Augment Data
        if self.augment and random.random() < self.augment_prob:
            audio = self._apply_augmentation(audio)

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
        
        return torch.FloatTensor(log_mel), torch.LongTensor([label])


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
    

class WakeWordModel(nn.Module):
    """Wake-word detection model based on Hey Buddy architecture"""
    
    def __init__(self, n_mels: int = 80, n_classes: int = 2, dropout: float = 0.3):
        super(WakeWordModel, self).__init__()
        
        self.n_mels = n_mels
        
        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        
        # Calculate size after conv layers
        # This is a rough calculation - in practice you'd compute this dynamically
        conv_output_size = 128 * (n_mels // 8) * (94 // 8)  # Approximate after 3 maxpool layers
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        # Add channel dimension if not present
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # Feature extraction
        features = self.conv_layers(x)
        
        # Classification
        output = self.classifier(features)
        
        return output


class WakeWordTrainer:
    """Trainer for wake-word detection models"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = None
    
    def setup_optimizer(self, learning_rate: float = 1e-3, weight_decay: float = 1e-4):
        """Setup optimizer and learning rate scheduler"""
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=5, 
            factor=0.5
        )
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.squeeze().to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                logger.info(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        return total_loss / num_batches
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.squeeze().to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                predictions = torch.argmax(output, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_targets, all_predictions)
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, save_path: Optional[str] = None):
        """Full training loop"""
        logger.info("Starting training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_accuracy = self.evaluate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            logger.info(f'Epoch {epoch+1}/{epochs}:')
            logger.info(f'  Train Loss: {train_loss:.4f}')
            logger.info(f'  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
            
            # Save best model
            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                logger.info(f'  New best model saved to {save_path}')
    
    def export_onnx(self, output_path: str, input_shape: Tuple[int, ...] = (1, 80, 94)):
        """Export model to ONNX format"""
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            dynamo=False
        )
        
        logger.info(f'Model exported to ONNX: {output_path}')
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info('ONNX model verification passed')

def get_samples(data_dir: str, phrase: str, frac: float=.8):
    manifest_path = Path(data_dir) / phrase / 'training_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    df = pd.DataFrame(manifest['train']['positive_samples'] + manifest['train']['negative_samples']).get(['file', 'label'])
    df['file'] = df.apply(lambda row: os.path.join(phrase, 'train', 'positive', row['file']) if row['label'] == 1 else os.path.join(phrase, 'train', 'negative', row['file']), axis=1)
    train_df = df.groupby('label', group_keys=False).sample(frac=frac)
    val_df = df.drop(train_df.index)
    return (train_df.to_records(index=False).tolist(), val_df.to_records(index=False).tolist())

def main():
    parser = argparse.ArgumentParser(description='Train Hey Ozwell wake-word model')
    parser.add_argument('--phrase', required=True, 
                       choices=['hey-ozwell', "ozwell-i'm-done", 'go-ozwell', 'ozwell-go'],
                       help='Wake phrase to train model for')
    parser.add_argument('--data-dir', default='../data',
                       help='Directory containing training data')
    parser.add_argument('--output', required=True,
                       help='Output path for trained ONNX model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', default='cpu',
                       help='Training device (cpu/cuda/mps)')
    parser.add_argument('--n-mels', type=int, default=80,
                       help='Number of mel frequency bands')
    parser.add_argument('--augment', action='store_true',
                       help='Augment training data')
    parser.add_argument('--augment-prob', type=float, default=0.5,
                       help='Probability of applying augmentation when looping through training data')
    
    args = parser.parse_args()
    os.makedirs('../logs/training', exist_ok=True)
    handler = logging.FileHandler(f'../logs/training/{str(Path(args.output).stem)}.log', 'w')
    logger.addHandler(handler)
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        logger.info('Using CUDA for training')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
        logger.info('Using MPS for training')
    else:
        device = 'cpu'
        logger.info('Using CPU for training')
    
   
    train_samples, val_samples = get_samples(args.data_dir, args.phrase, frac=0.8)
    train_dataset, val_dataset = AudioDataset(train_samples, args.data_dir, n_mels=args.n_mels, augment=args.augment, augment_prob=args.augment_prob), AudioDataset(val_samples, args.data_dir, n_mels=args.n_mels, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Create model
    model = WakeWordModel(n_mels=args.n_mels)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
    # Create trainer
    trainer = WakeWordTrainer(model, device)
    trainer.setup_optimizer(learning_rate=args.learning_rate)
    
    # Train model
    model_save_path = args.output.replace('.onnx', '.pth')
    trainer.train(train_loader, val_loader, epochs=args.epochs, save_path=model_save_path)
    
    # Load best model
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    
    # Export to ONNX
    trainer.export_onnx(args.output)
    
    # Final evaluation
    val_loss, val_accuracy = trainer.evaluate(val_loader)
    logger.info(f'Final validation accuracy: {val_accuracy:.4f}')
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()