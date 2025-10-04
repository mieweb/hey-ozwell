# Model Training & Testing

This directory contains the training pipeline and evaluation tools for Hey Ozwell wake-word models, built on top of the [Hey Buddy!](https://huggingface.co/spaces/bennyboy/hb) framework.

## Directory Structure

```
model/
├── tools/           # Training and data preparation scripts
├── testing/         # Evaluation and accuracy testing
├── data/           # Training datasets (gitignored)
├── exports/        # Trained ONNX models (gitignored)
└── README.md       # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchaudio onnx onnxruntime librosa soundfile numpy scipy
```

### 2. Prepare Training Data

```bash
cd tools
python prepare_data.py --phrase "hey ozwell" --positive-samples 500 --negative-samples 2000
```

### 3. Train Model

```bash
python train.py --phrase "hey ozwell" --epochs 100 --output ../exports/hey-ozwell.onnx
```

### 4. Evaluate Performance

```bash
cd ../testing
python evaluate.py --model ../exports/hey-ozwell.onnx --test-data ../data/hey-ozwell/test/
```

## Wake Phrases

The following wake phrases are supported:

1. **"hey ozwell"** → start recording
2. **"ozwell i'm done"** → stop recording  
3. **"go ozwell"** → stop recording + respond
4. **"ozwell go"** → stop recording + respond

## Training Process

### Data Collection

- **Positive samples**: Clear recordings of the target phrase
- **Negative samples**: Similar-sounding phrases, ambient noise, silence
- **Augmentation**: Speed variation (0.8x-1.2x), pitch shifting, background noise

### Model Architecture

Based on Hey Buddy's architecture:
- Mel-spectrogram feature extraction
- Convolutional neural network backbone
- Binary classification (wake phrase vs. not)
- Optimized for ONNX Runtime Web

### Evaluation Metrics

- **Detection Rate**: % of true positives correctly identified (target: >95%)
- **False Positive Rate**: False alarms per hour in quiet environment (target: <1/hour)
- **Latency**: Time from phrase end to detection (target: <250ms)

## Advanced Usage

### Custom Training Data

Place your training data in the following structure:

```
data/
├── hey-ozwell/
│   ├── positive/    # .wav files containing "hey ozwell"
│   ├── negative/    # .wav files without wake phrase
│   └── test/        # Evaluation dataset
├── ozwell-im-done/
│   ├── positive/
│   ├── negative/
│   └── test/
└── ...
```

### Model Optimization

For production deployment:

```bash
python optimize.py --input ../exports/hey-ozwell.onnx --output ../exports/hey-ozwell-optimized.onnx
```

### Batch Training

Train all phrases at once:

```bash
python train_all.py --config config.yaml
```

## Troubleshooting

### Low Detection Rate
- Increase positive training samples
- Add more diverse speaking styles
- Adjust detection threshold

### High False Positives
- Add more negative samples
- Include similar-sounding phrases in negative set
- Reduce detection threshold

### Performance Issues
- Use GPU training: `--device cuda`
- Reduce model complexity: `--model-size small`
- Enable mixed precision: `--mixed-precision`

## Integration with Production

Once trained, models are deployed to the JavaScript SDK:

```bash
cp exports/*.onnx ../prod/js/models/
```

See `/prod/js/README.md` for integration instructions.