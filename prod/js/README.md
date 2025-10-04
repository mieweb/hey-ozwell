# JavaScript SDK for Hey Ozwell

Browser-based wake-word detection using ONNX Runtime Web. Runs fully client-side with IndexedDB model caching and ring buffer recording.

## Installation

```bash
npm install onnxruntime-web idb-keyval
```

## Quick Start

```javascript
import { WakeListener, ModelManager, RingBufferRecorder } from './src/index.js';

// Request microphone access
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

// Load and cache models
const models = await ModelManager.fetchAndCache('/models/v1.0/', {
  'hey-ozwell': 'hey-ozwell.onnx',
  'im-done': 'ozwell-im-done.onnx',
  'go-ozwell': 'go-ozwell.onnx',
  'ozwell-go': 'ozwell-go.onnx'
});

// Set up wake-word listener
const listener = new WakeListener();
await listener.start(stream, {
  models,
  onWake: (label, confidence) => {
    console.log(`Wake phrase detected: ${label} (${confidence})`);
    
    switch(label) {
      case 'hey-ozwell':
        startRecording();
        break;
      case 'im-done':
        stopRecording();
        break;
      case 'go-ozwell':
      case 'ozwell-go':
        stopRecordingAndRespond();
        break;
    }
  }
});

// Optional: Set up ring buffer for pre-roll recording
const recorder = new RingBufferRecorder({ 
  bufferDuration: 30, // seconds
  stream 
});
```

## API Reference

### WakeListener

Main class for wake-word detection.

```javascript
const listener = new WakeListener(options);
```

**Options:**
- `threshold` (number): Detection confidence threshold (0.0-1.0, default: 0.7)
- `bufferSize` (number): Audio processing buffer size (default: 4096)
- `sampleRate` (number): Audio sample rate (default: 16000)

**Methods:**

#### `start(stream, config)`
Start listening for wake words.

- `stream` (MediaStream): Audio stream from getUserMedia()
- `config.models` (Object): Loaded ONNX models
- `config.onWake` (Function): Callback for wake detection

#### `stop()`
Stop wake-word detection.

#### `setThreshold(threshold)`
Update detection threshold dynamically.

### ModelManager

Handles model loading and IndexedDB caching.

```javascript
const models = await ModelManager.fetchAndCache(baseUrl, modelConfig);
```

**Methods:**

#### `fetchAndCache(baseUrl, config)`
Download and cache models in IndexedDB.

- `baseUrl` (string): Base URL for model files
- `config` (Object): Map of label to filename

Returns: Object with loaded ONNX models

#### `clearCache()`
Clear all cached models from IndexedDB.

#### `getCachedModel(key)`
Retrieve specific model from cache.

### RingBufferRecorder

Circular audio buffer for pre-roll recording.

```javascript
const recorder = new RingBufferRecorder(options);
```

**Options:**
- `bufferDuration` (number): Buffer length in seconds
- `stream` (MediaStream): Audio stream
- `sampleRate` (number): Sample rate (default: 16000)

**Methods:**

#### `getBuffer(duration?)`
Get audio buffer content.

- `duration` (number): Seconds to retrieve (default: all)

Returns: Float32Array with audio samples

#### `clear()`
Clear the ring buffer.

## Events

The WakeListener emits the following wake phrase labels:

- **`hey-ozwell`** → User wants to start recording
- **`im-done`** → User wants to stop recording
- **`go-ozwell`** → User wants to stop recording and get response
- **`ozwell-go`** → User wants to stop recording and get response

## Examples

### Basic Integration

See `/examples/basic/` for a minimal HTML page with wake-word detection.

### Advanced Integration

See `/examples/advanced/` for a complete implementation with:
- Ring buffer pre-roll recording
- Visual feedback
- Settings panel
- Error handling

### React Integration

```javascript
import { useWakeListener } from './hooks/useWakeListener';

function App() {
  const { isListening, startListening, stopListening } = useWakeListener({
    onWake: (label) => handleWakePhrase(label)
  });
  
  return (
    <div>
      <button onClick={isListening ? stopListening : startListening}>
        {isListening ? 'Stop Listening' : 'Start Listening'}
      </button>
    </div>
  );
}
```

## Browser Support

- Chrome 66+ (AudioWorklet support)
- Firefox 76+ (AudioWorklet support)  
- Safari 14.1+ (AudioWorklet support)
- Edge 79+ (Chromium-based)

## Performance

- **Latency**: ~200-250ms from phrase end to detection
- **CPU Usage**: <5% on modern devices
- **Memory**: ~20MB for all 4 models
- **Network**: Models cached in IndexedDB after first load

## Troubleshooting

### No Wake Detection
- Check microphone permissions
- Verify audio stream is active
- Lower detection threshold
- Check browser console for errors

### High False Positives
- Increase detection threshold
- Use noise-reduced environment for training
- Update models with more negative samples

### Performance Issues
- Reduce buffer size
- Use fewer concurrent models
- Enable hardware acceleration in browser

## Development

### Building

```bash
npm run build
```

### Testing

```bash
npm test
```

### Linting

```bash
npm run lint
```

## License

MIT License - see [LICENSE](../../LICENSE) for details.