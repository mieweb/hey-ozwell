
# Hey Ozwell — Wake-Word Model & Production SDK

## Trademarks

**Ozwell** is a trademark of **MIE**.
Use of the Ozwell name in this project is solely for identification of the intended wake-word phrases and does not imply any affiliation with or endorsement by MIE.

**Hey Ozwell** is an open-source wake-word solution for detecting:

* **“hey ozwell”** → start recording
* **“ozwell i’m done”** → stop recording
* **“go ozwell”** → stop recording & respond
* **“ozwell go”** → stop recording & respond

Runs fully client-side with **ONNX Runtime Web**, built on top of the excellent **[Hey Buddy!](https://github.com/painebenjamin/hey-buddy)** ([live demo](https://huggingface.co/spaces/benjamin-paine/hey-buddy)) wake-word framework, and designed for **low-latency, privacy-friendly** operation in browsers and on-device platforms.

---

## Repository Structure

```
hey-ozwell/
├── model/               # Model training, evaluation, testing
│   ├── tools/           # CLI tools & scripts for data prep & training
│   ├── testing/         # Unit tests, benchmarks, accuracy tests
│   ├── data/            # (gitignored) Training and eval datasets
│   ├── README.md        # Instructions for training & testing models
│   └── ...              
│
└── prod/                # Production-ready client implementations
    ├── js/              # JavaScript browser SDK
    │   ├── src/         # Source code for wake-word listener & ring buffer
    │   ├── examples/    # Example integrations (basic, advanced)
    │   └── README.md    # How to integrate in a web app
    ├── ios/             # (future) iOS native integration
    ├── mac/             # (future) macOS native integration
    ├── windows/         # (future) Windows native integration
    └── common/          # (future) Shared inference utils for cross-platform
```

---

## Features

* 🎙 **On-device detection** — no cloud processing required.
* ⚡ **Low latency** — detection within \~250 ms of phrase end.
* 🔒 **Privacy by design** — nothing leaves the device without consent.
* 💾 **Offline-capable** — models cached in IndexedDB (JS version).
* 🛠 **Extensible** — train new phrases or improve models with real-world samples.
* 🌍 **Multi-platform roadmap** — browser, iOS, macOS, Windows.

---

## Quick Start — Browser (JS)

```bash
# Install dependencies
npm install onnxruntime-web idb-keyval
```

```js
import { WakeListener, ModelManager } from './prod/js/src/index.js';

const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const models = await ModelManager.fetchAndCache('/models/v1.0/', {
  'hey-ozwell': 'hey-ozwell.onnx',
  'im-done': 'im-done.onnx',
  'go-ozwell': 'go-ozwell.onnx',
  'ozwell-go': 'ozwell-go.onnx'
});

const listener = new WakeListener();
await listener.start(stream, {
  models,
  onWake: (label) => console.log('Wake phrase detected:', label),
});
```

---

## Model Development

### Requirements

* Python 3.10+
* ONNX, PyTorch, NumPy
* Audio processing libs (`librosa`, `soundfile`)
* Optional: GPU for faster training

### Workflow

1. **Collect data**: Positive & negative samples for each phrase.
2. **Augment**: Apply noise, speed/pitch variation.
3. **Train**: Run `/model/tools/train.py` to produce `.onnx`.
4. **Evaluate**: Use `/model/testing/evaluate.py` for accuracy & false-positive rate.
5. **Deploy**: Move `.onnx` files into `/prod/js/models/` or CDN.

```bash
cd model/tools
python train.py --phrase "hey ozwell" --out ../exports/hey-ozwell.onnx
```

---

## Production Integration (JS)

* **Wake-word detection** runs continuously in a **Web Audio AudioWorklet**.
* **Ring buffer recorder** stores the last N seconds for manual capture.
* **Event API**:

  * `wake:hey-ozwell` → start recording
  * `wake:im-done` → stop recording
  * `wake:go-ozwell`, `wake:ozwell-go` → stop & respond

See `/prod/js/examples/basic/` for a minimal HTML+JS demo.

---

## Acknowledgments

* **[Hey Buddy!](https://huggingface.co/spaces/bennyboy/hb)** — by [Benny Paine](https://huggingface.co/bennyboy).
  This project’s wake-word detection is based on the Hey Buddy! framework, which provides a streamlined pipeline for training, exporting, and running wake-word models in ONNX format directly in the browser.

* **ONNX Runtime Web** — high-performance inference engine for running ONNX models in JavaScript.

* **Hugging Face** — hosting & community support for sharing and documenting model workflows.

---

## Contributing

Pull requests welcome for:

* Model improvements
* New phrase support
* Additional platform SDKs (iOS, macOS, Windows)
* Performance optimizations

---

## License

MIT License — see [LICENSE](LICENSE) for details.
