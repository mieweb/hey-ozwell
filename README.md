# Hey Ozwell

Browser-based wake-word detection for the Ozwell voice assistant.

This repo is the **single source of truth** for:

- **[`model/`](./model/)** — training framework (vendored [Hey Buddy](https://github.com/painebenjamin/hey-buddy)), training data, conversion tooling, and exported PyTorch checkpoints.
- **[`prod/js/`](./prod/js/)** — browser runtime + demo page that loads the trained ONNX models and listens for wake words.

## Wake words

Currently shipping in the demo:

| Phrase | Status |
|---|---|
| `hey ozwell` | shipped (MLP, `prod/js/models/hey-ozwell.onnx`) |
| `ozwell i'm done` | shipped (MLP, `prod/js/models/ozwell-i'm-done.onnx`) |

## Quickstart — run the demo

```bash
cd prod/js
npm install
npm run worklet && npm run build
PORT=3001 npm start
# open http://localhost:3001 and click Initialize
```

Details in [`prod/js/README.md`](./prod/js/README.md).

## Quickstart — convert a checkpoint to ONNX

```bash
cd model
pip install -r requirements.txt
python tools/convert.py   # defaults: ozwell-i'm-done → prod/js/models/
```

Training details (heybuddy MLP recipe + dataset extraction) in [`model/README.md`](./model/README.md).

## Repository layout

```
hey-ozwell/
├── model/                  Training source of truth
│   ├── heybuddy/           Vendored Hey Buddy framework (Apache-2.0)
│   ├── tools/convert.py    .pt → .onnx
│   ├── exports/heybuddy/   Trained checkpoints (LFS)
│   ├── data/data.zip       Training dataset, mirrored (LFS, ~290 MB)
│   ├── docs/onnx_export/   PyTorch→ONNX export reports
│   ├── legacy/             Archived Conv2d pipeline (4 wake words)
│   └── requirements.txt
└── prod/                   Runtime
    └── js/                 Browser demo + library (Hey Buddy ONNX pipeline)
        ├── src/            Page sources + AudioWorklet
        ├── models/         Trained wake-word ONNX (LFS)
        ├── index.html, server.js
        ├── webpack.config.js, worklet.config.js, babel.config.json
        ├── package.json
        ├── LICENSE-APACHE  Upstream Hey Buddy license
        └── LICENSE-UPSTREAM-MIT  Upstream amandamarg/hey-ozwell-demo license
```

## Git LFS

Required for clone. `.gitattributes` routes `*.wav`, `*.onnx`, `*.onnx.data`, `*.pt`, `*.pth`, `*.png`, `*.zip` to LFS. Install:

```bash
git lfs install
git lfs pull
```

## Credits

- [**Hey Buddy**](https://github.com/painebenjamin/hey-buddy) by Benjamin Paine — Apache-2.0. Vendored under `model/heybuddy/` and forms the inference pipeline in `prod/js/`.
- [**hey-ozwell-demo**](https://github.com/amandamarg/hey-ozwell-demo) and [**hey-ozwell-data**](https://github.com/amandamarg/hey-ozwell-data) by [@amandamarg](https://github.com/amandamarg) — MIT. Source of trained Ozwell checkpoints and dataset mirrored here.

## License

This repository's original code is MIT (see [LICENSE](./LICENSE)). Vendored Hey Buddy code retains its Apache-2.0 license (see [`prod/js/LICENSE-APACHE`](./prod/js/LICENSE-APACHE)).
