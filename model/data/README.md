# Training Data

`data.zip` (~290 MB, LFS-stored) holds the ElevenLabs-generated TTS audio used to train the Hey Ozwell wake-word models.

## Provenance

Mirrored from [`amandamarg/hey-ozwell-data`](https://github.com/amandamarg/hey-ozwell-data) at commit [`a85d8f8`](https://github.com/amandamarg/hey-ozwell-data/commit/a85d8f8) (March 2026).

The download/generation tooling that produced this data lives at [`../legacy/download_tools/`](../legacy/download_tools/) (the canonical scripts are in this repo; the data repo's copy is slightly older).

## Extract

```bash
cd model/data
unzip data.zip
# produces positive/, negative/, and metadata files
```

After extraction the dataset is consumed by [`../heybuddy/`](../heybuddy/) training (`python -m heybuddy train ...`) and historically by the legacy Conv2d pipeline in [`../legacy/tools/prepare_data.py`](../legacy/tools/prepare_data.py).

## Regenerating from scratch (optional)

If you want to regenerate the audio rather than use this snapshot:

```bash
cd model/legacy/download_tools
cp .env.example .env  # add ELEVENLABS_API_KEY
python download.py
```

This will hit the ElevenLabs API and reproduce a comparable dataset. Note: the exact audio will differ (different voice samples each run), so the resulting trained models will not be bit-identical to those in `../exports/heybuddy/`.

## Why mirrored here

To make this repo the single source of truth, the data is committed via Git LFS rather than left as an external dependency. Downside: ~290 MB of LFS storage and per-clone bandwidth. If LFS quota becomes a concern, consider moving to a release asset or external bucket and replacing this file with a `fetch_data.sh`.
