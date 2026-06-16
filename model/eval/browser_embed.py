# Browser-FAITHFUL offline embedder: reproduces prod/js exactly so we can mass-generate training data in the
# browser's representation (fixes the offline/browser mismatch without per-user capture).
# Pipeline mirrored from prod/js: AudioBatcher(1.08s buf, 0.12s hop) -> per-buffer peak-norm -> mel /10+2
# -> speech-embedding(win76/stride8 => 4 frames/buffer) -> rolling 16-frame embeddingBuffer (last 4 buffers).
import numpy as np, onnxruntime as ort, os
_P=os.path.join(os.path.dirname(__file__),"pretrained")
_mel=ort.InferenceSession(f"{_P}/mel-spectrogram.onnx",providers=["CPUExecutionProvider"])
_emb=ort.InferenceSession(f"{_P}/speech-embedding.onnx",providers=["CPUExecutionProvider"])
BATCH=17280      # 1.08s * 16000  (AudioBatcher.batchSamples)
HOP=1920         # 0.12s * 16000  (batchIntervalSamples)
WIN,STR=76,8     # speech-embedding window/stride (mel frames)
EMB_FRAMES,EMB_DIM=16,96

def _buffer_embed(buf):
    """One 1.08s buffer -> [4,96] embedding frames (mirrors MelSpectrogram.execute + SpeechEmbedding.execute)."""
    peak=float(np.max(np.abs(buf)))
    if peak>1e-5: buf=buf/peak                      # per-buffer peak-norm (mel-spectrogram.js)
    m=_mel.run(None,{"input":buf[None,:].astype("float32")})[0]
    mf=(m.reshape(-1,32)/10.0+2.0).astype("float32")  # /10+2 (mel-spectrogram.js)
    nt=mf.shape[0]-(mf.shape[0]-WIN)%STR
    wins=np.stack([mf[s:s+WIN] for s in range(0,nt-WIN+1,STR)])[...,None].astype("float32")  # [4,76,32,1]
    e=_emb.run(None,{"input_1":wins})[0].reshape(-1,EMB_DIM).astype("float32")               # [4,96]
    return e

def stream_embeddingbuffers(audio):
    """Yield the [16,96] embeddingBuffer at each hop, exactly as prod/js assembles it (rolling last 4 buffers)."""
    audio=audio.astype("float32")
    buf=np.zeros(BATCH,dtype="float32")             # batcher starts zero-filled
    recent=[]                                       # embeddingBufferArray (last maxEmbeddings=4 buffers)
    maxE=EMB_FRAMES//4                              # 16/4 = 4
    for i in range(0,len(audio),HOP):
        chunk=audio[i:i+HOP]
        buf=np.roll(buf,-len(chunk)); buf[-len(chunk):]=chunk   # AudioBatcher.push (shift + append)
        e=_buffer_embed(buf)                        # [4,96]
        recent.append(e)
        if len(recent)>maxE: recent.pop(0)
        if len(recent)==maxE:
            yield np.concatenate(recent,axis=0)     # [16,96]

if __name__=="__main__":
    import sys; sys.path.insert(0,".")
    from evaluate_wakeword import load_16k_mono
    wake=ort.InferenceSession("../checkpoints/scratch-onnx/ozwelldone_surgical.onnx",providers=["CPUExecutionProvider"])
    a=load_16k_mono("../../real_audio/Oz-done.wav")
    scores=[float(wake.run(None,{"input":eb[None].astype("float32")})[0].reshape(-1)[0]) for eb in stream_embeddingbuffers(a)]
    scores=np.array(scores)
    print(f"Oz-done.wav: {len(scores)} embeddingBuffers, wake fired (>=0.5) on {int((scores>=0.5).sum())}, max {scores.max():.2f}")
    print("SANITY: wake model should fire on real wakes via the browser-faithful pipeline ^")
