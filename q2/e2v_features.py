"""
emotion2vec+ large embeddings (Ma et al., "emotion2vec: Self-Supervised
Pre-Training for Speech Emotion Representation", ACL Findings 2024).

emotion2vec+ is a data2vec-style speech model further trained on ~40,000 hours
of emotional speech (pseudo-labelled from many public sources), which makes
its utterance embedding far more robust to new speakers and microphones than
general-purpose speech models. It is used FROZEN as a feature extractor; the
classifiers are the scikit-learn models required by the handout.

Model files: q2/pretrained/emotion2vec_plus_large/ (model.pt, config.yaml,
tokens.txt) from https://huggingface.co/emotion2vec/emotion2vec_plus_large

Usage (shard extraction):
    python q2/e2v_features.py LIST.txt OUT.npz [threads] [cutoff_hz or 0]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audio_features as af  # noqa: E402

MODEL_DIR = os.path.join(HERE, "pretrained", "emotion2vec_plus_large")
HF_ID = "emotion2vec/emotion2vec_plus_large"
DIM = 1024
_model = None


def _load():
    global _model
    if _model is None:
        import torch
        from funasr import AutoModel
        # The checkpoint is 1.9 GB, but only 0.65 GB are model weights (the
        # rest is training-optimizer state). Memory-mapping it means the
        # unused parts are never loaded into RAM, which cuts peak memory by
        # several GB on machines with little free memory.
        if not getattr(torch.load, "_mmap_patched", False):
            _orig = torch.load

            def _load_mmap(f, *a, **kw):
                if isinstance(f, (str, os.PathLike)):
                    kw.setdefault("mmap", True)
                return _orig(f, *a, **kw)
            _load_mmap._mmap_patched = True
            torch.load = _load_mmap
        src = MODEL_DIR if os.path.exists(os.path.join(MODEL_DIR, "model.pt")) else HF_ID
        kw = {} if src == MODEL_DIR else {"hub": "hf"}
        _model = AutoModel(model=src, disable_update=True, disable_pbar=True, **kw)
    return _model


def clean(path, cutoff=af.LOWPASS_HZ):
    """Same pre-processing as the rest of the project; cutoff=0 skips the filter."""
    import librosa
    y, _ = librosa.load(path, sr=af.SR, mono=True)
    if cutoff:
        y = af.lowpass_filter(y, af.SR, cutoff)
    yt, _ = librosa.effects.trim(y, top_db=30)
    if len(yt) >= af.SR // 4:
        y = yt
    return (y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


def embed_signal(y):
    """-> (1024-d utterance embedding, 9 emotion2vec class scores)."""
    r = _load().generate(y, granularity="utterance", extract_embedding=True)[0]
    return np.asarray(r["feats"], np.float32).reshape(-1)[:DIM], np.asarray(r["scores"], np.float32)


def labels():
    r = _load().generate(np.zeros(af.SR, np.float32), granularity="utterance")[0]
    return list(r["labels"])


def main():
    lst, out = sys.argv[1], sys.argv[2]
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    cutoff = int(sys.argv[4]) if len(sys.argv) > 4 else af.LOWPASS_HZ
    import torch
    torch.set_num_threads(threads)
    paths = [p.strip() for p in open(lst, encoding="utf-8") if p.strip()]
    E = np.zeros((len(paths), DIM), np.float32)
    S = np.zeros((len(paths), 9), np.float32)
    part = out + ".partial.npz"
    start = 0
    if os.path.exists(part):                      # resume after a crash
        d = np.load(part)
        start = int(d["n"])
        E[:start], S[:start] = d["E"][:start], d["S"][:start]
        print(f"resuming at {start}", flush=True)
    for i in range(start, len(paths)):
        E[i], S[i] = embed_signal(clean(paths[i], cutoff))
        if i % 250 == 0:
            print(f"{os.path.basename(out)}: {i}/{len(paths)}", flush=True)
            np.savez(part, E=E, S=S, n=i + 1)
    np.savez(out, E=E, S=S, labels=np.array(labels()))
    print(f"{os.path.basename(out)}: done {len(paths)}", flush=True)


if __name__ == "__main__":
    main()
