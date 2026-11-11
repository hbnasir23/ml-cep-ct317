"""
Self-supervised speech embeddings (WavLM) for emotion recognition.

A pretrained WavLM-base-plus model (Microsoft, trained without labels on
94k hours of speech) turns a clip into a sequence of 768-d vectors at each
of its 13 layers (CNN output + 12 transformer layers). We average every layer
over time ("mean pooling"), giving a 13 x 768 matrix per clip. The best layer
(or layer average) is chosen on the validation speakers, and then the
scikit-learn classifiers from the handout are trained on it.

The same pre-processing as before is applied first (4 kHz low-pass filter,
silence trimming, peak normalisation), which also makes clean microphone
recordings look more like the noisy training clips.

Usage (extract embeddings for a list of wav files, used by q2_audio_sentiment.py):
    python q2/ssl_features.py LIST.txt OUT.npy [threads]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_features as af  # noqa: E402

MODEL_NAME = "microsoft/wavlm-base-plus"
N_LAYERS = 13
DIM = 768

_model = None
_fe = None


def _load():
    global _model, _fe
    if _model is None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
        _fe = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True).eval()
        torch.set_grad_enabled(False)
    return _model, _fe


def embed_signal(y, sr=af.SR):
    """y: pre-processed mono float signal at 16 kHz -> (13, 768) float32."""
    import torch
    model, fe = _load()
    x = fe(y, sampling_rate=sr, return_tensors="pt")
    with torch.inference_mode():
        hs = model(**x).hidden_states
    return torch.stack([h[0].mean(dim=0) for h in hs]).numpy().astype(np.float32)


def embed_file(path, cutoff=af.LOWPASS_HZ):
    return embed_signal(af.load_and_clean(path, cutoff=cutoff))


def main():
    lst, out = sys.argv[1], sys.argv[2]
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    import torch
    torch.set_num_threads(threads)
    paths = [p.strip() for p in open(lst, encoding="utf-8") if p.strip()]
    E = np.zeros((len(paths), N_LAYERS, DIM), np.float16)
    for i, p in enumerate(paths):
        E[i] = embed_file(p)
        if i % 200 == 0:
            print(f"{os.path.basename(out)}: {i}/{len(paths)}", flush=True)
    np.save(out, E)
    print(f"{os.path.basename(out)}: done {len(paths)}", flush=True)


if __name__ == "__main__":
    main()
