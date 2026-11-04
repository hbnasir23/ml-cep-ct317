"""
Q2 - audio pre-processing and feature extraction.

Pipeline for one clip:
    load (16 kHz mono) -> low-pass Butterworth filter (noise removal)
    -> trim leading/trailing silence -> extract a fixed-length feature vector

Feature groups (all summarised over time with statistics so every clip gives
a vector of the same length, whatever its duration):
    fft        : log energy in 16 linear frequency bands of the magnitude FFT
    mfcc       : 20 MFCCs + their deltas (mean, std)
    mel        : 40-band log mel-spectrogram (mean, std)
    chroma     : 12 pitch-class chromagram (mean, std)
    spectral   : spectral centroid, bandwidth, roll-off, flatness, 7-band
                 spectral contrast (mean, std each)
    prosody    : pitch (F0 via YIN) stats, RMS energy, zero-crossing rate,
                 voiced ratio, duration - intonation/loudness/tempo cues
    bicoherence: mean, variance, skewness, kurtosis of magnitude and phase
                 of the bicoherence
"""
import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt
from scipy.stats import skew, kurtosis

SR = 16000            # all CREMA-D clips are 16 kHz
LOWPASS_HZ = 4000     # speech/emotion information is mostly below 4 kHz
FILTER_ORDER = 6
N_MFCC = 20
N_MELS = 40
N_FFT = 512
HOP = 160             # 10 ms


def lowpass_filter(y, sr=SR, cutoff=LOWPASS_HZ, order=FILTER_ORDER):
    """Zero-phase Butterworth low-pass filter (removes high-frequency hiss)."""
    sos = butter(order, cutoff, btype="low", fs=sr, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def load_and_clean(path, sr=SR, cutoff=LOWPASS_HZ):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = lowpass_filter(y, sr, cutoff)
    y_trim, _ = librosa.effects.trim(y, top_db=30)
    if len(y_trim) >= sr // 4:          # keep trimmed version only if >= 0.25 s
        y = y_trim
    peak = np.max(np.abs(y)) + 1e-9
    return y / peak                     # amplitude normalisation


def _stats(m):
    """mean and std over time for a (features x frames) matrix."""
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])


def fft_features(y, n_bands=16, max_hz=LOWPASS_HZ, sr=SR):
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    edges = np.linspace(0, max_hz, n_bands + 1)
    bands = [spec[(freqs >= lo) & (freqs < hi)] for lo, hi in zip(edges[:-1], edges[1:])]
    energy = np.array([np.sum(b ** 2) for b in bands]) + 1e-12
    return np.log(energy / energy.sum())


def bicoherence_features(y, seg_len=256, n_bins=64, max_segments=64):
    """Bicoherence b(f1,f2) = |E[X(f1)X(f2)X*(f1+f2)]| /
       sqrt(E[|X(f1)X(f2)|^2] E[|X(f1+f2)|^2]), estimated by averaging over
       windowed segments. Returns mean/var/skew/kurtosis of its magnitude and
       of the bispectrum phase over the region f1 >= f2, f1+f2 < n_bins."""
    hop = seg_len // 2
    n_seg = max(1, (len(y) - seg_len) // hop + 1)
    idx = np.linspace(0, max(0, len(y) - seg_len), min(n_seg, max_segments)).astype(int)
    win = np.hanning(seg_len)
    segs = np.stack([y[i:i + seg_len] if i + seg_len <= len(y)
                     else np.pad(y[i:], (0, seg_len - len(y[i:]))) for i in idx])
    X = np.fft.rfft((segs - segs.mean(axis=1, keepdims=True)) * win, axis=1)

    f = np.arange(n_bins)
    f1, f2 = np.meshgrid(f, f, indexing="ij")
    mask = (f1 >= f2) & (f1 + f2 < n_bins) & (f2 > 0)
    a, b = f1[mask], f2[mask]
    X1, X2, X3 = X[:, a], X[:, b], np.conj(X[:, a + b])
    prod = X1 * X2
    bispec = np.mean(prod * X3, axis=0)
    denom = np.sqrt(np.mean(np.abs(prod) ** 2, axis=0) * np.mean(np.abs(X3) ** 2, axis=0)) + 1e-12
    mag = np.abs(bispec) / denom
    phase = np.angle(bispec)
    out = []
    for v in (mag, phase):
        out += [v.mean(), v.var(), skew(v), kurtosis(v)]
    return np.nan_to_num(np.array(out))


def prosody_features(y, sr=SR):
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    voiced = rms > 0.1 * rms.max()
    f0v = f0[voiced[:len(f0)]] if voiced[:len(f0)].any() else f0
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0]
    return np.array([
        f0v.mean(), f0v.std(), np.percentile(f0v, 10), np.percentile(f0v, 90),
        np.ptp(f0v), rms.mean(), rms.std(), rms.max(), zcr.mean(), zcr.std(),
        voiced.mean(), len(y) / sr])


def extract_features(y, sr=SR):
    """Returns dict {group_name: 1-D feature array}."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=N_MELS, fmax=LOWPASS_HZ)
    log_mel = librosa.power_to_db(mel)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC)
    d_mfcc = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    centroid = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=np.sqrt(S), sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=np.sqrt(S), sr=sr)
    flatness = librosa.feature.spectral_flatness(S=np.sqrt(S))
    contrast = librosa.feature.spectral_contrast(S=np.sqrt(S), sr=sr, fmin=100.0, n_bands=6)

    return {
        "fft": fft_features(y, sr=sr),
        "mfcc": np.concatenate([_stats(mfcc), _stats(d_mfcc)]),
        "mel": _stats(log_mel),
        "chroma": _stats(chroma),
        "spectral": np.concatenate([_stats(centroid), _stats(bandwidth), _stats(rolloff),
                                    _stats(flatness), _stats(contrast)]),
        "prosody": prosody_features(y, sr),
        "bicoherence": bicoherence_features(y),
    }


FEATURE_GROUPS = ["fft", "mfcc", "mel", "chroma", "spectral", "prosody", "bicoherence"]


def features_from_file(path, cutoff=LOWPASS_HZ):
    feats = extract_features(load_and_clean(path, cutoff=cutoff))
    return np.concatenate([feats[g] for g in FEATURE_GROUPS]).astype(np.float32)


def feature_group_slices():
    """Column ranges of each group inside the concatenated vector."""
    dummy = extract_features(np.random.RandomState(0).randn(SR).astype(np.float32) * 0.1)
    slices, start = {}, 0
    for g in FEATURE_GROUPS:
        slices[g] = slice(start, start + len(dummy[g]))
        start += len(dummy[g])
    return slices
