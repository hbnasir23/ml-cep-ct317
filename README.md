# Voice Sentiment & Emotion Recognition (ML CEP, CT-317)

**Team:** Sumayyah Mohammedy (SE-23013), Amna Khan (SE-23039), Mariha Mahmood (SE-23047), Haris Bin Nasir (SE-23084)
Department of Software Engineering, SE Batch 2023, Fall 2026. The full report is in [report/report.pdf](report/report.pdf).

The system recognises the **sentiment** (positive / neutral / negative) and the **emotion** (anger, disgust, fear, happy, neutral, sad) in a short voice recording. The pipeline is:

```
voice -> 4 kHz low-pass filter -> WavLM self-supervised speech features (layer 7)
      -> scikit-learn classifier (KNN / Logistic Regression / Naive Bayes / SVM / MLP)
```

## Results (speakers never seen in training)

| System | 6 emotions | Sentiment |
|---|---|---|
| Hand-crafted features (MFCC, mel, chroma, spectral, prosody, FFT, bicoherence) + SVM | 52.4% | n/a |
| **WavLM + Logistic Regression**, CREMA-D test speakers | **68.9%** | **86.4%** (balanced 78.5%) |
| Same model on another corpus (RAVDESS), never trained on it | 45.7% | 76.0% |
| + multi-corpus training (CREMA-D + RAVDESS), unseen RAVDESS actors | **70.5%** | **85.2%** |
| + **few-shot personalisation** with 12 of the user's own clips | **82.4%** | **89.6%** |

Chance level is 16.7% for 6 emotions. Always answering "Negative" gives 68.3% sentiment accuracy, but only 33.3% balanced accuracy.

**Why the first model failed on our own recordings:** it was trained on one corpus of acted, noisy studio clips. Your microphone, room and voice are a *new domain*. Multi-corpus training and personalisation fix most of that gap, as the table shows.

## Try it: the app

```bash
pip install -r requirements.txt
streamlit run app.py
```
* **Record with the microphone** (or upload a WAV) and get the sentiment and emotion with probabilities, plus a spectrogram of what the noise filter removed.
* **Input-quality warnings** flag recordings that are too short, too quiet or clipped.
* **Teach it your voice:** after each prediction, choose the emotion you actually meant and press *Add as my example*. With about 2 examples per emotion, press **Personalise**, and the model retrains on your voice in a few seconds.

The trained models are in `q2/models/`, so the app works right after cloning. WavLM (~380 MB) is downloaded automatically on first use.

Command line:
```bash
python q2/predict.py my_voice.wav
```

## Reproducing everything

1. Get the data (not stored in the repo because of its size):
   ```bash
   git clone https://github.com/naweedfatima03/SentimentAnalysisDataset q2/dataset_repo   # CREMA-D, 7,442 clips
   # RAVDESS speech (208 MB): https://zenodo.org/records/1188976
   #   Audio_Speech_Actors_01-24.zip -> extract into q2/ravdess/
   ```
2. Run the pipeline:
   ```bash
   python q2/q2_audio_sentiment.py --cutoff-study   # speaker-independent split + hand-crafted baseline
   python q2/q2_extras.py                           # baseline extras (ensemble, per-speaker norm.)
   # WavLM features, 4 shards in parallel (~45 min on CPU); writes q2/cache/ssl/emb_{0..3}.npy
   python -c "import pandas as pd,numpy as np,os;d=pd.read_csv('q2/results/split.csv');p=[os.path.abspath('q2/dataset_repo/NoiseAudioWAV/'+f) for f in d.file];[open(f'q2/cache/ssl/list_{k}.txt','w').write('\n'.join(c)) for k,c in enumerate(np.array_split(p,4))]"
   for k in 0 1 2 3; do python q2/ssl_features.py q2/cache/ssl/list_$k.txt q2/cache/ssl/emb_$k.npy 3 & done; wait
   python q2/q2_wavlm_emotion.py                    # layer choice, 5 classifiers, RAVDESS, multi-corpus, final model
   python q2/personalization.py                     # few-shot personalisation experiment
   ```

## Files

| File | Purpose |
|---|---|
| `q2/audio_features.py` | low-pass filter, silence trim, 242 hand-crafted features |
| `q2/q2_audio_sentiment.py` | speaker-independent split, baseline classifiers, ablation, cut-off study |
| `q2/ssl_features.py` | WavLM embedding extraction |
| `q2/q2_wavlm_emotion.py` | main experiment and deployed model |
| `q2/personalization.py` | few-shot personalisation (experiment and function used by the app) |
| `q2/predict.py` | prediction for WAV files, input-quality checks |
| `app.py` | Streamlit application |
| `q2/results/`, `q2/results/wavlm/` | all tables and figures |
| `report/` | LaTeX report and PDF (`tectonic -X compile report/report.tex`) |
