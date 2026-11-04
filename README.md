# ML CEP (CT-317): three machine-learning problems

**Team:** Sumayyah Mohammedy (SE-23013), Amna Khan (SE-23039), Mariha Mahmood (SE-23047), Haris Bin Nasir (SE-23084)
Department of Software Engineering, SE Batch 2023, Fall 2026. The full report is in [report/report.pdf](report/report.pdf).

| | Result (held-out test set) |
|---|---|
| Q1 fake-news KNN | 63.9% (Euclidean) → **75.7%** (cosine) |
| Q2 emotion from audio, unseen speakers | **52.4%** (SVM), **61.3%** with per-speaker normalization (chance 16.7%) |
| Q3 paper defects (3 classes) + bounding boxes | **97–99%** with all four classifiers; boxes on the real defect in every inspected image |

**Innovation:** an explainable web app (`streamlit run app.py`) where you test every model with your own input:
* **Q1:** type headlines and see the prediction plus the 12 nearest training headlines that explain it.
* **Q2:** record your voice with the microphone (or upload a WAV), and see the emotion probabilities and what the noise filter removed.
* **Q3:** upload a paper image and see the class, the bounding box, a zoom on the defect, and every candidate region with its score.

The trained models are included, so the app works right after cloning. Only the "pick a test sample" options need the datasets.

| Folder | Question | What it does |
|---|---|---|
| [q1/](q1/) | Q1 Classification | KNN (scikit-learn) classifying real vs. fake news headlines, Euclidean vs. cosine distance |
| [q2/](q2/) | Q2 NLP / audio | Emotion recognition from noisy CREMA-D speech (low-pass filter, audio features, 5 classifiers) |
| [q3/](q3/) | Q3 Computer vision | Paper-reel defect classification (paper / hole / pen mark) and bounding-box localization |
| [app.py](app.py) | all | Streamlit web app to try all three trained models interactively |
| [report/](report/) | all | LaTeX report (`report.tex`) and compiled `report.pdf`. Rebuild with `tectonic -X compile report/report.tex` |

## 1. Setup

```bash
pip install -r requirements.txt
```
Tested with Python 3.13 and scikit-learn 1.8 on Windows 11.

### Data
| Question | Where it goes | Source |
|---|---|---|
| Q1 | `q1/data/clean_real.txt`, `q1/data/clean_fake.txt` | included in this repository |
| Q2 | `q2/dataset_repo/NoiseAudioWAV/*.wav` (7,442 files) | `git clone https://github.com/naweedfatima03/SentimentAnalysisDataset q2/dataset_repo` |
| Q3 | `q3/data/out/{paper,hole,penMark}/*.jpg` (1,500 files) | Google Drive folder linked in the CEP handout. Download the `out` folder as a zip and extract it into `q3/data/` |

## 2. Running

Run everything from the `ml_cep` folder. Each script is self-contained, fixes its random seed (42), and writes figures and tables to its own `results/` folder.

```bash
python q1/q1_knn.py                          # ~10 s
python q2/q2_audio_sentiment.py              # ~3 min (feature extraction is cached)
python q2/q2_audio_sentiment.py --cutoff-study   # + low-pass cut-off comparison
python q2/q2_extras.py                       # ensemble + per-speaker normalization
python q3/q3_defect_detection.py             # ~20 min on 12 cores (features cached)

# use the trained models
python q2/predict.py q2/dataset_repo/NoiseAudioWAV/1001_DFA_ANG_XX.wav
python q3/predict.py q3/data/out/hole/0.jpg q3/data/out/penMark/100.jpg
streamlit run app.py                         # web demo of all three
```

## 3. Q1: KNN on news headlines

* `load_data()` reads 1,968 real and 1,298 fake headlines and makes a stratified **70/15/15** split (2286 / 490 / 490). It then fits a `CountVectorizer` on the training split only, so validation and test vocabulary does not leak into training.
* `select_knn_model()` trains `KNeighborsClassifier` for k = 1..20, plots training and validation accuracy, picks the best k on validation (ties go to the larger k), and reports test accuracy.

| Metric | Best k | Validation acc. | **Test acc.** |
|---|---|---|---|
| Euclidean (default) | 12 | 0.694 | **0.639** |
| Cosine | 12 | 0.796 | **0.757** |

**Why cosine works better.** Cosine distance is `1 - u·v / (|u||v|)`, which depends only on the angle between two count vectors, not their length. For the hint `['cat', 'bulldozer', 'cat cat cat']`, the Euclidean distance from *cat* to *bulldozer* is 1.41, smaller than the 2.0 from *cat* to *cat cat cat*. Cosine gives 1 and 0, which is what we want. In bag-of-words data, Euclidean distance mostly measures headline length and word repetition. Cosine measures which words are shared, which carries the topic and style that separate real from fake headlines.

Plots: `q1/results/knn_euclidean.png`, `knn_cosine.png`, `knn_comparison.png`.

## 4. Q2: Emotion from audio (CREMA-D, 6 classes)

**Split.** The 91 actors are split 70/15/15 into 63 / 14 / 14 *speakers* (5152 / 1142 / 1148 clips), so no voice appears in more than one set. A random clip-level split would let the classifier memorise speakers and inflate the scores. Every emotion is present in each split, and all 12 sentences are in every split.

**Noise removal.** A 6th-order zero-phase Butterworth **low-pass filter at 4 kHz** (`sosfiltfilt`) is applied, followed by silence trimming and peak normalization. See `results/lowpass_demo.png`.

**Features (242 per clip, summarised over time with mean and std):**

| Group | Dims | Content |
|---|---|---|
| fft | 16 | log energy in 16 FFT bands (0 to 4 kHz) |
| mfcc | 80 | 20 MFCCs + deltas |
| mel | 80 | 40-band log mel-spectrogram |
| chroma | 24 | 12 pitch-class chromagram |
| spectral | 22 | centroid, bandwidth, roll-off, flatness, 7-band contrast |
| prosody | 12 | F0 (YIN) statistics, RMS energy, ZCR, voiced ratio, duration |
| bicoherence | 8 | mean, var, skew, kurtosis of bicoherence magnitude and phase |

**Classifiers.** KNN, Logistic Regression, Gaussian Naive Bayes, RBF-SVM and an MLP neural network, each behind a `StandardScaler`. Each is tuned by grid search on the validation speakers, refit on train+val, and tested once on the unseen test speakers.

| Model | Validation acc. | **Test acc.** | Test macro-F1 |
|---|---|---|---|
| KNN (k=15, cosine) | 0.469 | 0.408 | 0.381 |
| Logistic Regression (C=1) | 0.525 | 0.519 | 0.519 |
| Naive Bayes | 0.387 | 0.402 | 0.354 |
| **SVM (C=3, gamma=scale)** | 0.531 | **0.524** | 0.522 |
| Neural Network (128, alpha=0.1) | 0.537 | 0.513 | 0.513 |

Chance level is 0.167. Anger is the easiest emotion to recognise; disgust and fear are confused most (`results/confusion_matrices.png`).

**Extra analysis**
* **Feature ablation** (`results/feature_ablation.csv`): MFCCs are the strongest single group (0.49 on their own) and removing them hurts most (to 0.50). Bicoherence alone is weak (0.29) but costs nothing in the full set.
* **Cut-off study** (`results/cutoff_study.csv`): cut-offs of 2, 3, 4 and 6 kHz and no filter all land at 0.524 to 0.531. Most emotion cues are below 2 to 3 kHz, so the filter removes noise without hurting accuracy.
* **Ensemble and per-speaker normalization** (`q2_extras.py`, `results/extras.csv`): a soft-voting ensemble of SVM, LR and NN reaches 0.538. Normalizing every feature per speaker (no labels needed) raises the SVM to **0.613**.

## 5. Q3: Paper defect detection

Images are 2000×2000 and effectively grayscale (all three channels are equal). Holes are dark specks of 5 to 25 px and pen marks are thin dark strokes. The paper texture and the lighting gradient are much more visible than the defects, so classical global features need help.

**Pipeline** (`q3/defect_features.py`, illustrated in `results/preprocessing_pipeline.png`)
1. Downsample to 1000 px with `INTER_AREA`.
2. Flatten the lighting by subtracting a Gaussian background (sigma 20), then compute a robust **z-score map**. Clean paper stays within about −7σ; defects reach −10σ to −50σ.
3. **Proposals:** threshold at −4σ, apply morphological closing (which joins broken pen strokes), and rank the connected components by "darkness mass".
4. **Features:**
   * **Global:** intensity histogram, z histogram, spatial binning (32×32 min-pooled), HOG.
   * **Patch** (around a proposal): z histogram, spatial binning, HOG, **LBP**, **GLCM** and **shape descriptors** (area, eccentricity, solidity, circularity, Hu moments, ...).
5. **Data augmentation:** training images are rotated 90° and 180° and flipped, giving ×4 data.

**Task (i): three-class classification** with SVM, Naive Bayes, Logistic Regression and an MLP on three feature sets. Each is tuned on validation and tested on 225 held-out images:

| Classifier | Global (2852 features) | Patch (2107) | All (4959) |
|---|---|---|---|
| SVM (RBF) | 0.969 | 0.982 | 0.987 |
| Naive Bayes | 0.942 | 0.960 | 0.987 |
| Logistic Regression | 0.978 | 0.978 | **0.991** |
| Neural Network | 0.982 | 0.973 | 0.987 |

**Task (ii): localization.** A *region classifier* (the same four classifier families) is trained on the top-5 proposals of every image. The top proposal of a defect image is labelled with the image class, and all other proposals are labelled background. That second group includes static blemishes on the lens, which appear in every frame, so they act as hard negatives. At inference, every proposal is scored and the most defect-like one is boxed (`results/localization/`, `localization_zoom.png`).

The multi-proposal SVM had the best validation accuracy of all models (0.996), so it is the final model. On test it gets 0.973: all 77 clean images correct, 68 of 73 holes, and 74 of 75 pen marks. For 146 of 148 defective test images the chosen box is also the darkest region; in the other two, the classifier correctly skipped a lens blemish. Visual inspection (`localization_zoom.png`, `errors.png` from `q3/error_analysis.py`) shows the box on the real defect in every inspected case, including all 6 misclassified images.

## 6. Design choices that matter
* **No data leakage.** The CountVectorizer is fitted on training data only (Q1), the audio split is by speaker (Q2), and test sets are used exactly once after validation-based model selection (Q1 to Q3).
* **Reproducible.** Seeds are fixed and features are cached in `*/cache/`; delete that folder to recompute.
* **Ease of use.** Each question has one command, `predict.py` scripts handle new files, and `app.py` is a browser demo with upload or pick-a-test-sample for each question.
