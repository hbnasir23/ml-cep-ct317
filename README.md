# Paper Surface Defect Detection (ML CEP, CT-317)

**Team:** Sumayyah Mohammedy (SE-23013), Amna Khan (SE-23039), Mariha Mahmood (SE-23047), Haris Bin Nasir (SE-23084)
Department of Software Engineering, SE Batch 2023, Fall 2026. The full report is in [report/report.pdf](report/report.pdf).

The system classifies images of a paper reel as **no defect / hole / pen mark** and draws a **bounding box** around the defect. It uses classical machine vision (no deep learning) and the four classifiers required by the handout.

```
image -> illumination flattening (robust z-score map) -> dark-region candidates
      -> features: colour histogram, spatial binning, HOG, LBP, GLCM, shape + Hu moments
      -> SVM / Naive Bayes / Logistic Regression / Neural Network
```

## Results on 225 held-out test images

| Classifier | Global features | Candidate-patch features | All features |
|---|---|---|---|
| SVM (RBF) | 96.9% | 98.2% | 98.7% |
| Naive Bayes | 94.2% | 96.0% | 98.7% |
| Logistic Regression | 97.8% | 97.8% | **99.1%** |
| Neural Network (MLP) | 98.2% | 97.3% | 98.7% |

**Final system** (selected on the validation set): a hard-negative region classifier (SVM) that scores every candidate. It gets **97.3% test accuracy**: 77/77 clean images (no false alarms), 74/75 pen marks and 68/73 holes. The bounding box is on the real defect in every inspected test image. To re-check it from the saved model, run `python q3/verify_test.py`.

## Try it

**Deploy online (1 click):** [share.streamlit.io/deploy](https://share.streamlit.io/deploy?repository=hbnasir23/ml-cep-ct317&branch=main&mainModule=app.py). The online app ships 24 sample test images in `q3/samples/`.

Locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```
* **Single image:** upload a photo or pick or randomly draw a test image. You get the class, confidence, bounding box and a zoom on the defect.
* **How the model sees it:** the illumination-flattened map with every candidate region and its score.
* **Batch:** upload many images and get the defect count, defect rate and a downloadable CSV.

Command line: `python q3/predict.py image.jpg` (annotated image saved to `q3/predictions/`).

## Reproducing

1. Download the image folder `out` from the Google Drive link in the CEP handout and extract it to `q3/data/out/{paper,hole,penMark}/` (1,500 JPGs, 1.6 GB, not stored in the repo).
2. Run the pipeline:
   ```bash
   python q3/q3_defect_detection.py   # split, augmentation, 4 classifiers x 3 feature sets,
                                      # region classifier, localisation, figures (~20 min, 12 cores)
   python q3/verify_test.py           # final test accuracy straight from the image files
   python q3/error_analysis.py        # figure of the misclassified test images
   ```

## Files

| File | Purpose |
|---|---|
| `q3/defect_features.py` | pre-processing, candidate proposals, all features |
| `q3/q3_defect_detection.py` | full experiment and model training |
| `q3/predict.py` | prediction and bounding boxes for new images |
| `q3/verify_test.py` | independent test-set check |
| `app.py` | Streamlit inspection application |
| `q3/results/` | tables and figures |
| `report/` | LaTeX report and PDF (`tectonic -X compile report/report.tex`) |

An earlier speech-emotion version of this project is kept on the [`voice-sentiment`](https://github.com/hbnasir23/ml-cep-ct317/tree/voice-sentiment) branch.
