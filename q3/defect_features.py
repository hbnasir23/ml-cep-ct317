"""
Q3 - image pre-processing, defect-candidate proposal and feature extraction.

Pipeline for one 2000x2000 image of the paper reel
  1. grayscale -> downsample to 1000x1000 (INTER_AREA keeps small dark spots)
  2. illumination flattening: residual = image - GaussianBlur(image, sigma=20)
     -> robust z-score map  z = (residual - median) / (1.4826 * MAD)
     Clean paper texture stays within about -7..+10; holes and ink are much
     darker (z << -10), independent of the lighting gradient across the image.
  3. candidate proposal: threshold z < -4, morphological closing (joins the
     broken strokes of a pen mark), connected components ranked by
     "darkness mass" = sum of -z over the component.
  4. features
     global (whole image)          : intensity histogram, z histogram,
                                     spatial binning of z, HOG of z
     patch (around top candidate)  : z histogram, spatial binning, HOG,
                                     LBP texture histogram, GLCM statistics,
                                     shape descriptors (area, eccentricity,
                                     solidity, circularity, Hu moments, ...)
"""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import regionprops, label as sk_label

WORK = 1000            # working resolution (original images are 2000x2000)
BG_SIGMA = 20          # background (illumination) blur
Z_THR = -4.0           # candidate threshold on the z map
PATCH = 64             # patch side after resizing
MIN_PATCH = 64         # minimum crop side at working resolution
Z_CLIP = (-30.0, 10.0)
PAD = 160


# ---------------------------------------------------------- preprocessing --
def load_gray(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    return g


def preprocess(gray):
    """gray (any size) -> (small image float32 WORKxWORK, z map, scale)."""
    scale = gray.shape[1] / WORK
    s = cv2.resize(gray, (WORK, WORK), interpolation=cv2.INTER_AREA).astype(np.float32)
    bg = cv2.GaussianBlur(s, (0, 0), BG_SIGMA)
    r = s - bg
    med = np.median(r)
    mad = 1.4826 * np.median(np.abs(r - med)) + 1e-6
    z = (r - med) / mad
    return s, z.astype(np.float32), scale


AUGMENTATIONS = {
    "orig": lambda a: a,
    "rot90": lambda a: np.ascontiguousarray(np.rot90(a, 1)),
    "rot180": lambda a: np.ascontiguousarray(np.rot90(a, 2)),
    "flip": lambda a: np.ascontiguousarray(a[:, ::-1]),
}


# ------------------------------------------------------- candidate regions --
def find_candidates(z, k=5, thr=Z_THR):
    """Return up to k dark regions, strongest first. Each is a dict with
    bbox (x, y, w, h) at working resolution, centroid, score, mask crop."""
    zs = cv2.GaussianBlur(z, (0, 0), 1.0)
    mask = (zs < thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cands = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < 3:
            continue
        comp = lab[y:y + h, x:x + w] == i
        vals = zs[y:y + h, x:x + w][comp]
        cands.append({"bbox": (int(x), int(y), int(w), int(h)), "centroid": tuple(cen[i]),
                      "area": int(area), "score": float(-np.minimum(vals, 0).sum()),
                      "min_z": float(vals.min()), "mean_z": float(vals.mean()), "mask": comp})
    cands.sort(key=lambda c: c["score"], reverse=True)
    if not cands:   # no region passed the threshold: fall back to the darkest pixel
        yy, xx = np.unravel_index(np.argmin(zs), zs.shape)
        cands = [{"bbox": (int(xx) - 2, int(yy) - 2, 5, 5), "centroid": (float(xx), float(yy)),
                  "area": 1, "score": 0.0, "min_z": float(zs.min()), "mean_z": float(zs.min()),
                  "mask": np.ones((5, 5), bool)}]
    return cands[:k]


# ---------------------------------------------------------------- features --
def _znorm(z):
    """z map -> [0, 1] (0 = very dark defect, ~0.75 = normal paper)."""
    return (np.clip(z, *Z_CLIP) - Z_CLIP[0]) / (Z_CLIP[1] - Z_CLIP[0])


def global_features(s, z):
    hist_int = np.histogram(s, bins=32, range=(0, 255))[0].astype(np.float32)
    hist_int /= hist_int.sum()
    hist_z = np.histogram(np.clip(z, -60, 20), bins=32, range=(-60, 20))[0].astype(np.float32)
    hist_z = np.log1p(hist_z)
    zn = _znorm(z)
    # min-pooling keeps a 5-pixel hole visible after strong downsampling
    zmin = zn.reshape(32, WORK // 32, 32, WORK // 32).min(axis=(1, 3)) if WORK % 32 == 0 else \
        cv2.erode(zn, np.ones((31, 31))).astype(np.float32)[::31, ::31][:32, :32]
    spatial = zmin.ravel()
    small = cv2.erode(zn, np.ones((5, 5), np.uint8))
    small = cv2.resize(small, (128, 128), interpolation=cv2.INTER_AREA)
    h = hog(small, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
            feature_vector=True)
    return {"g_hist_int": hist_int, "g_hist_z": hist_z, "g_spatial": spatial, "g_hog": h}


def crop_patch(z, cand):
    x, y, w, h = cand["bbox"]
    cx, cy = x + w / 2.0, y + h / 2.0
    side = int(max(MIN_PATCH, 1.4 * max(w, h)))
    zp = np.pad(z, PAD, mode="reflect")
    x0, y0 = int(round(cx - side / 2)) + PAD, int(round(cy - side / 2)) + PAD
    x0 = int(np.clip(x0, 0, zp.shape[1] - side))
    y0 = int(np.clip(y0, 0, zp.shape[0] - side))
    crop = zp[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (PATCH, PATCH), interpolation=cv2.INTER_AREA), side


def shape_features(cand):
    m = cand["mask"].astype(np.uint8)
    props = regionprops(sk_label(m))
    p = max(props, key=lambda r: r.area)
    perim = p.perimeter + 1e-6
    hu = cv2.HuMoments(cv2.moments(m)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    x, y, w, h = cand["bbox"]
    return np.concatenate([[
        np.log1p(cand["area"]), w, h, max(w, h) / (min(w, h) + 1e-6),
        p.extent, p.solidity, p.eccentricity,
        p.major_axis_length, p.minor_axis_length,
        4 * np.pi * p.area / perim ** 2,             # circularity
        cand["min_z"], cand["mean_z"], np.log1p(cand["score"])],
        np.clip(hu, -30, 30)]).astype(np.float32)


def patch_features(z, cand):
    patch, side = crop_patch(z, cand)
    pn = _znorm(patch)
    hist = np.histogram(np.clip(patch, -60, 20), bins=16, range=(-60, 20))[0].astype(np.float32)
    hist = np.log1p(hist)
    spatial = cv2.resize(pn, (16, 16), interpolation=cv2.INTER_AREA).ravel()
    h = hog(pn, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
            feature_vector=True)
    u8 = (pn * 255).astype(np.uint8)
    lbp = local_binary_pattern(u8, P=8, R=1, method="uniform")
    lbp_hist = np.histogram(lbp, bins=10, range=(0, 10))[0].astype(np.float32)
    lbp_hist /= lbp_hist.sum()
    q = (pn * 31).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    glcm_f = np.concatenate([graycoprops(glcm, prop).ravel() for prop in
                             ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")])
    return {"p_hist": hist, "p_spatial": spatial, "p_hog": h, "p_lbp": lbp_hist,
            "p_glcm": np.nan_to_num(glcm_f), "p_shape": np.append(shape_features(cand), np.log(side))}


GLOBAL_GROUPS = ["g_hist_int", "g_hist_z", "g_spatial", "g_hog"]
PATCH_GROUPS = ["p_hist", "p_spatial", "p_hog", "p_lbp", "p_glcm", "p_shape"]
ALL_GROUPS = GLOBAL_GROUPS + PATCH_GROUPS


def features_from_arrays(s, z):
    """Feature dict for one (already preprocessed / augmented) image."""
    cand = find_candidates(z, k=1)[0]
    f = global_features(s, z)
    f.update(patch_features(z, cand))
    return f


def image_features(path, augs=("orig",)):
    """List of flat feature vectors (one per augmentation) for one image file."""
    s, z, _ = preprocess(load_gray(path))
    out = []
    for a in augs:
        fa = AUGMENTATIONS[a]
        f = features_from_arrays(fa(s), fa(z))
        out.append(np.concatenate([f[g] for g in ALL_GROUPS]).astype(np.float32))
    return out


def proposal_features(path, k=5):
    """Patch features of the top-k dark-region proposals of one image.
    Returns (matrix k' x n_patch_features, proposals, scale to original px)."""
    s, z, scale = preprocess(load_gray(path))
    cands = find_candidates(z, k=k)
    rows = []
    for c in cands:
        f = patch_features(z, c)
        rows.append(np.concatenate([f[g] for g in PATCH_GROUPS]))
    return np.vstack(rows).astype(np.float32), cands, scale


def group_slices():
    rng = np.random.RandomState(0)
    s = rng.normal(180, 3, (WORK, WORK)).astype(np.float32)
    s[500:506, 500:506] = 60
    _, z, _ = preprocess(s.astype(np.uint8))
    f = features_from_arrays(s, z)
    out, i = {}, 0
    for g in ALL_GROUPS:
        out[g] = slice(i, i + len(f[g]))
        i += len(f[g])
    return out
