# filename: codebase/cosmology_inference.py
import os
import sys
import json
import zipfile
import time
import math
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt


def ensure_sys_path():
    """
    Ensure that the codebase directory is importable by appending it to sys.path.

    Returns
    -------
    None
    """
    base = os.getcwd()
    codebase_dir = os.path.join(base, "codebase")
    if os.path.isdir(codebase_dir) and (codebase_dir not in sys.path):
        sys.path.append(codebase_dir)


ensure_sys_path()
try:
    from baseline_pipeline import DataLoaderWL, Utility as BaseUtility, SimpleLabelScaler, WLDataset, SimpleCNN
except Exception as e:
    raise RuntimeError("Required module baseline_pipeline.py not found in codebase/. Error: " + str(e))

try:
    from improved_training_pipeline import EnhancedCNN
except Exception as e:
    EnhancedCNN = None



def ensure_dir(path):
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str
        Directory path to ensure exists.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)



def timestamp():
    """
    Generate a compact timestamp string for filenames.

    Returns
    -------
    str
        Timestamp in the form YYYYMMDD_HHMMSS.
    """
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())



def get_database_path():
    """
    Get the plotting output directory from environment variable or default.

    Returns
    -------
    str
        Path to directory where plots will be saved. Defaults to 'data' if not set.
    """
    dbp = os.environ.get("database_path", "data")
    ensure_dir(dbp)
    return dbp



def find_latest_report(pattern):
    """
    Find the latest JSON report under data/artifacts matching a glob pattern.

    Parameters
    ----------
    pattern : str
        Glob pattern such as 'improved_validation_report_*.json' or 'validation_report_*.json'.

    Returns
    -------
    str or None
        Path to the latest report file, or None if none are found.
    """
    artifacts_dir = os.path.join("data", "artifacts")
    ensure_dir(artifacts_dir)
    files = sorted(glob.glob(os.path.join(artifacts_dir, pattern)))
    if not files:
        return None
    return files[-1]



def pick_best_model_report():
    """
    Choose the best available model report between improved and baseline.

    Returns
    -------
    dict
        Dictionary with keys:
        - report_path: str, path to the chosen report
        - type: str, 'improved' or 'baseline'
    """
    rep_impr = find_latest_report("improved_validation_report_*.json")
    rep_base = find_latest_report("validation_report_*.json")
    if rep_impr is not None:
        return {"report_path": rep_impr, "type": "improved"}
    if rep_base is not None:
        return {"report_path": rep_base, "type": "baseline"}
    return {"report_path": None, "type": None}



def train_if_missing():
    """
    Train an improved model if no artifacts are available, otherwise do nothing.

    Returns
    -------
    None
    """
    status = pick_best_model_report()
    if status["report_path"] is not None:
        return
    try:
        from improved_training_pipeline import main as improved_main
        improved_main()
    except Exception:
        try:
            from baseline_pipeline import main as baseline_main
            baseline_main()
        except Exception as e:
            raise RuntimeError("Failed to create training artifacts via improved and baseline pipelines. Error: " + str(e))



def load_model_and_scaler_from_report(report_path, device):
    """
    Load a trained model and its scaler given a report JSON.

    Parameters
    ----------
    report_path : str
        Path to the model report JSON.
    device : torch.device
        Torch device to map the model onto.

    Returns
    -------
    tuple
        (model, scaler_params) where scaler_params is a dict with keys:
        - mean: np.ndarray shape (2,)
        - scale: np.ndarray shape (2,)
        - img_mean: float
        - img_std: float
        - model_type: 'improved' or 'baseline'
    """
    with open(report_path, "r") as f:
        rep = json.load(f)
    model_path = rep.get("model_path", "")
    scaler_path = rep.get("scaler_path", "")
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError("Model or scaler file not found. Model: " + model_path + " Scaler: " + scaler_path)
    scaler_npz = np.load(scaler_path)
    mean = scaler_npz["mean"]
    scale = scaler_npz["scale"]
    img_mean = float(scaler_npz["img_mean"][0])
    img_std = float(scaler_npz["img_std"][0])
    model_type = "improved" if "improved" in os.path.basename(model_path) else "baseline"
    if model_type == "improved":
        if EnhancedCNN is None:
            raise RuntimeError("EnhancedCNN class is not available to load improved model.")
        model = EnhancedCNN(min_sigma=1e-6).to(device)
    else:
        model = SimpleCNN(num_targets=4, min_sigma=1e-6).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, {"mean": mean, "scale": scale, "img_mean": img_mean, "img_std": img_std, "model_type": model_type}



def find_file(candidates, root_dir):
    """
    Return the first existing file among candidates.

    Parameters
    ----------
    candidates : list of str
        Candidate filenames to check.
    root_dir : str
        Root directory containing the files.

    Returns
    -------
    str or None
        Full path of the first candidate that exists, or None.
    """
    for name in candidates:
        fp = os.path.join(root_dir, name)
        if os.path.exists(fp):
            return fp
    return None



def load_mask_and_test_images(data_dir, use_public=True):
    """
    Load the survey mask and reconstruct full-resolution test images from flattened masked arrays.

    Parameters
    ----------
    data_dir : str
        Directory containing input .npy files.
    use_public : bool
        If True, use public test filenames; else use sampled test filenames.

    Returns
    -------
    tuple
        (mask, Xtest) where mask is a boolean array of shape (H, W), and Xtest is a float32 array
        of shape (Ntest, H, W) with masked pixels set to zero.

    Notes
    -----
    If test files are not found, this function returns (None, None).
    """
    mask_file = find_file(["WIDE12H_bin2_2arcmin_mask.npy", "mask.npy"], data_dir)
    if mask_file is None:
        return None, None
    mask = np.load(mask_file).astype(bool)
    H, W = mask.shape
    if use_public:
        test_candidates = ["WIDE12H_bin2_2arcmin_kappa_noisy_test.npy"]
    else:
        test_candidates = ["sampled_WIDE12H_bin2_2arcmin_kappa_noisy_test.npy"]
    test_file = find_file(test_candidates, data_dir)
    if test_file is None:
        return None, None
    flat = np.load(test_file)
    if flat.ndim != 2:
        return None, None
    nmask = int(np.sum(mask))
    if flat.shape[1] != nmask:
        return None, None
    X = np.zeros((flat.shape[0], H, W), dtype=np.float32)
    idx = np.where(mask)
    for i in range(flat.shape[0]):
        X[i][idx] = flat[i].astype(np.float32)
    return mask, X



def build_inference_loader(images, img_mean, img_std, device, batch_size=16):
    """
    Create a DataLoader for inference given images and normalization stats.

    Parameters
    ----------
    images : np.ndarray
        Test images of shape (N, H, W), float32, dimensionless.
    img_mean : float
        Global mean used in training normalization (dimensionless).
    img_std : float
        Global std used in training normalization (dimensionless).
    device : torch.device
        Device for prefetch pinning decisions.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    DataLoader
        Loader that yields only images tensors with shape (B, 1, H, W).
    """
    ds = WLDataset(images, None, img_mean, img_std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    return loader



def run_inference(model, loader, device):
    """
    Run inference to obtain predicted means and sigmas in standardized label space.

    Parameters
    ----------
    model : nn.Module
        Trained model placed on the desired device.
    loader : DataLoader
        Inference data loader.
    device : torch.device
        Device for execution.

    Returns
    -------
    tuple
        (means_scaled, sigmas_scaled) as numpy arrays of shape (N, 2).
    """
    model.eval()
    means_list = []
    sigmas_list = []
    with torch.no_grad():
        for batch in loader:
            x = batch
            x = x.to(device, non_blocking=True)
            means, sigmas = model(x)
            means_list.append(means.detach().cpu().numpy())
            sigmas_list.append(sigmas.detach().cpu().numpy())
    pm = np.concatenate(means_list, axis=0) if means_list else np.zeros((0, 2), dtype=np.float32)
    ps = np.concatenate(sigmas_list, axis=0) if sigmas_list else np.zeros((0, 2), dtype=np.float32)
    return pm, ps



def invert_label_transform(means_scaled, sigmas_scaled, scaler_params):
    """
    Invert label standardization to obtain predictions in the original dimensionless scale.

    Parameters
    ----------
    means_scaled : np.ndarray
        Predicted means in standardized space, shape (N, 2).
    sigmas_scaled : np.ndarray
        Predicted sigmas in standardized space, shape (N, 2).
    scaler_params : dict
        Dictionary with 'mean' and 'scale' keys (np.ndarray shape (2,)).

    Returns
    -------
    tuple
        (means, sigmas) in original scale, both shape (N, 2).
    """
    mean = scaler_params["mean"]
    scale = scaler_params["scale"]
    means = means_scaled * scale + mean
    sigmas = sigmas_scaled * scale
    return means, sigmas



def enforce_nonnegativity_prior(means, sigmas):
    """
    Enforce the prior that parameters are non-negative by bounding uncertainties:
    for entries where mean - sigma < 0, set sigma to mean.

    Parameters
    ----------
    means : np.ndarray
        Predicted means, shape (N, 2).
    sigmas : np.ndarray
        Predicted sigmas, shape (N, 2).

    Returns
    -------
    np.ndarray
        Adjusted sigmas with non-negativity prior enforced.
    """
    mask = means - sigmas < 0.0
    sigmas_adj = sigmas.copy()
    sigmas_adj[mask] = means[mask]
    return sigmas_adj



def validate_submission_arrays(means, errorbars, n_expected):
    """
    Validate shape, finiteness, and non-negativity properties of submission arrays.

    Parameters
    ----------
    means : np.ndarray
        Predicted means (N, 2).
    errorbars : np.ndarray
        Predicted one-sigma uncertainties (N, 2).
    n_expected : int
        Expected number of test samples.

    Raises
    ------
    ValueError
        If validation fails.
    """
    if means.ndim != 2 or errorbars.ndim != 2:
        raise ValueError("means and errorbars must be 2D arrays.")
    if means.shape[1] != 2 or errorbars.shape[1] != 2:
        raise ValueError("means and errorbars must have shape (N, 2).")
    if means.shape[0] != errorbars.shape[0]:
        raise ValueError("means and errorbars must have the same number of rows.")
    if not (means.shape[0] == n_expected):
        raise ValueError("Number of predictions " + str(means.shape[0]) + " does not match expected " + str(n_expected))
    if not np.all(np.isfinite(means)):
        raise ValueError("Non-finite values found in means.")
    if not np.all(np.isfinite(errorbars)):
        raise ValueError("Non-finite values found in errorbars.")
    if np.any(errorbars < 0.0):
        raise ValueError("Negative uncertainties found.")



def save_submission_zip(means, errorbars, out_dir):
    """
    Save predictions to result.json and compress into a zip suitable for submission.

    Parameters
    ----------
    means : np.ndarray
        Predicted means (N, 2).
    errorbars : np.ndarray
        Predicted uncertainties (N, 2).
    out_dir : str
        Directory to save the submission files.

    Returns
    -------
    tuple
        (json_path, zip_path)
    """
    ensure_dir(out_dir)
    payload = {"means": means.tolist(), "errorbars": errorbars.tolist()}
    stamp = timestamp()
    json_path = os.path.join(out_dir, "result.json")
    with open(json_path, "w") as f:
        json.dump(payload, f)
    zip_name = "Submission_" + stamp + ".zip"
    zip_path = os.path.join(out_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="result.json")
    os.remove(json_path)
    return json_path, zip_path



def plot_test_summary(means, sigmas, save_dir, stamp):
    """
    Create a single summary figure (2x2 subplots) for test predictions:
    - Scatter of Omega_m vs S8 (means)
    - Histogram of predicted uncertainties (both parameters)
    - Histogram of predicted Omega_m means
    - Histogram of predicted S8 means

    Parameters
    ----------
    means : np.ndarray
        Predicted means (N, 2), dimensionless.
    sigmas : np.ndarray
        Predicted uncertainties (N, 2), dimensionless.
    save_dir : str
        Directory to save the plot (usually $database_path).
    stamp : str
        Timestamp string for filename.

    Returns
    -------
    str
        Path to the saved figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    names = ["Omega_m", "S8"]
    ax = axes[0, 0]
    ax.scatter(means[:, 0], means[:, 1], s=16, alpha=0.8)
    ax.set_xlabel("Prediction " + names[0] + " (dimensionless)")
    ax.set_ylabel("Prediction " + names[1] + " (dimensionless)")
    ax.set_title("Predicted " + names[0] + " vs " + names[1])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax = axes[0, 1]
    vals0 = sigmas[:, 0]
    vals1 = sigmas[:, 1]
    all_vals = np.concatenate([vals0, vals1])
    bins = min(50, max(15, int(np.sqrt(all_vals.size))))
    rng = (float(np.min(all_vals)), float(np.max(all_vals)))
    ax.hist(vals0, bins=bins, range=rng, color="tab:blue", alpha=0.6, edgecolor="black", label=names[0])
    ax.hist(vals1, bins=bins, range=rng, color="tab:orange", alpha=0.6, edgecolor="black", label=names[1])
    ax.set_xlabel("Predicted sigma (dimensionless)")
    ax.set_ylabel("Count")
    ax.set_title("Uncertainty distributions")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    ax = axes[1, 0]
    m0 = means[:, 0]
    bins0 = min(50, max(15, int(np.sqrt(m0.size))))
    ax.hist(m0, bins=bins0, color="tab:green", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Predicted " + names[0] + " (dimensionless)")
    ax.set_ylabel("Count")
    ax.set_title(names[0] + " predictions")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax = axes[1, 1]
    m1 = means[:, 1]
    bins1 = min(50, max(15, int(np.sqrt(m1.size))))
    ax.hist(m1, bins=bins1, color="tab:red", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Predicted " + names[1] + " (dimensionless)")
    ax.set_ylabel("Count")
    ax.set_title(names[1] + " predictions")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.suptitle("Test predictions summary")
    fig.tight_layout()
    fname = os.path.join(save_dir, "test_predictions_summary_1_" + stamp + ".png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    return fname



def print_test_stats(means, sigmas):
    """
    Print comprehensive statistics for test predictions.

    Parameters
    ----------
    means : np.ndarray
        Predicted means (N, 2).
    sigmas : np.ndarray
        Predicted uncertainties (N, 2).
    """
    names = ["Omega_m", "S8"]
    N = means.shape[0]
    print("Number of test samples: " + str(N))
    for i in range(2):
        m = means[:, i]
        s = sigmas[:, i]
        print("Parameter " + names[i] + " mean stats:")
        print("  mean=" + str(float(np.mean(m))) + ", std=" + str(float(np.std(m))) + ", min=" + str(float(np.min(m))) + ", max=" + str(float(np.max(m))))
        percs_m = {}
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            percs_m[str(p)] = float(np.percentile(m, p))
        print("  percentiles=" + json.dumps(percs_m))
        print("Parameter " + names[i] + " sigma stats:")
        print("  mean=" + str(float(np.mean(s))) + ", std=" + str(float(np.std(s))) + ", min=" + str(float(np.min(s))) + ", max=" + str(float(np.max(s))))
        percs_s = {}
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            percs_s[str(p)] = float(np.percentile(s, p))
        print("  percentiles=" + json.dumps(percs_s))



def build_synthetic_test_from_loader(loader, nmax=None):
    """
    Build a synthetic test set using noisy training maps if real test files are missing.

    Parameters
    ----------
    loader : DataLoaderWL
        Loaded synthetic or real dataset loader with kappa and mask.
    nmax : int or None
        Maximum number of samples to include; if None, include all.

    Returns
    -------
    tuple
        (mask, Xtest) where mask is boolean (H, W) and Xtest is float32 (N, H, W).
    """
    H, W = loader.mask.shape
    noisy = loader.add_noise_to_kappa()
    Ncosmo = noisy.shape[0]
    Nsys = noisy.shape[1]
    X = noisy.reshape(Ncosmo * Nsys, H, W)
    if nmax is not None:
        X = X[:nmax]
    return loader.mask, X.astype(np.float32)



def main():
    """
    Main inference routine for generating submission-ready predictions and plots.

    Workflow
    --------
    1) Ensure data/ directories exist; attempt to locate the best model report; train if missing.
    2) Load model and scaler; attempt to load real test data; if missing, build synthetic test.
    3) Normalize images with saved training stats; run batched inference on device.
    4) Invert label scaling and enforce non-negativity prior on uncertainties.
    5) Validate submission arrays; save JSON and ZIP; save compact NPZ; generate and save plots.
    6) Print detailed summary statistics to the console.

    Units
    -----
    All reported parameters and uncertainties are dimensionless.
    """
    ensure_dir("data")
    ensure_dir(os.path.join("data", "artifacts"))
    ensure_dir(os.path.join("data", "models"))
    train_if_missing()
    pick = pick_best_model_report()
    if pick["report_path"] is None:
        raise RuntimeError("No model artifacts found; training did not produce a report.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler_params = load_model_and_scaler_from_report(pick["report_path"], device)
    data_dir = "input_data"
    use_public = True
    mask, Xtest = load_mask_and_test_images(data_dir, use_public=use_public)
    if Xtest is None:
        use_public = False
        mask, Xtest = load_mask_and_test_images(data_dir, use_public=use_public)
    if Xtest is None:
        loader = DataLoaderWL(data_dir=data_dir, use_public=False, ng=30.0, pixel_size_arcmin=2.0)
        used_synth = False
        try:
            loader.load()
        except Exception:
            loader.load_synthetic(Ncosmo=6, Nsys=12, H=1274, W=176)
            used_synth = True
        mask, Xtest = build_synthetic_test_from_loader(loader, nmax=None)
        if used_synth:
            print("Using synthetic test set constructed from synthetic loader.")
        else:
            print("Using synthetic test set constructed from available training maps.")
    img_mean = scaler_params["img_mean"]
    img_std = scaler_params["img_std"]
    loader_inf = build_inference_loader(Xtest, img_mean, img_std, device, batch_size=16)
    means_scaled, sigmas_scaled = run_inference(model, loader_inf, device)
    means, sigmas = invert_label_transform(means_scaled, sigmas_scaled, scaler_params)
    sigmas_adj = enforce_nonnegativity_prior(means, sigmas)
    validate_submission_arrays(means, sigmas_adj, Xtest.shape[0])
    stamp = timestamp()
    preds_path = os.path.join("data", "test_predictions_" + stamp + ".npz")
    np.savez(preds_path, means=means, sigmas=sigmas_adj)
    print("Saved test predictions NPZ at " + preds_path)
    _, zip_path = save_submission_zip(means, sigmas_adj, out_dir="data")
    print("Submission ZIP saved at " + zip_path)
    plot_dir = get_database_path()
    fig_path = plot_test_summary(means, sigmas_adj, plot_dir, stamp)
    print("Saved plot: Test predictions summary (scatter and histograms) at " + fig_path)
    print_test_stats(means, sigmas_adj)



if __name__ == "__main__":
    main()