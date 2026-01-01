# filename: codebase/evaluation.py
import os
import json
import glob
import time
import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt


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


def load_latest_validation_artifacts():
    """
    Locate and load the latest validation artifacts: predictions and ground truths.

    Returns
    -------
    dict
        Dictionary with keys:
        - pred_means: np.ndarray of shape (N, 2), predicted point estimates (dimensionless).
        - pred_sigmas: np.ndarray of shape (N, 2), predicted 1-sigma uncertainties (dimensionless).
        - y_true: np.ndarray of shape (N, 2), ground-truth labels (dimensionless).
        - report_path: str, path to the validation report JSON used.

    Notes
    -----
    If no validation report is found, this function attempts to run the baseline pipeline to
    generate the artifacts, and then reloads them. This may take several minutes on first run.
    """
    artifacts_dir = os.path.join("data", "artifacts")
    ensure_dir(artifacts_dir)
    reports = sorted(glob.glob(os.path.join(artifacts_dir, "validation_report_*.json")))
    if not reports:
        try:
            from baseline_pipeline import main as baseline_main
            baseline_main()
            reports = sorted(glob.glob(os.path.join(artifacts_dir, "validation_report_*.json")))
        except Exception as e:
            raise RuntimeError("No validation artifacts found and baseline pipeline could not be executed. Error: " + str(e))
    report_path = reports[-1]
    with open(report_path, "r") as f:
        report = json.load(f)
    pred_means_path = report.get("pred_means_path", "")
    pred_sigmas_path = report.get("pred_sigmas_path", "")
    y_val_path = report.get("y_val_path", "")
    if not (os.path.exists(pred_means_path) and os.path.exists(pred_sigmas_path) and os.path.exists(y_val_path)):
        raise FileNotFoundError("One or more artifact files listed in the report do not exist. Report: " + report_path)
    pred_means = np.load(pred_means_path)
    pred_sigmas = np.load(pred_sigmas_path)
    y_true = np.load(y_val_path)
    return {
        "pred_means": pred_means,
        "pred_sigmas": pred_sigmas,
        "y_true": y_true,
        "report_path": report_path
    }


def compute_challenge_score(y_true, mu, sigma, lam=1e3, eps=1e-12):
    """
    Compute the challenge success score as the negative mean of the defined terms.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (N, 2), dimensionless.
    mu : np.ndarray
        Predicted means (N, 2), dimensionless.
    sigma : np.ndarray
        Predicted 1-sigma uncertainties (N, 2), dimensionless.
    lam : float
        Penalty coefficient for point estimate error term.
    eps : float
        Small constant to avoid division by zero and log(0).

    Returns
    -------
    float
        Averaged negative score; closer to 0 is better.
    """
    resid_sq = (mu - y_true) ** 2
    denom = sigma ** 2 + eps
    terms = resid_sq / denom + np.log(denom) + lam * resid_sq
    per_sample = -np.sum(terms, axis=1)
    return float(np.mean(per_sample))


def summarize_statistics(y_true, mu, sigma):
    """
    Compute detailed residual and uncertainty statistics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels (N, 2), dimensionless.
    mu : np.ndarray
        Predicted means (N, 2), dimensionless.
    sigma : np.ndarray
        Predicted uncertainties (N, 2), dimensionless.

    Returns
    -------
    dict
        Summary statistics including MSE, MAE, RMSE, residual mean/std, sigma mean/median/min/max,
        percentiles, and coverage metrics at common Gaussian levels.
    """
    resid = mu - y_true
    mse = float(np.mean(resid ** 2))
    mae = np.mean(np.abs(resid), axis=0)
    rmse = np.sqrt(np.mean(resid ** 2, axis=0))
    resid_mean = np.mean(resid, axis=0)
    resid_std = np.std(resid, axis=0)
    sigma_mean = np.mean(sigma, axis=0)
    sigma_median = np.median(sigma, axis=0)
    sigma_min = np.min(sigma, axis=0)
    sigma_max = np.max(sigma, axis=0)
    perc = {}
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        perc[str(p)] = np.percentile(sigma, p, axis=0).tolist()
    z = np.abs(resid) / (sigma + 1e-12)
    levels = {
        "0.50": 0.67448975,
        "0.68": 1.0,
        "0.80": 1.28155157,
        "0.90": 1.64485363,
        "0.95": 1.95996398
    }
    coverage = {}
    for k in levels:
        thr = levels[k]
        coverage[k] = np.mean(z <= thr, axis=0).tolist()
    within_1sigma = np.mean(z <= 1.0, axis=0)
    return {
        "mse_overall": mse,
        "mae_per_param": mae.tolist(),
        "rmse_per_param": rmse.tolist(),
        "residual_mean_per_param": resid_mean.tolist(),
        "residual_std_per_param": resid_std.tolist(),
        "sigma_mean_per_param": sigma_mean.tolist(),
        "sigma_median_per_param": sigma_median.tolist(),
        "sigma_min_per_param": sigma_min.tolist(),
        "sigma_max_per_param": sigma_max.tolist(),
        "sigma_percentiles_per_param": perc,
        "coverage_empirical": coverage,
        "coverage_within_1sigma": within_1sigma.tolist()
    }


def print_summary(score, summary):
    """
    Print key quantitative results in a concise and complete manner.

    Parameters
    ----------
    score : float
        Challenge success score (closer to 0 is better).
    summary : dict
        Summary statistics dictionary from summarize_statistics.
    """
    print("Challenge success score (higher closer to 0 is better): " + str(score))
    print("Overall MSE (both parameters): " + str(summary["mse_overall"]))
    print("MAE per parameter [Omega_m, S8]: " + str(summary["mae_per_param"]))
    print("RMSE per parameter [Omega_m, S8]: " + str(summary["rmse_per_param"]))
    print("Residual mean per parameter [Omega_m, S8]: " + str(summary["residual_mean_per_param"]))
    print("Residual std per parameter [Omega_m, S8]: " + str(summary["residual_std_per_param"]))
    print("Predicted sigma mean per parameter [Omega_m, S8]: " + str(summary["sigma_mean_per_param"]))
    print("Predicted sigma median per parameter [Omega_m, S8]: " + str(summary["sigma_median_per_param"]))
    print("Predicted sigma min per parameter [Omega_m, S8]: " + str(summary["sigma_min_per_param"]))
    print("Predicted sigma max per parameter [Omega_m, S8]: " + str(summary["sigma_max_per_param"]))
    print("Predicted sigma percentiles per parameter (1,5,10,25,50,75,90,95,99): " + str(summary["sigma_percentiles_per_param"]))
    print("Empirical coverage per parameter at nominal levels 50%, 68%, 80%, 90%, 95%:")
    print(str(summary["coverage_empirical"]))
    print("Fraction within 1 sigma [Omega_m, S8]: " + str(summary["coverage_within_1sigma"]))


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


def plot_pred_vs_true_with_errorbars(y_true, mu, sigma, save_dir, stamp):
    """
    Create a 1x2 subplot: predicted vs true with error bars for Omega_m and S8.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (N, 2), dimensionless.
    mu : np.ndarray
        Predicted means (N, 2), dimensionless.
    sigma : np.ndarray
        Predicted uncertainties (N, 2), dimensionless.
    save_dir : str
        Directory to save the plot.
    stamp : str
        Timestamp string for filename.

    Notes
    -----
    Axes labels include units; all are dimensionless.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = ["Omega_m", "S8"]
    for i in range(2):
        ax = axes[i]
        ax.errorbar(y_true[:, i], mu[:, i], yerr=sigma[:, i], fmt="o", ms=4, alpha=0.8, ecolor="gray", elinewidth=1, capsize=2)
        lo = float(np.min(y_true[:, i]))
        hi = float(np.max(y_true[:, i]))
        pad = 0.02 * (hi - lo + 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Truth " + names[i] + " (dimensionless)")
        ax.set_ylabel("Prediction " + names[i] + " (dimensionless)")
        ax.set_title(names[i] + " prediction")
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.suptitle("Predicted vs True with 1-sigma error bars")
    fig.tight_layout()
    fname = os.path.join(save_dir, "pred_vs_true_1_" + stamp + ".png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print("Saved plot: Predicted vs True with error bars for Omega_m and S8 at " + fname)


def plot_sigma_histograms(sigma, save_dir, stamp):
    """
    Plot histograms of predicted uncertainties for Omega_m and S8.

    Parameters
    ----------
    sigma : np.ndarray
        Predicted uncertainties (N, 2), dimensionless.
    save_dir : str
        Directory to save the plot.
    stamp : str
        Timestamp string for filename.

    Notes
    -----
    Uses linear scales since sigma is near small values. Axes are dimensionless.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = ["Omega_m", "S8"]
    for i in range(2):
        ax = axes[i]
        vals = sigma[:, i]
        bins = min(40, max(10, int(np.sqrt(vals.size))))
        ax.hist(vals, bins=bins, color="tab:blue", alpha=0.8, edgecolor="black")
        ax.set_xlabel("Predicted sigma " + names[i] + " (dimensionless)")
        ax.set_ylabel("Count")
        ax.set_title("Uncertainty distribution " + names[i])
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.suptitle("Predicted uncertainties")
    fig.tight_layout()
    fname = os.path.join(save_dir, "sigma_histograms_2_" + stamp + ".png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print("Saved plot: Histograms of predicted uncertainties for Omega_m and S8 at " + fname)


def plot_residuals_vs_truth(y_true, mu, save_dir, stamp):
    """
    Plot residuals (prediction - truth) vs truth for each parameter to reveal biases/heteroscedasticity.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (N, 2), dimensionless.
    mu : np.ndarray
        Predicted means (N, 2), dimensionless.
    save_dir : str
        Directory to save the plot.
    stamp : str
        Timestamp string for filename.

    Notes
    -----
    Axes are dimensionless.
    """
    resid = mu - y_true
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = ["Omega_m", "S8"]
    for i in range(2):
        ax = axes[i]
        ax.scatter(y_true[:, i], resid[:, i], s=18, alpha=0.8)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Truth " + names[i] + " (dimensionless)")
        ax.set_ylabel("Residual (Prediction - Truth) (dimensionless)")
        ax.set_title("Residuals vs Truth " + names[i])
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.suptitle("Residual analysis")
    fig.tight_layout()
    fname = os.path.join(save_dir, "residuals_vs_truth_3_" + stamp + ".png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print("Saved plot: Residuals vs Truth for Omega_m and S8 at " + fname)


def plot_calibration_coverage(y_true, mu, sigma, save_dir, stamp):
    """
    Plot empirical coverage versus nominal Gaussian coverage for Omega_m and S8.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (N, 2), dimensionless.
    mu : np.ndarray
        Predicted means (N, 2), dimensionless.
    sigma : np.ndarray
        Predicted uncertainties (N, 2), dimensionless.
    save_dir : str
        Directory to save the plot.
    stamp : str
        Timestamp string for filename.

    Notes
    -----
    The nominal coverage levels used are: 0.50, 0.68, 0.80, 0.90, 0.95.
    z-thresholds are precomputed for two-sided Gaussian coverage.
    """
    z = np.abs(mu - y_true) / (sigma + 1e-12)
    levels = [0.50, 0.68, 0.80, 0.90, 0.95]
    zthr = {
        0.50: 0.67448975,
        0.68: 1.0,
        0.80: 1.28155157,
        0.90: 1.64485363,
        0.95: 1.95996398
    }
    emp = np.zeros((len(levels), 2), dtype=np.float64)
    for idx, a in enumerate(levels):
        emp[idx, 0] = np.mean(z[:, 0] <= zthr[a])
        emp[idx, 1] = np.mean(z[:, 1] <= zthr[a])
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
    ax.plot(levels, levels, color="black", linestyle="--", linewidth=1, label="Ideal")
    ax.plot(levels, emp[:, 0], marker="o", label="Omega_m")
    ax.plot(levels, emp[:, 1], marker="s", label="S8")
    ax.set_xlabel("Nominal coverage (dimensionless)")
    ax.set_ylabel("Empirical coverage (dimensionless)")
    ax.set_title("Calibration coverage")
    ax.set_xlim(0.45, 0.98)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fname = os.path.join(save_dir, "calibration_coverage_4_" + stamp + ".png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print("Saved plot: Calibration coverage for Omega_m and S8 at " + fname)


def save_summary_json(summary_path, payload):
    """
    Save a JSON summary file to disk.

    Parameters
    ----------
    summary_path : str
        Output JSON file path.
    payload : dict
        Dictionary of metrics and metadata to save.
    """
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    """
    Main evaluation routine.

    Workflow
    --------
    1) Load latest validation artifacts (or generate via baseline pipeline if missing).
    2) Compute the challenge success score and residual/uncertainty statistics.
    3) Print detailed metrics to console.
    4) Save plots (PNG, dpi>=300) into $database_path (default 'data') with required filenames.
    5) Save a JSON summary under data/ with all computed metrics and paths used.

    Units
    -----
    - Omega_m, S8, and all uncertainties are dimensionless.
    """
    ensure_dir("data")
    ensure_dir(os.path.join("data", "artifacts"))
    stamp = timestamp()
    data = load_latest_validation_artifacts()
    y_true = data["y_true"]
    mu = data["pred_means"]
    sigma = data["pred_sigmas"]

    score = compute_challenge_score(y_true, mu, sigma, lam=1e3)
    summary = summarize_statistics(y_true, mu, sigma)
    print_summary(score, summary)

    plot_dir = get_database_path()
    plot_pred_vs_true_with_errorbars(y_true, mu, sigma, plot_dir, stamp)
    plot_sigma_histograms(sigma, plot_dir, stamp)
    plot_residuals_vs_truth(y_true, mu, plot_dir, stamp)
    plot_calibration_coverage(y_true, mu, sigma, plot_dir, stamp)

    summary_payload = {
        "challenge_score": score,
        "stats": summary,
        "artifacts_used": {
            "report_path": data["report_path"]
        },
        "timestamp": stamp
    }
    summary_path = os.path.join("data", "evaluation_summary_" + stamp + ".json")
    save_summary_json(summary_path, summary_payload)
    print("Saved evaluation summary JSON at " + summary_path)


if __name__ == "__main__":
    main()
