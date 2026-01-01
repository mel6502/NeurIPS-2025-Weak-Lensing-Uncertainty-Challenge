# filename: codebase/comparative_evaluation.py
import os
import sys
import json
import time
import math
import glob
import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt


def ensure_sys_path():
    """
    Ensure the codebase directory is in sys.path so previously generated modules can be imported.

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
    from evaluation import compute_challenge_score, summarize_statistics
except Exception:
    raise RuntimeError("Required module evaluation.py not found in codebase/. Please run prior steps to generate it.")



def ensure_dir(path):
    """
    Ensure a directory exists by creating it if missing.

    Parameters
    ----------
    path : str
        Directory path.
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
    Get plotting output directory from environment or default to 'data'.

    Returns
    -------
    str
        Path for saving plots.
    """
    dbp = os.environ.get("database_path", "data")
    ensure_dir(dbp)
    return dbp



def load_latest_report(pattern):
    """
    Load the most recent JSON report matching the provided glob pattern.

    Parameters
    ----------
    pattern : str
        Glob pattern, e.g., 'validation_report_*.json' or 'improved_validation_report_*.json'.

    Returns
    -------
    str
        Path to latest report JSON.

    Raises
    ------
    FileNotFoundError
        If no matching files are found.
    """
    artifacts_dir = os.path.join("data", "artifacts")
    ensure_dir(artifacts_dir)
    files = sorted(glob.glob(os.path.join(artifacts_dir, pattern)))
    if not files:
        raise FileNotFoundError("No report matching pattern found: " + pattern + " in " + artifacts_dir)
    return files[-1]



def load_predictions_from_report(report_path):
    """
    Load predictions and truths arrays from a report JSON file.

    Parameters
    ----------
    report_path : str
        Path to a validation report JSON.

    Returns
    -------
    dict
        Dictionary with keys:
        - pred_means: np.ndarray (N, 2)
        - pred_sigmas: np.ndarray (N, 2)
        - y_true: np.ndarray (N, 2)
        - report_path: str
    """
    with open(report_path, "r") as f:
        rep = json.load(f)
    pm = rep.get("pred_means_path", "")
    ps = rep.get("pred_sigmas_path", "")
    yt = rep.get("y_val_path", "")
    if not (os.path.exists(pm) and os.path.exists(ps) and os.path.exists(yt)):
        raise FileNotFoundError("Artifact paths listed in " + report_path + " do not exist.")
    pred_means = np.load(pm)
    pred_sigmas = np.load(ps)
    y_true = np.load(yt)
    return {"pred_means": pred_means, "pred_sigmas": pred_sigmas, "y_true": y_true, "report_path": report_path}



def compute_per_sample_score(y_true, mu, sigma, lam=1e3, eps=1e-12):
    """
    Compute the per-sample challenge success score (negative sum of terms).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (N, 2), dimensionless.
    mu : np.ndarray
        Predicted means (N, 2), dimensionless.
    sigma : np.ndarray
        Predicted uncertainties (N, 2), dimensionless.
    lam : float
        Penalty coefficient.
    eps : float
        Small constant for stability.

    Returns
    -------
    np.ndarray
        Per-sample score array of shape (N,), where higher and closer to 0 is better.
    """
    resid_sq = (mu - y_true) ** 2
    denom = sigma ** 2 + eps
    terms = resid_sq / denom + np.log(denom) + lam * resid_sq
    per_sample = -np.sum(terms, axis=1)
    return per_sample



def compute_per_sample_se_sum(y_true, mu):
    """
    Compute per-sample sum of squared errors across the two parameters.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (N, 2), dimensionless.
    mu : np.ndarray
        Predicted means (N, 2), dimensionless.

    Returns
    -------
    np.ndarray
        Per-sample sum of squared errors (N,), dimensionless.
    """
    resid_sq = (mu - y_true) ** 2
    return np.sum(resid_sq, axis=1)



def paired_t_stat(diffs):
    """
    Compute paired t-statistic and degrees of freedom for paired samples.

    Parameters
    ----------
    diffs : np.ndarray
        Differences x_i - y_i per sample (N,).

    Returns
    -------
    tuple
        (t_stat, dof, mean_diff, std_diff) where dof=N-1, std is sample std (ddof=1).

    Notes
    -----
    p-values are not returned to avoid distribution dependencies; a separate sign test p-value is computed.
    """
    diffs = np.asarray(diffs, dtype=np.float64)
    n = diffs.size
    dof = max(0, n - 1)
    if n <= 1:
        return float("nan"), int(dof), float("nan"), float("nan")
    mean_diff = float(np.mean(diffs))
    if n > 1:
        std_diff = float(np.std(diffs, ddof=1))
    else:
        std_diff = float("nan")
    if std_diff == 0.0 or math.isnan(std_diff):
        t_stat = float("inf") if mean_diff > 0 else float("-inf")
    else:
        t_stat = mean_diff / (std_diff / math.sqrt(n))
    return float(t_stat), int(dof), mean_diff, std_diff



def log_binom_pmf(n, k):
    """
    Compute the log of the binomial PMF with p=0.5 at k successes out of n.

    Parameters
    ----------
    n : int
        Number of trials.
    k : int
        Number of successes.

    Returns
    -------
    float
        log P(X=k) where X~Bin(n, 0.5).
    """
    return math.log(math.comb(n, k)) - n * math.log(2.0)



def binomial_cdf_two_sided_pvalue(n, k):
    """
    Compute a two-sided p-value for a sign test with X~Bin(n,0.5) observing k positives.

    Parameters
    ----------
    n : int
        Number of non-zero paired differences.
    k : int
        Number of positive differences.

    Returns
    -------
    float
        Two-sided p-value.

    Notes
    -----
    Uses log-sum-exp for numerical stability for n up to moderate sizes; for very large n,
    falls back to normal approximation with continuity correction.
    """
    if n <= 0:
        return 1.0
    k_eff = min(k, n - k)
    if n > 2000:
        mean = 0.5 * n
        var = 0.25 * n
        z = (k_eff + 0.5 - mean) / math.sqrt(var)
        tail = 0.5 * (1.0 - math.erf(abs(z) / math.sqrt(2.0)))
        return min(1.0, 2.0 * tail)
    logs = []
    for i in range(0, k_eff + 1):
        logs.append(log_binom_pmf(n, i))
    m = max(logs)
    s = 0.0
    for v in logs:
        s += math.exp(v - m)
    lower_tail = math.exp(m) * s
    p_two = min(1.0, 2.0 * lower_tail)
    return p_two



def paired_sign_test_pvalue(diffs):
    """
    Compute exact two-sided sign test p-value for paired differences.

    Parameters
    ----------
    diffs : np.ndarray
        Differences x_i - y_i per sample (N,).

    Returns
    -------
    tuple
        (p_value, n_used, k_positive) where n_used excludes zero differences.
    """
    diffs = np.asarray(diffs, dtype=np.float64)
    mask = diffs != 0.0
    d = diffs[mask]
    n = d.size
    if n == 0:
        return 1.0, 0, 0
    k = int(np.sum(d > 0.0))
    p = binomial_cdf_two_sided_pvalue(n, k)
    return float(p), int(n), int(k)



def print_model_summary(prefix, score, summary):
    """
    Print a concise summary for a model.

    Parameters
    ----------
    prefix : str
        Model name prefix, e.g., 'Baseline' or 'Improved'.
    score : float
        Challenge success score.
    summary : dict
        Summary statistics from summarize_statistics.
    """
    print(prefix + " model challenge success score (higher closer to 0 is better): " + str(score))
    print(prefix + " model overall MSE: " + str(summary["mse_overall"]))
    print(prefix + " model MAE per param [Omega_m, S8]: " + str(summary["mae_per_param"]))
    print(prefix + " model RMSE per param [Omega_m, S8]: " + str(summary["rmse_per_param"]))
    print(prefix + " model residual mean per param [Omega_m, S8]: " + str(summary["residual_mean_per_param"]))
    print(prefix + " model residual std per param [Omega_m, S8]: " + str(summary["residual_std_per_param"]))
    print(prefix + " model predicted sigma mean per param [Omega_m, S8]: " + str(summary["sigma_mean_per_param"]))
    print(prefix + " model predicted sigma median per param [Omega_m, S8]: " + str(summary["sigma_median_per_param"]))
    print(prefix + " model coverage within 1 sigma [Omega_m, S8]: " + str(summary["coverage_within_1sigma"]))



def plot_pred_vs_true_compare(y_true, mu_base, sig_base, mu_impr, sig_impr, save_dir, stamp):
    """
    Plot predicted vs true with error bars comparing baseline and improved models.

    Parameters
    ----------
    y_true : np.ndarray
        Truth (N, 2).
    mu_base : np.ndarray
        Baseline predicted means (N, 2).
    sig_base : np.ndarray
        Baseline predicted sigmas (N, 2).
    mu_impr : np.ndarray
        Improved predicted means (N, 2).
    sig_impr : np.ndarray
        Improved predicted sigmas (N, 2).
    save_dir : str
        Directory to save the plot.
    stamp : str
        Timestamp string.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    names = ["Omega_m", "S8"]
    for i in range(2):
        lo = float(np.min(y_true[:, i]))
        hi = float(np.max(y_true[:, i]))
        pad = 0.02 * (hi - lo + 1e-12)
        ax1 = axes[0, i]
        ax1.errorbar(y_true[:, i], mu_base[:, i], yerr=sig_base[:, i], fmt="o", ms=4, alpha=0.8, ecolor="gray", elinewidth=1, capsize=2, label="Baseline")
        ax1.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linestyle="--", linewidth=1)
        ax1.set_xlabel("Truth " + names[i] + " (dimensionless)")
        ax1.set_ylabel("Prediction (dimensionless)")
        ax1.set_title(names[i] + " baseline")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax2 = axes[1, i]
        ax2.errorbar(y_true[:, i], mu_impr[:, i], yerr=sig_impr[:, i], fmt="o", ms=4, alpha=0.8, ecolor="gray", elinewidth=1, capsize=2, label="Improved", color="tab:orange")
        ax2.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linestyle="--", linewidth=1)
        ax2.set_xlabel("Truth " + names[i] + " (dimensionless)")
        ax2.set_ylabel("Prediction (dimensionless)")
        ax2.set_title(names[i] + " improved")
        ax2.grid(True, linestyle="--", alpha=0.5)
    fig.suptitle("Predicted vs True with error bars (baseline vs improved)")
    fig.tight_layout()
    fname = os.path.join(save_dir, "pred_vs_true_compare_5_" + stamp + ".png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print("Saved plot: Predicted vs True with error bars, baseline vs improved, at " + fname)



def plot_sigma_hist_compare(sig_base, sig_impr, save_dir, stamp):
    """
    Plot histograms of predicted uncertainties comparing baseline and improved models.

    Parameters
    ----------
    sig_base : np.ndarray
        Baseline predicted sigmas (N, 2).
    sig_impr : np.ndarray
        Improved predicted sigmas (N, 2).
    save_dir : str
        Directory to save the plot.
    stamp : str
        Timestamp string.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = ["Omega_m", "S8"]
    for i in range(2):
        ax = axes[i]
        vals_b = sig_base[:, i]
        vals_i = sig_impr[:, i]
        all_vals = np.concatenate([vals_b, vals_i])
        bins = min(40, max(10, int(np.sqrt(all_vals.size))))
        rng = (float(np.min(all_vals)), float(np.max(all_vals)))
        ax.hist(vals_b, bins=bins, range=rng, color="tab:blue", alpha=0.6, edgecolor="black", label="Baseline")
        ax.hist(vals_i, bins=bins, range=rng, color="tab:orange", alpha=0.6, edgecolor="black", label="Improved")
        ax.set_xlabel("Predicted sigma " + names[i] + " (dimensionless)")
        ax.set_ylabel("Count")
        ax.set_title("Uncertainty distribution " + names[i])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
    fig.suptitle("Predicted uncertainties: baseline vs improved")
    fig.tight_layout()
    fname = os.path.join(save_dir, "sigma_hist_compare_6_" + stamp + ".png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print("Saved plot: Histograms of predicted uncertainties, baseline vs improved, at " + fname)



def plot_residuals_vs_truth_compare(y_true, mu_base, mu_impr, save_dir, stamp):
    """
    Plot residuals vs truth comparing baseline and improved models.

    Parameters
    ----------
    y_true : np.ndarray
        Truth (N, 2).
    mu_base : np.ndarray
        Baseline predicted means (N, 2).
    mu_impr : np.ndarray
        Improved predicted means (N, 2).
    save_dir : str
        Directory to save the plot.
    stamp : str
        Timestamp string.
    """
    resid_b = mu_base - y_true
    resid_i = mu_impr - y_true
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = ["Omega_m", "S8"]
    for i in range(2):
        ax = axes[i]
        ax.scatter(y_true[:, i], resid_b[:, i], s=24, alpha=0.8, label="Baseline")
        ax.scatter(y_true[:, i], resid_i[:, i], s=24, alpha=0.8, label="Improved")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Truth " + names[i] + " (dimensionless)")
        ax.set_ylabel("Residual (Prediction - Truth) (dimensionless)")
        ax.set_title("Residuals vs Truth " + names[i])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
    fig.suptitle("Residual analysis: baseline vs improved")
    fig.tight_layout()
    fname = os.path.join(save_dir, "residuals_vs_truth_compare_7_" + stamp + ".png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print("Saved plot: Residuals vs Truth, baseline vs improved, at " + fname)



def plot_calibration_coverage_compare(y_true, mu_b, sig_b, mu_i, sig_i, save_dir, stamp):
    """
    Plot calibration coverage (empirical vs nominal) for baseline and improved models.

    Parameters
    ----------
    y_true : np.ndarray
        Truth (N, 2).
    mu_b : np.ndarray
        Baseline predicted means (N, 2).
    sig_b : np.ndarray
        Baseline predicted sigmas (N, 2).
    mu_i : np.ndarray
        Improved predicted means (N, 2).
    sig_i : np.ndarray
        Improved predicted sigmas (N, 2).
    save_dir : str
        Directory to save the plot.
    stamp : str
        Timestamp string.
    """
    levels = [0.50, 0.68, 0.80, 0.90, 0.95]
    zthr = {
        0.50: 0.67448975,
        0.68: 1.0,
        0.80: 1.28155157,
        0.90: 1.64485363,
        0.95: 1.95996398
    }
    zb = np.abs(mu_b - y_true) / (sig_b + 1e-12)
    zi = np.abs(mu_i - y_true) / (sig_i + 1e-12)
    emp_b = np.zeros((len(levels), 2), dtype=np.float64)
    emp_i = np.zeros((len(levels), 2), dtype=np.float64)
    for idx, a in enumerate(levels):
        thr = zthr[a]
        emp_b[idx, 0] = np.mean(zb[:, 0] <= thr)
        emp_b[idx, 1] = np.mean(zb[:, 1] <= thr)
        emp_i[idx, 0] = np.mean(zi[:, 0] <= thr)
        emp_i[idx, 1] = np.mean(zi[:, 1] <= thr)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = ["Omega_m", "S8"]
    for j in range(2):
        ax = axes[j]
        ax.plot(levels, levels, color="black", linestyle="--", linewidth=1, label="Ideal")
        ax.plot(levels, emp_b[:, j], marker="o", label="Baseline")
        ax.plot(levels, emp_i[:, j], marker="s", label="Improved")
        ax.set_xlabel("Nominal coverage (dimensionless)")
        ax.set_ylabel("Empirical coverage (dimensionless)")
        ax.set_title("Calibration " + names[j])
        ax.set_xlim(0.45, 0.98)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
    fig.suptitle("Calibration coverage: baseline vs improved")
    fig.tight_layout()
    fname = os.path.join(save_dir, "calibration_coverage_compare_8_" + stamp + ".png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print("Saved plot: Calibration coverage, baseline vs improved, at " + fname)



def main():
    """
    Main comparative evaluation routine.

    Workflow
    --------
    1) Load latest baseline and improved validation artifacts; auto-generate if missing.
    2) Compute challenge score and detailed statistics for each model.
    3) Compute per-sample paired comparisons using t-stat and exact sign test.
    4) Save comparative plots to $database_path with required filenames and print descriptions.
    5) Save a JSON summary under data/ with all computed metrics and plot paths.

    Units
    -----
    All reported quantities (Omega_m, S8, and uncertainties) are dimensionless.
    """
    ensure_dir("data")
    ensure_dir(os.path.join("data", "artifacts"))
    stamp = timestamp()

    have_baseline = True
    have_improved = True
    try:
        rep_base_path = load_latest_report("validation_report_*.json")
    except FileNotFoundError:
        have_baseline = False
    try:
        rep_impr_path = load_latest_report("improved_validation_report_*.json")
    except FileNotFoundError:
        have_improved = False

    if not have_baseline:
        try:
            from baseline_pipeline import main as baseline_main
            baseline_main()
            rep_base_path = load_latest_report("validation_report_*.json")
        except Exception as e:
            raise RuntimeError("Baseline artifacts missing and baseline training failed. Error: " + str(e))
    if not have_improved:
        try:
            from improved_training_pipeline import main as improved_main
            improved_main()
            rep_impr_path = load_latest_report("improved_validation_report_*.json")
        except Exception as e:
            raise RuntimeError("Improved artifacts missing and improved training failed. Error: " + str(e))

    base = load_predictions_from_report(rep_base_path)
    impr = load_predictions_from_report(rep_impr_path)

    yb = base["y_true"]
    mb = base["pred_means"]
    sb = base["pred_sigmas"]

    yi = impr["y_true"]
    mi = impr["pred_means"]
    si = impr["pred_sigmas"]

    if yb.shape[0] != yi.shape[0]:
        nmin = min(yb.shape[0], yi.shape[0])
        yb = yb[:nmin]
        mb = mb[:nmin]
        sb = sb[:nmin]
        yi = yi[:nmin]
        mi = mi[:nmin]
        si = si[:nmin]

    score_b = compute_challenge_score(yb, mb, sb, lam=1e3)
    score_i = compute_challenge_score(yi, mi, si, lam=1e3)
    summ_b = summarize_statistics(yb, mb, sb)
    summ_i = summarize_statistics(yi, mi, si)

    print_model_summary("Baseline", score_b, summ_b)
    print_model_summary("Improved", score_i, summ_i)

    ps_b = compute_per_sample_score(yb, mb, sb, lam=1e3)
    ps_i = compute_per_sample_score(yi, mi, si, lam=1e3)
    diffs_score = ps_i - ps_b
    t_stat_s, dof_s, mean_diff_s, std_diff_s = paired_t_stat(diffs_score)
    p_sign_s, n_used_s, k_pos_s = paired_sign_test_pvalue(diffs_score)

    se_b = compute_per_sample_se_sum(yb, mb)
    se_i = compute_per_sample_se_sum(yi, mi)
    diffs_se = se_i - se_b
    t_stat_e, dof_e, mean_diff_e, std_diff_e = paired_t_stat(diffs_se)
    p_sign_e, n_used_e, k_pos_e = paired_sign_test_pvalue(diffs_se)

    print("Comparative statistics (Improved minus Baseline):")
    print("Score per-sample diff: mean=" + str(mean_diff_s) + ", std=" + str(std_diff_s) + ", t=" + str(t_stat_s) + ", dof=" + str(dof_s) + ", sign-test p=" + str(p_sign_s) + ", n_used=" + str(n_used_s) + ", k_positive=" + str(k_pos_s))
    print("Sum of squared errors per-sample diff: mean=" + str(mean_diff_e) + ", std=" + str(std_diff_e) + ", t=" + str(t_stat_e) + ", dof=" + str(dof_e) + ", sign-test p=" + str(p_sign_e) + ", n_used=" + str(n_used_e) + ", k_positive=" + str(k_pos_e))

    cov_b = summ_b["coverage_within_1sigma"]
    cov_i = summ_i["coverage_within_1sigma"]
    print("Coverage within 1 sigma baseline [Omega_m, S8]: " + str(cov_b))
    print("Coverage within 1 sigma improved [Omega_m, S8]: " + str(cov_i))
    print("Coverage change (improved - baseline): " + str([float(cov_i[0] - cov_b[0]), float(cov_i[1] - cov_b[1])]))

    plot_dir = get_database_path()
    plot_pred_vs_true_compare(yb, mb, sb, mi, si, plot_dir, stamp)
    plot_sigma_hist_compare(sb, si, plot_dir, stamp)
    plot_residuals_vs_truth_compare(yb, mb, mi, plot_dir, stamp)
    plot_calibration_coverage_compare(yb, mb, sb, mi, si, plot_dir, stamp)

    summary_payload = {
        "timestamp": stamp,
        "artifacts_used": {
            "baseline_report": base["report_path"],
            "improved_report": impr["report_path"]
        },
        "baseline": {
            "score": score_b,
            "stats": summ_b
        },
        "improved": {
            "score": score_i,
            "stats": summ_i
        },
        "paired_comparison": {
            "per_sample_score_diff": {
                "mean": mean_diff_s,
                "std": std_diff_s,
                "t_stat": t_stat_s,
                "degrees_of_freedom": dof_s,
                "sign_test_p_value": p_sign_s,
                "n_used": n_used_s,
                "k_positive": k_pos_s
            },
            "per_sample_se_sum_diff": {
                "mean": mean_diff_e,
                "std": std_diff_e,
                "t_stat": t_stat_e,
                "degrees_of_freedom": dof_e,
                "sign_test_p_value": p_sign_e,
                "n_used": n_used_e,
                "k_positive": k_pos_e
            }
        },
        "plots": {
            "pred_vs_true_compare": os.path.join(plot_dir, "pred_vs_true_compare_5_" + stamp + ".png"),
            "sigma_hist_compare": os.path.join(plot_dir, "sigma_hist_compare_6_" + stamp + ".png"),
            "residuals_vs_truth_compare": os.path.join(plot_dir, "residuals_vs_truth_compare_7_" + stamp + ".png"),
            "calibration_coverage_compare": os.path.join(plot_dir, "calibration_coverage_compare_8_" + stamp + ".png")
        }
    }
    summary_path = os.path.join("data", "comparative_evaluation_" + stamp + ".json")
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)
    print("Saved comparative evaluation summary JSON at " + summary_path)



if __name__ == "__main__":
    main()
