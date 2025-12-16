# filename: codebase/improved_training_pipeline.py
import os
import sys
import json
import time
import math
import traceback
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def set_global_seeds(seed):
    """
    Set global random seeds for reproducibility across numpy and torch.

    Parameters
    ----------
    seed : int
        The random seed to use for numpy and torch operations.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_sys_path():
    """
    Ensure that the 'codebase' directory is importable by appending it to sys.path.

    Returns
    -------
    None

    Notes
    -----
    This makes previously generated modules under codebase/ available for import.
    """
    base = os.getcwd()
    codebase_dir = os.path.join(base, "codebase")
    if os.path.isdir(codebase_dir) and (codebase_dir not in sys.path):
        sys.path.append(codebase_dir)

ensure_sys_path()
from baseline_pipeline import DataLoaderWL, Utility, SimpleLabelScaler, WLDataset, compute_score


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block to recalibrate channel-wise feature responses.

    Parameters
    ----------
    channels : int
        Number of input channels.
    reduction : int
        Reduction ratio for the bottleneck in the excitation MLP.

    Notes
    -----
    This block performs global average pooling, then a two-layer MLP with SiLU activation,
    ending with a sigmoid to generate channel-wise weights multiplied back to the input tensor.

    Units
    -----
    Dimensionless activations; no unit-carrying quantities.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.sig(w)
        return x * w

def best_group_count(c):
    """
    Choose a suitable number of groups for GroupNorm to evenly divide channels.

    Parameters
    ----------
    c : int
        Number of channels.

    Returns
    -------
    int
        Number of groups for GroupNorm.
    """
    for g in [32, 16, 8, 4, 2, 1]:
        if c % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """
    Residual block with GroupNorm and SiLU activations.

    Parameters
    ----------
    channels : int
        Number of input and output channels for the residual block.
    use_se : bool
        If True, include a squeeze-and-excitation block.

    Notes
    -----
    The block consists of two 3x3 conv layers with GroupNorm and SiLU activations.
    If use_se=True, an SEBlock is appended to recalibrate channels.

    Units
    -----
    All activations are dimensionless.
    """
    def __init__(self, channels, use_se=True):
        super(ResBlock, self).__init__()
        g = best_group_count(channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(g, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(g, channels)
        )
        self.act = nn.SiLU(inplace=True)
        self.use_se = use_se
        self.se = SEBlock(channels, reduction=16) if use_se else nn.Identity()

    def forward(self, x):
        y = self.net(x)
        y = self.se(y)
        y = y + x
        y = self.act(y)
        return y


class DownsampleBlock(nn.Module):
    """
    Downsampling block to reduce spatial dimensions and increase channels.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.

    Notes
    -----
    Uses a strided 3x3 convolution followed by GroupNorm and SiLU activation.

    Units
    -----
    Dimensionless.
    """
    def __init__(self, in_ch, out_ch):
        super(DownsampleBlock, self).__init__()
        g = best_group_count(out_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class EnhancedCNN(nn.Module):
    """
    Enhanced CNN for predicting two cosmological parameters and their uncertainties.

    Outputs
    -------
    means : torch.Tensor
        Predicted means for (Omega_m, S8), shape (B, 2), dimensionless.
    sigmas : torch.Tensor
        Predicted one-standard-deviation uncertainties (positive), shape (B, 2), dimensionless.

    Architecture
    ------------
    - Stem strided conv
    - Three stages with downsampling followed by residual blocks and squeeze-excitation
    - Adaptive average pooling to a fixed 4x4 spatial size
    - Fully-connected head to 4 outputs (2 means + 2 raw sigma), with softplus on sigma

    Parameters
    ----------
    min_sigma : float
        Minimum sigma added for numerical stability.
    """
    def __init__(self, min_sigma=1e-6):
        super(EnhancedCNN, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(best_group_count(32), 32),
            nn.SiLU(inplace=True)
        )
        self.stage1_down = DownsampleBlock(32, 64)
        self.stage1_res = nn.Sequential(ResBlock(64, use_se=True), ResBlock(64, use_se=True))
        self.stage2_down = DownsampleBlock(64, 128)
        self.stage2_res = nn.Sequential(ResBlock(128, use_se=True), ResBlock(128, use_se=True))
        self.stage3_down = DownsampleBlock(128, 192)
        self.stage3_res = nn.Sequential(ResBlock(192, use_se=True), ResBlock(192, use_se=True))
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 4 * 4, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 4)
        )
        self.softplus = nn.Softplus()
        self.min_sigma = float(min_sigma)

    def forward(self, x):
        z = self.stem(x)
        z = self.stage1_down(z)
        z = self.stage1_res(z)
        z = self.stage2_down(z)
        z = self.stage2_res(z)
        z = self.stage3_down(z)
        z = self.stage3_res(z)
        z = self.pool(z)
        out = self.head(z)
        means = out[:, :2]
        raw_sig = out[:, 2:]
        sigmas = self.softplus(raw_sig) + self.min_sigma
        return means, sigmas


def challenge_aligned_calibrated_loss(means_pred, sigmas_pred, truths, lam=1e3, alpha=0.05, eps=1e-12):
    """
    Loss aligned with the challenge metric plus a calibration penalty.

    For each sample i:
        base_i = sum_k [ (mu - y)^2 / (sigma^2 + eps) + log(sigma^2 + eps) + lam * (mu - y)^2 ]
        calib_i = mean_k [ ( |mu - y| / (sigma + eps) - 1 )^2 ]
        loss_i = base_i + alpha * calib_i

    Parameters
    ----------
    means_pred : torch.Tensor
        Predicted means, shape (B, 2).
    sigmas_pred : torch.Tensor
        Predicted stds, positive, shape (B, 2).
    truths : torch.Tensor
        Ground truths, shape (B, 2).
    lam : float
        Penalty coefficient on point estimate squared error.
    alpha : float
        Weight on calibration penalty encouraging |residual| ~ sigma.
    eps : float
        Small stability constant.

    Returns
    -------
    torch.Tensor
        Scalar loss averaged over the batch.
    """
    resid = means_pred - truths
    resid_sq = resid ** 2
    denom = sigmas_pred ** 2 + eps
    term1 = resid_sq / denom
    term2 = torch.log(denom)
    term3 = resid_sq * lam
    base = torch.sum(term1 + term2 + term3, dim=1)
    z = torch.abs(resid) / (sigmas_pred + eps)
    calib = torch.mean((z - 1.0) ** 2, dim=1)
    loss = torch.mean(base + alpha * calib)
    return loss


def train_one_epoch(model, loader, optimizer, device, lam=1e3, alpha=0.05, clip=1.0):
    """
    Train the model for one epoch with calibrated challenge-aligned loss.

    Parameters
    ----------
    model : nn.Module
        The EnhancedCNN model.
    loader : DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    device : torch.device
        Device for computation.
    lam : float
        Penalty coefficient for challenge-aligned term.
    alpha : float
        Weight for calibration penalty.
    clip : float
        Gradient clipping max norm.

    Returns
    -------
    float
        Average training loss.
    """
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        means, sigmas = model(x)
        loss = challenge_aligned_calibrated_loss(means, sigmas, y, lam=lam, alpha=alpha)
        loss.backward()
        if clip is not None and clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()
        total += float(loss.detach().cpu().item())
        n += 1
    return total / max(1, n)


def validate_epoch(model, loader, device, lam=1e3, alpha=0.05):
    """
    Validate the model without verbose outputs.

    Parameters
    ----------
    model : nn.Module
        The EnhancedCNN model.
    loader : DataLoader
        Validation data loader.
    device : torch.device
        Device for computation.
    lam : float
        Penalty coefficient for challenge-aligned term.
    alpha : float
        Weight for calibration penalty.

    Returns
    -------
    tuple
        (avg_loss, pred_means_all, pred_sigmas_all) where avg_loss is float, and predictions are np.ndarrays.
    """
    model.eval()
    total = 0.0
    n = 0
    means_list = []
    sigmas_list = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            means, sigmas = model(x)
            loss = challenge_aligned_calibrated_loss(means, sigmas, y, lam=lam, alpha=alpha)
            total += float(loss.detach().cpu().item())
            n += 1
            means_list.append(means.detach().cpu().numpy())
            sigmas_list.append(sigmas.detach().cpu().numpy())
    avg = total / max(1, n)
    pm = np.concatenate(means_list, axis=0) if means_list else np.zeros((0, 2), dtype=np.float32)
    ps = np.concatenate(sigmas_list, axis=0) if sigmas_list else np.zeros((0, 2), dtype=np.float32)
    return avg, pm, ps


def main():
    """
    Main routine for the improved model training and validation.

    Workflow
    --------
    1) Configure paths, seeds, and device, ensuring data/ folders exist.
    2) Load starting-kit formatted data or fallback to a synthetic dataset.
    3) Add realistic shape noise, split along nuisance realizations for train/val.
    4) Normalize images (global mean/std) and standardize labels for (Omega_m, S8).
    5) Train EnhancedCNN with AdamW, cosine annealing, gradient clipping.
    6) Validate, invert transforms, enforce non-negativity prior on uncertainties, compute metrics.
    7) Print key metrics concisely and save all artifacts under data/.

    Units
    -----
    - Omega_m and S8: dimensionless.
    - Predicted uncertainties: dimensionless.
    - ng: galaxies per arcmin^2.
    - pixel_size: arcminutes.
    """
    try:
        set_global_seeds(20231117)
        Utility.ensure_dir("data")
        Utility.ensure_dir(os.path.join("data", "models"))
        Utility.ensure_dir(os.path.join("data", "artifacts"))

        use_public_dataset = False
        data_dir = "input_data"
        ng = 30.0
        pixel_size_arcmin = 2.0

        print("Configured data directory: " + str(data_dir))
        print("Using public dataset: " + str(use_public_dataset))
        print("Galaxy density ng (gal/arcmin^2): " + str(ng))
        print("Pixel size (arcmin): " + str(pixel_size_arcmin))

        loader = DataLoaderWL(data_dir=data_dir, use_public=use_public_dataset, ng=ng, pixel_size_arcmin=pixel_size_arcmin)
        used_synthetic = False
        try:
            loader.load()
        except Exception as e:
            print("Data files not found or invalid. Falling back to synthetic dataset. Reason: " + str(e))
            loader.load_synthetic(Ncosmo=6, Nsys=12, H=1274, W=176)
            used_synthetic = True

        H, W = loader.mask.shape
        Ncosmo = loader.kappa.shape[0]
        Nsys = loader.kappa.shape[1]
        print("Loaded mask shape: " + str(loader.mask.shape))
        print("Loaded noiseless kappa shape: " + str(loader.kappa.shape))
        print("Loaded labels shape: " + str(loader.labels.shape))
        print("Map height H: " + str(H) + " pixels, width W: " + str(W) + " pixels")
        if used_synthetic:
            print("Using synthetic dataset with Ncosmo=" + str(Ncosmo) + " and Nsys=" + str(Nsys))

        t0 = time.time()
        noisy_kappa = loader.add_noise_to_kappa()
        t1 = time.time()
        print("Added noise to training maps in seconds: " + str(round(t1 - t0, 3)))

        
        labels_all = loader.labels[:, :, :2]

        # Avoid data contamination
        from sklearn.model_selection import train_test_split

        idx_sys = np.arange(Nsys)

        # --- MATCH train_test_split behavior EXACTLY ---
        val_fraction = 0.2
        seed = 5566

        # train_test_split uses floor(N * test_size), not round()
        train_sys_idx, val_sys_idx = train_test_split(
            idx_sys,
            test_size=val_fraction,
            random_state=seed,
            shuffle=True
        )

        # Now use the same indices for slicing (identical to your second block)
        X_train_4d = noisy_kappa[:, train_sys_idx, :, :]
        X_val_4d   = noisy_kappa[:, val_sys_idx,   :, :]

        y_train_4d = labels_all[:, train_sys_idx, :]
        y_val_4d   = labels_all[:, val_sys_idx,   :]

        Ntrain = X_train_4d.shape[0] * X_train_4d.shape[1]
        Nval   = X_val_4d.shape[0]   * X_val_4d.shape[1]

        X_train = X_train_4d.reshape(Ntrain, H, W)
        X_val   = X_val_4d.reshape(Nval,   H, W)

        y_train = y_train_4d.reshape(Ntrain, 2)
        y_val   = y_val_4d.reshape(Nval,   2)

        # idx_sys = np.arange(Nsys)
        # rng = np.random.default_rng(10101)
        # rng.shuffle(idx_sys)
        # val_fraction = 0.2
        # n_val_sys = max(1, int(round(val_fraction * Nsys)))
        # val_sys_idx = idx_sys[:n_val_sys]
        # train_sys_idx = idx_sys[n_val_sys:]

        # X_train_4d = noisy_kappa[:, train_sys_idx, :, :]
        # X_val_4d = noisy_kappa[:, val_sys_idx, :, :]
        # y_train_4d = labels_all[:, train_sys_idx, :]
        # y_val_4d = labels_all[:, val_sys_idx, :]

        # Ntrain = X_train_4d.shape[0] * X_train_4d.shape[1]
        # Nval = X_val_4d.shape[0] * X_val_4d.shape[1]
        # X_train = X_train_4d.reshape(Ntrain, H, W)
        # X_val = X_val_4d.reshape(Nval, H, W)
        # y_train = y_train_4d.reshape(Ntrain, 2)
        # y_val = y_val_4d.reshape(Nval, 2)

        img_mean = float(np.mean(X_train, dtype=np.float64))
        img_std = float(np.std(X_train, dtype=np.float64))
        if img_std <= 0:
            img_std = 1.0
        

        # scaler = SimpleLabelScaler()
        # scaler.fit(y_train)
        # y_train_scaled = scaler.transform(y_train)
        # y_val_scaled = scaler.transform(y_val)

        # print("Train images shape: " + str(X_train.shape) + ", Val images shape: " + str(X_val.shape))
        # print("Train labels shape: " + str(y_train.shape) + ", Val labels shape: " + str(y_val.shape))
        # print("Image normalization: mean=" + str(img_mean) + ", std=" + str(img_std))
        # print("Label scaler mean: " + str(scaler.mean_) + ", std: " + str(scaler.scale_))

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("Using device: " + str(device))


        # from torchvision import transforms # type: ignore

        # img_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[img_mean], std=[img_std])
        # ])

        # label_transform = lambda y: (y - scaler.mean_) / scaler.scale_

        # train_ds = WLDataset(X_train, y_train, transform=img_transform, label_transform=label_transform)
        # val_ds   = WLDataset(X_val,   y_val,   transform=img_transform, label_transform=label_transform)

        # train_ds = WLDataset(X_train, y_train_scaled, img_mean, img_std)
        # val_ds = WLDataset(X_val, y_val_scaled, img_mean, img_std)

        # ---- Label scaling ----
        scaler = SimpleLabelScaler()
        scaler.fit(y_train)

        print(f"Train images: {X_train.shape},  Val images: {X_val.shape}")
        print(f"Train labels: {y_train.shape}, Val labels: {y_val.shape}")
        print(f"Image normalization: mean={img_mean:.6f}, std={img_std:.6f}")
        print(f"Label scaler mean: {scaler.mean_}, std: {scaler.scale_}")

        # ---- Device ----
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # ---- Transforms ----
        from torchvision import transforms # type: ignore

        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[img_mean], std=[img_std]),
        ])

        def label_transform(y):
            return (y - scaler.mean_) / scaler.scale_

        # ---- Datasets ----
        train_ds = WLDataset(
            images=X_train,
            labels=y_train,
            transform=img_transform,
            label_transform=label_transform
        )

        val_ds = WLDataset(
            images=X_val,
            labels=y_val,
            transform=img_transform,
            label_transform=label_transform
        )



        batch_size = 16
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

        model = EnhancedCNN(min_sigma=1e-6).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-6)

        epochs = 10
        lam_penalty = 1e3
        calib_alpha = 0.05
        best_val_loss = float("inf")
        best_state = None
        start_train = time.time()
        for e in range(epochs):
            tr_loss = train_one_epoch(model, train_loader, optimizer, device, lam=lam_penalty, alpha=calib_alpha, clip=1.0)
            val_loss, _, _ = validate_epoch(model, val_loader, device, lam=lam_penalty, alpha=calib_alpha)
            print("Epoch " + str(e + 1) + "/" + str(epochs) + " - Train loss: " + str(tr_loss) + " - Val loss: " + str(val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            scheduler.step()
        end_train = time.time()
        print("Training time (minutes): " + str(round((end_train - start_train) / 60.0, 3)))

        if best_state is not None:
            model.load_state_dict(best_state)

        val_loss, pred_means_scaled, pred_sigmas_scaled = validate_epoch(model, val_loader, device, lam=lam_penalty, alpha=calib_alpha)
        pred_means = scaler.inverse_transform(pred_means_scaled)
        pred_sigmas = pred_sigmas_scaled * scaler.scale_

        negative_mask = pred_means - pred_sigmas < 0.0
        pred_sigmas[negative_mask] = pred_means[negative_mask]

        score = compute_score(true_vals=y_val, pred_means=pred_means, pred_sigmas=pred_sigmas, lam=lam_penalty)
        mse = float(np.mean((pred_means - y_val) ** 2))
        abs_err = np.abs(pred_means - y_val)
        within_1sigma = np.mean(abs_err <= pred_sigmas, axis=0)
        avg_sigma = np.mean(pred_sigmas, axis=0)
        med_sigma = np.median(pred_sigmas, axis=0)
        mean_residual = np.mean(pred_means - y_val, axis=0)
        std_residual = np.std(pred_means - y_val, axis=0)

        print("Validation results summary (improved model):")
        print("Challenge success score (higher closer to 0 is better): " + str(score))
        print("MSE of point estimates: " + str(mse))
        print("Coverage within 1 sigma [Omega_m, S8]: " + str(within_1sigma.tolist()))
        print("Average predicted sigma [Omega_m, S8]: " + str(avg_sigma.tolist()))
        print("Median predicted sigma [Omega_m, S8]: " + str(med_sigma.tolist()))
        print("Residual mean [Omega_m, S8]: " + str(mean_residual.tolist()))
        print("Residual std [Omega_m, S8]: " + str(std_residual.tolist()))

        stamp = Utility.timestamp()
        model_path = os.path.join("data", "models", "improved_cnn_" + stamp + ".pth")
        torch.save(model.state_dict(), model_path)
        scaler_path = os.path.join("data", "artifacts", "improved_label_scaler_" + stamp + ".npz")
        np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_, img_mean=np.array([img_mean]), img_std=np.array([img_std]))
        pred_means_path = os.path.join("data", "artifacts", "improved_val_pred_means_" + stamp + ".npy")
        pred_sigmas_path = os.path.join("data", "artifacts", "improved_val_pred_sigmas_" + stamp + ".npy")
        y_val_path = os.path.join("data", "artifacts", "improved_val_truth_" + stamp + ".npy")
        Utility.save_np(pred_means_path, pred_means)
        Utility.save_np(pred_sigmas_path, pred_sigmas)
        Utility.save_np(y_val_path, y_val)

        report = {
            "model_path": model_path,
            "scaler_path": scaler_path,
            "pred_means_path": pred_means_path,
            "pred_sigmas_path": pred_sigmas_path,
            "y_val_path": y_val_path,
            "val_score": score,
            "val_mse": mse,
            "coverage_within_1sigma": within_1sigma.tolist(),
            "avg_sigma": avg_sigma.tolist(),
            "med_sigma": med_sigma.tolist(),
            "mean_residual": mean_residual.tolist(),
            "std_residual": std_residual.tolist(),
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
                "lam_penalty": lam_penalty,
                "calibration_alpha": calib_alpha,
                "scheduler": "CosineAnnealingLR",
                "device": str(device),
                "H": H,
                "W": W,
                "Ncosmo": int(Ncosmo),
                "Nsys": int(Nsys),
                "train_samples": int(Ntrain),
                "val_samples": int(Nval),
                "img_mean": img_mean,
                "img_std": img_std,
                "used_synthetic": used_synthetic
            },
            "modifications": [
                "EnhancedCNN with residual blocks, GroupNorm, SiLU, squeeze-excitation, and adaptive pooling",
                "Challenge-aligned loss with added calibration penalty on normalized residuals",
                "AdamW optimizer with cosine annealing schedule",
                "Gradient clipping for stability",
                "Concise epoch-level logging and artifact saving"
            ]
        }
        report_path = os.path.join("data", "artifacts", "improved_validation_report_" + stamp + ".json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print("Artifacts saved:")
        print("Model: " + model_path)
        print("Scaler: " + scaler_path)
        print("Val predictions (means): " + pred_means_path)
        print("Val predictions (sigmas): " + pred_sigmas_path)
        print("Val truths: " + y_val_path)
        print("Report: " + report_path)

    except Exception as e:
        print("Fatal error encountered. Full traceback below:\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()