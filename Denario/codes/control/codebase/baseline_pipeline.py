# filename: codebase/baseline_pipeline.py
import os
import sys
import time
import json
import math
import datetime
import traceback
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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


class SimpleLabelScaler:
    """
    A minimal label scaler that standardizes and inverse-transforms labels.

    Attributes
    ----------
    mean_ : np.ndarray
        Per-feature mean computed on training labels (shape (n_features,)).
    scale_ : np.ndarray
        Per-feature standard deviation computed on training labels (shape (n_features,)).
    eps : float
        Small value added to scale to avoid division by zero.

    Methods
    -------
    fit(y)
        Compute mean and std of labels.
    transform(y)
        Standardize labels to zero mean and unit variance per feature.
    inverse_transform(y_scaled)
        Convert standardized labels back to the original scale.
    """
    def __init__(self, eps=1e-12):
        self.mean_ = None
        self.scale_ = None
        self.eps = eps

    def fit(self, y):
        y = np.asarray(y)
        self.mean_ = np.mean(y, axis=0)
        self.scale_ = np.std(y, axis=0)
        self.scale_ = np.where(self.scale_ < self.eps, self.eps, self.scale_)
        return self

    def transform(self, y):
        y = np.asarray(y)
        return (y - self.mean_) / self.scale_

    def inverse_transform(self, y_scaled):
        y_scaled = np.asarray(y_scaled)
        return y_scaled * self.scale_ + self.mean_


class Utility:
    """
    Utility functions for noise addition and file handling.

    Methods
    -------
    add_noise(data, mask, ng, pixel_size)
        Add Gaussian shape noise to convergence maps, scaled by ng and pixel size.
    ensure_dir(path)
        Ensure a directory exists, creating it if needed.
    timestamp()
        Return a compact timestamp string suitable for filenames.
    save_np(path, arr)
        Save numpy array to path.
    load_np(path)
        Load numpy array from path.
    """
    @staticmethod
    def add_noise(data, mask, ng, pixel_size=2.0):
        """
        Add Gaussian noise to noiseless convergence maps.

        Parameters
        ----------
        data : np.ndarray
            Noiseless convergence maps, shape (Ncosmo, Nsys, H, W), dimensionless.
        mask : np.ndarray
            Binary mask of observed pixels, shape (H, W).
        ng : float
            Galaxy number density in galaxies per arcmin^2.
        pixel_size : float
            Pixel size in arcminutes.

        Returns
        -------
        np.ndarray
            Noisy maps of the same shape as data. The noise amplitude is proportional
            to 1/sqrt(2*ng*pixel_size^2) and applied only on observed (mask==True) pixels.
        """
        if data.ndim != 4:
            raise ValueError("data must have shape (Ncosmo, Nsys, H, W)")
        if mask.ndim != 2:
            raise ValueError("mask must have shape (H, W)")
        sigma = 0.4 / math.sqrt(2.0 * float(ng) * float(pixel_size) * float(pixel_size))
        noise = np.random.randn(*data.shape).astype(np.float32) * np.float32(sigma)
        noise_mask = np.broadcast_to(mask[None, None, :, :], data.shape)
        noisy = data + noise * noise_mask.astype(np.float32)
        return noisy

    @staticmethod
    def ensure_dir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def timestamp():
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def save_np(path, arr):
        np.save(path, arr)

    @staticmethod
    def load_np(path):
        return np.load(path, allow_pickle=False)


class DataLoaderWL:
    """
    Data loader for weak-lensing maps following the starting kit file structure.

    Attributes
    ----------
    data_dir : str
        Directory containing input .npy files.
    use_public : bool
        If True, use public dataset filenames; else use sampled dataset filenames.
    ng : float
        Galaxy number density in galaxies per arcmin^2.
    pixel_size_arcmin : float
        Pixel size in arcminutes.
    mask : np.ndarray
        Binary mask array of shape (H, W).
    kappa : np.ndarray
        Noiseless convergence maps, shape (Ncosmo, Nsys, H, W).
    labels : np.ndarray
        Labels with 5 parameters, shape (Ncosmo, Nsys, 5).

    Methods
    -------
    load()
        Load mask, noiseless kappa, and labels into memory.
    add_noise_to_kappa()
        Return a noisy version of the convergence maps with the same shape.
    load_synthetic(Ncosmo, Nsys, H, W)
        Generate a synthetic dataset with correlated fields and realistic priors.
    """
    def __init__(self, data_dir, use_public=True, ng=30.0, pixel_size_arcmin=2.0):
        self.data_dir = data_dir
        self.use_public = use_public
        self.ng = float(ng)
        self.pixel_size_arcmin = float(pixel_size_arcmin)
        self.mask = None
        self.kappa = None
        self.labels = None

    def _find_file(self, candidates):
        for name in candidates:
            full = os.path.join(self.data_dir, name)
            if os.path.exists(full):
                return full
        raise FileNotFoundError("None of the candidate files exist in " + self.data_dir + " candidates: " + str(candidates))

    def load(self):
        mask_file = self._find_file(["WIDE12H_bin2_2arcmin_mask.npy", "mask.npy"])
        if self.use_public:
            kappa_candidates = ["WIDE12H_bin2_2arcmin_kappa.npy"]
            label_candidates = ["label.npy"]
        else:
            kappa_candidates = ["sampled_WIDE12H_bin2_2arcmin_kappa.npy"]
            label_candidates = ["sampled_label.npy"]
        kappa_file = self._find_file(kappa_candidates)
        label_file = self._find_file(label_candidates)
        self.mask = Utility.load_np(mask_file).astype(bool)
        mask_pixels = int(np.sum(self.mask))
        kappa_flat = Utility.load_np(kappa_file)
        labels = Utility.load_np(label_file)
        if labels.ndim != 3 or labels.shape[2] < 2:
            raise ValueError("Labels must have shape (Ncosmo, Nsys, >=5) or at least (Ncosmo, Nsys, 2).")
        if kappa_flat.ndim != 3:
            raise ValueError("Kappa flattened must have shape (Ncosmo, Nsys, Nmask_pixels).")
        if kappa_flat.shape[2] != mask_pixels:
            raise ValueError("Kappa flattened pixel count does not match mask pixels.")
        Ncosmo = kappa_flat.shape[0]
        Nsys = kappa_flat.shape[1]
        H, W = self.mask.shape
        kappa_full = np.zeros((Ncosmo, Nsys, H, W), dtype=np.float32)
        idx = np.where(self.mask)
        for i in range(Ncosmo):
            for j in range(Nsys):
                kappa_full[i, j][idx] = kappa_flat[i, j].astype(np.float32)
        self.kappa = kappa_full
        self.labels = labels.astype(np.float32)
        return self

    def add_noise_to_kappa(self):
        return Utility.add_noise(self.kappa, self.mask, self.ng, self.pixel_size_arcmin)

    def load_synthetic(self, Ncosmo=6, Nsys=12, H=1274, W=176):
        """
        Generate a synthetic dataset with realistic spatial correlations and priors.

        Parameters
        ----------
        Ncosmo : int
            Number of cosmologies.
        Nsys : int
            Number of nuisance realizations per cosmology.
        H : int
            Image height in pixels.
        W : int
            Image width in pixels.

        Returns
        -------
        DataLoaderWL
            Self with mask, kappa, and labels populated.

        Notes
        -----
        - Mask is generated with approximately 70 percent observed pixels and simple structure.
        - Convergence fields are generated by filtering white noise in Fourier space to impose
          spatial correlations. Field amplitude depends on Omega_m and S8 to create learnable signal.
        - Labels include 5 parameters: (Omega_m, S8, T_AGN, f0, delta_z).
        """
        self.mask = np.ones((H, W), dtype=bool)
        top_cut = int(0.05 * H)
        bottom_cut = int(0.05 * H)
        if top_cut > 0:
            self.mask[:top_cut, :] = False
        if bottom_cut > 0:
            self.mask[-bottom_cut:, :] = False
        rng = np.random.default_rng(20231111)
        for col in range(0, W, 16):
            if rng.random() < 0.25:
                r0 = rng.integers(low=int(0.1 * H), high=int(0.3 * H))
                r1 = rng.integers(low=int(0.7 * H), high=int(0.9 * H))
                self.mask[r0:r1, col:col + 8] = False
        kappa = np.zeros((Ncosmo, Nsys, H, W), dtype=np.float32)
        labels = np.zeros((Ncosmo, Nsys, 5), dtype=np.float32)
        om_vals = rng.uniform(0.1, 0.5, size=Ncosmo)
        s8_vals = rng.uniform(0.6, 1.0, size=Ncosmo)
        om_mean = float(np.mean(om_vals))
        s8_mean = float(np.mean(s8_vals))

        def correlated_field(h, w, seed=None):
            r = np.random.RandomState(seed)
            white = r.randn(h, w)
            F = np.fft.fft2(white)
            ky = np.fft.fftfreq(h)
            kx = np.fft.fftfreq(w)
            KX, KY = np.meshgrid(kx, ky)
            k2 = KX * KX + KY * KY
            k0 = 0.05
            filt = 1.0 / (1.0 + (k2 / (k0 * k0)))
            Ff = F * filt
            field = np.fft.ifft2(Ff).real
            field = field - np.mean(field)
            std = np.std(field)
            if std <= 0:
                std = 1.0
            field = field / std
            return field.astype(np.float32)

        for i in range(Ncosmo):
            base = correlated_field(H, W, seed=1000 + i)
            om = float(om_vals[i])
            s8 = float(s8_vals[i])
            amp = 0.02 + 0.08 * max(0.0, s8 - s8_mean) + 0.05 * max(0.0, om - om_mean)
            for j in range(Nsys):
                sys_field = correlated_field(H, W, seed=2000 + 50 * i + j)
                field = amp * base + 0.3 * amp * sys_field
                img = field.copy()
                img[~self.mask] = 0.0
                kappa[i, j] = img.astype(np.float32)
                T_AGN = rng.uniform(7.2, 8.5)
                f0 = rng.uniform(0.0, 0.0265)
                dz = rng.normal(0.0, 0.022)
                labels[i, j, 0] = om
                labels[i, j, 1] = s8
                labels[i, j, 2] = T_AGN
                labels[i, j, 3] = f0
                labels[i, j, 4] = dz
        self.kappa = kappa
        self.labels = labels
        return self


class WLDataset(Dataset):
    """
    Torch dataset for weak-lensing images and labels with on-the-fly normalization.

    Parameters
    ----------
    images : np.ndarray
        Image array of shape (N, H, W), dimensionless convergence kappa.
    labels : np.ndarray
        Label array of shape (N, 2) for (Omega_m, S8), dimensionless. Can be None for test-like usage.
    img_mean : float
        Global mean of training images for normalization.
    img_std : float
        Global std of training images for normalization.
    """
    def __init__(self, images, labels, img_mean, img_std):
        self.images = images.astype(np.float32)
        self.labels = None if labels is None else labels.astype(np.float32)
        self.img_mean = np.float32(img_mean)
        self.img_std = np.float32(img_std if img_std > 0 else 1.0)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        x = self.images[idx]
        x = (x - self.img_mean) / self.img_std
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x)
        if self.labels is None:
            return x
        y = torch.from_numpy(self.labels[idx])
        return x, y


class SimpleCNN(nn.Module):
    """
    Baseline CNN that maps a single-channel tall image to two means and two standard deviations.

    The network outputs:
    - means: predicted point estimates for (Omega_m, S8), dimensionless
    - sigmas: predicted one-standard-deviation uncertainties for (Omega_m, S8), dimensionless

    Positive sigmas are enforced via softplus with a small epsilon added.

    Parameters
    ----------
    num_targets : int
        Number of total outputs (2 means + 2 sigma) = 4.
    min_sigma : float
        Minimum sigma floor added to ensure numerical stability.

    Notes
    -----
    Input tensor shape: (batch, 1, H, W).
    """
    def __init__(self, num_targets=4, min_sigma=1e-6):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_targets)
        )
        self.softplus = nn.Softplus()
        self.min_sigma = float(min_sigma)

    def forward(self, x):
        z = self.features(x)
        z = self.pool(z)
        out = self.head(z)
        means = out[:, :2]
        raw_sig = out[:, 2:]
        sigmas = self.softplus(raw_sig) + self.min_sigma
        return means, sigmas


def challenge_loss(means_pred, sigmas_pred, truths, lam=1e3, eps=1e-12):
    """
    Loss aligned with the challenge metric (without the leading minus sign).

    For each sample i:
        loss_i = sum_k [ ( (mu_k - y_k)^2 / (sigma_k^2 + eps) ) + log(sigma_k^2 + eps) + lam * (mu_k - y_k)^2 ]

    Parameters
    ----------
    means_pred : torch.Tensor
        Predicted means, shape (batch, 2), dimensionless.
    sigmas_pred : torch.Tensor
        Predicted std deviations, shape (batch, 2), positive, dimensionless.
    truths : torch.Tensor
        Ground-truth labels, shape (batch, 2), dimensionless.
    lam : float
        Penalty coefficient for poor point estimates.
    eps : float
        Small value to avoid divide-by-zero and log(0).

    Returns
    -------
    torch.Tensor
        Scalar loss averaged over the batch.
    """
    resid_sq = (means_pred - truths) ** 2
    denom = sigmas_pred ** 2 + eps
    term1 = resid_sq / denom
    term2 = torch.log(denom)
    term3 = resid_sq * lam
    loss_per_sample = torch.sum(term1 + term2 + term3, dim=1)
    loss = torch.mean(loss_per_sample)
    return loss


def compute_score(true_vals, pred_means, pred_sigmas, lam=1e3, eps=1e-12):
    """
    Compute the challenge success score on CPU as defined.

    Parameters
    ----------
    true_vals : np.ndarray
        True labels, shape (N, 2), dimensionless.
    pred_means : np.ndarray
        Predicted means, shape (N, 2), dimensionless.
    pred_sigmas : np.ndarray
        Predicted sigmas, shape (N, 2), dimensionless.
    lam : float
        Penalty coefficient for point estimate errors.
    eps : float
        Small number to stabilize division and log.

    Returns
    -------
    float
        The averaged negative score across samples (closer to 0 is better).
    """
    resid_sq = (pred_means - true_vals) ** 2
    denom = pred_sigmas ** 2 + eps
    score_terms = resid_sq / denom + np.log(denom) + lam * resid_sq
    per_sample = -np.sum(score_terms, axis=1)
    return float(np.mean(per_sample))


def train_one_epoch(model, loader, optimizer, device, lam=1e3):
    """
    Train the model for one epoch without verbose batch outputs.

    Parameters
    ----------
    model : nn.Module
        The CNN model.
    loader : DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    device : torch.device
        Device for computation.
    lam : float
        Penalty coefficient for loss.

    Returns
    -------
    float
        Average training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    count = 0
    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        means, sigmas = model(x)
        loss = challenge_loss(means, sigmas, y, lam=lam)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        count += 1
    return total_loss / max(count, 1)


def validate_epoch(model, loader, device, lam=1e3):
    """
    Validate the model without verbose batch outputs.

    Parameters
    ----------
    model : nn.Module
        The CNN model.
    loader : DataLoader
        Validation data loader.
    device : torch.device
        Device for computation.
    lam : float
        Penalty coefficient for loss.

    Returns
    -------
    tuple
        (avg_loss, pred_means_all, pred_sigmas_all) where avg_loss is float, and predictions are np.ndarrays.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    means_list = []
    sigmas_list = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            means, sigmas = model(x)
            loss = challenge_loss(means, sigmas, y, lam=lam)
            total_loss += float(loss.detach().cpu().item())
            count += 1
            means_list.append(means.detach().cpu().numpy())
            sigmas_list.append(sigmas.detach().cpu().numpy())
    avg_loss = total_loss / max(count, 1)
    pred_means_all = np.concatenate(means_list, axis=0) if means_list else np.zeros((0, 2), dtype=np.float32)
    pred_sigmas_all = np.concatenate(sigmas_list, axis=0) if sigmas_list else np.zeros((0, 2), dtype=np.float32)
    return avg_loss, pred_means_all, pred_sigmas_all


def main():
    """
    Main entry point for preprocessing, training, and validation.

    Workflow
    --------
    1. Configure paths and settings, set seeds and device.
    2. Load data in starting-kit format, or generate synthetic data if files are missing.
    3. Add realistic shape noise to training maps.
    4. Split along nuisance dimension to train/val sets.
    5. Normalize images and standardize labels (cosmological parameters only).
    6. Train CNN with loss aligned with challenge metric, suppress verbose outputs.
    7. Evaluate on validation set and print key metrics.
    8. Save model, scaler, and predictions under data/.

    Notes
    -----
    - All quantities are dimensionless except ng (gal/arcmin^2) and pixel size (arcmin).
    - Prints essential metrics; suppresses batch-wise outputs.
    """
    try:
        set_global_seeds(12345)
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

        all_labels = loader.labels
        y_all = all_labels[:, :, :2]
        idx_sys = np.arange(Nsys)
        rng = np.random.default_rng(20240202)
        rng.shuffle(idx_sys)
        val_fraction = 0.2
        n_val_sys = max(1, int(round(val_fraction * Nsys)))
        val_sys_idx = idx_sys[:n_val_sys]
        train_sys_idx = idx_sys[n_val_sys:]

        X_train_4d = noisy_kappa[:, train_sys_idx, :, :]
        X_val_4d = noisy_kappa[:, val_sys_idx, :, :]
        y_train_4d = y_all[:, train_sys_idx, :]
        y_val_4d = y_all[:, val_sys_idx, :]

        Ntrain = X_train_4d.shape[0] * X_train_4d.shape[1]
        Nval = X_val_4d.shape[0] * X_val_4d.shape[1]

        X_train = X_train_4d.reshape(Ntrain, H, W)
        X_val = X_val_4d.reshape(Nval, H, W)
        y_train = y_train_4d.reshape(Ntrain, 2)
        y_val = y_val_4d.reshape(Nval, 2)

        img_mean = float(np.mean(X_train, dtype=np.float64))
        img_std = float(np.std(X_train, dtype=np.float64))
        if img_std <= 0:
            img_std = 1.0

        scaler = SimpleLabelScaler()
        scaler.fit(y_train)
        y_train_scaled = scaler.transform(y_train)
        y_val_scaled = scaler.transform(y_val)

        print("Train images shape: " + str(X_train.shape) + ", Val images shape: " + str(X_val.shape))
        print("Train labels shape: " + str(y_train.shape) + ", Val labels shape: " + str(y_val.shape))
        print("Image normalization: mean=" + str(img_mean) + ", std=" + str(img_std))
        print("Label scaler mean: " + str(scaler.mean_) + ", std: " + str(scaler.scale_))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: " + str(device))

        train_ds = WLDataset(X_train, y_train_scaled, img_mean, img_std)
        val_ds = WLDataset(X_val, y_val_scaled, img_mean, img_std)

        batch_size = 16
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

        model = SimpleCNN(num_targets=4, min_sigma=1e-6).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

        epochs = 8
        lam_penalty = 1e3
        best_val_loss = float("inf")
        best_state = None
        start_train = time.time()
        for e in range(epochs):
            tr_loss = train_one_epoch(model, train_loader, optimizer, device, lam=lam_penalty)
            val_loss, _, _ = validate_epoch(model, val_loader, device, lam=lam_penalty)
            print("Epoch " + str(e + 1) + "/" + str(epochs) + " - Train loss: " + str(tr_loss) + " - Val loss: " + str(val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        end_train = time.time()
        print("Training time (minutes): " + str(round((end_train - start_train) / 60.0, 3)))

        if best_state is not None:
            model.load_state_dict(best_state)

        val_loss, pred_means_scaled, pred_sigmas_scaled = validate_epoch(model, val_loader, device, lam=lam_penalty)
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

        print("Validation results summary:")
        print("Challenge success score (higher closer to 0 is better): " + str(score))
        print("MSE of point estimates: " + str(mse))
        print("Coverage within 1 sigma for Omega_m and S8: " + str(within_1sigma))
        print("Average predicted sigma for Omega_m and S8: " + str(avg_sigma))
        print("Median predicted sigma for Omega_m and S8: " + str(med_sigma))
        print("Residual mean for Omega_m and S8: " + str(mean_residual))
        print("Residual std for Omega_m and S8: " + str(std_residual))

        stamp = Utility.timestamp()
        model_path = os.path.join("data", "models", "baseline_cnn_" + stamp + ".pth")
        torch.save(model.state_dict(), model_path)
        scaler_path = os.path.join("data", "artifacts", "label_scaler_" + stamp + ".npz")
        np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_, img_mean=np.array([img_mean]), img_std=np.array([img_std]))
        pred_means_path = os.path.join("data", "artifacts", "val_pred_means_" + stamp + ".npy")
        pred_sigmas_path = os.path.join("data", "artifacts", "val_pred_sigmas_" + stamp + ".npy")
        y_val_path = os.path.join("data", "artifacts", "val_truth_" + stamp + ".npy")
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
                "learning_rate": 2e-4,
                "weight_decay": 1e-4,
                "lam_penalty": lam_penalty,
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
            }
        }
        report_path = os.path.join("data", "artifacts", "validation_report_" + stamp + ".json")
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