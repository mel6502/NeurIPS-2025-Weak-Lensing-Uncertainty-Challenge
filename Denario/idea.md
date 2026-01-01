Title: Nyx — Score-Consistent Spectro-Scattering Fusion with Noise-Replication Consistency for Weak-Lensing Cosmology

Idea:
Train a heteroscedastic predictor that directly minimizes the exact per-sample score (Gaussian NLL + λ·MSE) while fusing mask-safe spectral/scattering summaries with an anisotropic image CNN, and stabilize uncertainty learning via a noise-replication consistency regularizer. Post-train, jointly calibrate the ensemble’s mean and variance to minimize the validation score.

Core components:
- End-to-end score-consistent training:
  - Loss per sample: L = Σ_k [(y_k−μ_k)^2/σ_k^2 + log σ_k^2 + λ·(y_k−μ_k)^2], k ∈ {Ω_m, S_8}, with λ = 10^3.
  - Outputs: μ_Ωm, μ_S8, log σ_Ωm^2, log σ_S8^2; unconstrained log-variances via log σ_k^2 = log(σ_min^2 + softplus(z_k)), σ_min ≈ 1e−4.

- Dual-branch fusion:
  1) Image branch: anisotropic ConvNeXt-lite with dilated/atrous convs and axial depthwise convs to capture long-range structure; anti-aliased downsampling; GroupNorm; mixed precision.
  2) Mask-safe spectral–scattering branch:
     - Multitaper 2D power spectrum (DPSS tapers + Tukey apodization) with elliptic binning to ~128D.
     - Second-order wavelet scattering (Kymatio) with anisotropic Morlet wavelets to ~128D, capturing non-Gaussian morphology.
  - Late fusion via gated FiLM or cross-attention into a 256D embedding; small MLP head outputs μ and log σ^2.

- Noise-Replication Consistency (NRC):
  - For each training map x, generate K noisy replicas x^(k) by re-drawing or jittering noise consistent with the data’s noise model (or calibrated Gaussian jitter if only approximate).
  - Regularizer: L_NRC = Σ_k ||μ(x^(k)) − μ(x)||^2 + α·|Var_k[μ(x^(k))] − σ(x)^2|, encouraging σ to match the empirical variability across replicas and stabilizing heteroscedastic learning.

- Practical training:
  - Stripe multi-crop during training (e.g., 352×176 overlapping stripes) with attention pooling; full-image inference.
  - Physics-safe augmentations: horizontal/vertical flips, 180° rotations, sub-pixel translations with reflection padding, mild noise-level jitter; E/B-mode parity preserved as appropriate.
  - Early stopping on the full validation score; EMA/SWA weights; AdamW with cosine schedule and warmup.

- Ensemble and joint calibration:
  - Train 4–5 lightweight seeds; ensemble μ by averaging and combine σ via total variance (mean σ^2 plus between-model variance of μ).
  - On a held-out calibration split, jointly tune per-parameter affine μ corrections (a_k, b_k) and temperature scales τ_k on the ensemble outputs to minimize the validation score; apply once to test-time predictions.

- Diagnostics and deliverables:
  - Provide a single notebook implementing: spectral/scattering precomputation cache, dual-branch fusion, score-consistent training with NRC, ensemble and joint calibration; include PIT/z-QQ, sharpness–calibration curves, ablations (no spectral branch, no NRC, no multi-crop).
  - Include a concise list of changes relative to the baseline kit (architecture, loss, uncertainty parameterization, augmentations, training protocol, calibration, and ensembling).

Expected outcome:
- Bias-suppressed point estimates μ through λ-weighted fitting while learning informative, per-map σ that reflect noise-induced variability.
- Robustness to survey edges and anisotropy via multitaper spectra and scattering fusion.
- Improved score through end-to-end optimization, replica-consistency regularization, and ensemble-level joint calibration.