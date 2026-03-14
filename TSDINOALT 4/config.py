config = {

    # ── Task ─────────────────────────────────────────────────────────────────
    # "dino"  |  "classification"  |  "forecasting"
    "task": "dino",
    "seed": 0,
    "output_dir": "./checkpoints",
    "saveckp_freq": 10,
    "test_only": False,

    # ── Datasets ──────────────────────────────────────────────────────────────
    # Names must match keys in dataset_registry.py.
    # forecast_dataset defaults to pretrain_dataset when left as None.
    "pretrain_dataset":  "ettm1",
    "forecast_dataset":  None,

    # ── Data ──────────────────────────────────────────────────────────────────
    # Paths are relative to the TSDINOALT 4/ directory
    "data_path": "data/ETTm1.csv",
    "data_path_forecast_training": "data/ETTm1.csv",
    "data_path_forecast_test": "data/ETTm1.csv",
    "data_path_classification": "UCI HAR Dataset",
    "num_workers": 0,
    "batch_size_per_gpu": 64,

    # ── Model architecture ────────────────────────────────────────────────────
    "c_in": 7,          # number of input variables  (9 for UCI HAR)
    "patch_len": 12,
    "step_size": 12,    # stride between patches
    "num_patches": 32,  # window length in patches
    "n_layers": 5,
    "n_heads": 16,
    "embed_dim": 128,
    "d_ff": 512,
    "dropout": 0.1,
    "head_dropout": 0.1,
    "drop_path_rate": 0.1,

    # ── DINO head ─────────────────────────────────────────────────────────────
    "out_dim": 20000,
    "use_bn_in_head": False,
    "norm_last_layer": True,

    # ── DINO loss / teacher temperatures ─────────────────────────────────────
    "warmup_teacher_temp": 0.04,
    "teacher_temp": 0.04,
    "warmup_teacher_temp_epochs": 0,

    # ── EMA teacher ───────────────────────────────────────────────────────────
    "momentum_teacher": 0.9995,     # base EMA, cosine-scheduled up to 1.0

    # ── Optimizer ─────────────────────────────────────────────────────────────
    "optimizer": "adamw",           # "adamw" | "sgd"
    "lr": 0.0005,
    "min_lr": 1e-6,
    "warmup_epochs": 10,
    "weight_decay": 0.04,
    "weight_decay_end": 0.4,
    "clip_grad": 3.0,
    "use_fp16": False,
    "freeze_last_layer": 1,

    # ── DINO pretraining ──────────────────────────────────────────────────────
    "epochs": 100,

    # ── DWT defaults (shared across all dwt_* aug types) ─────────────────────
    #
    # Things to experiment with:
    #   wavelet family  – "haar" (sharpest transitions), "db4" (smooth, 4 vanishing moments),
    #                     "db8" (smoother), "sym4"/"sym8" (near-symmetric), "coif2" (compact)
    #   level           – how many frequency bands to create.
    #                     higher = coarser decomposition, more bands to manipulate.
    #                     rule of thumb: level ≤ log2(seq_len) - 1
    #   soft_threshold_sigma  – fraction of max(|coeff|) used as threshold per level.
    #                           0.1 = light denoising, 0.5 = aggressive denoising.
    #   zero_out_ratio        – fraction of finest-level coeffs to drop (0.0–1.0).
    #   finest_levels         – how many of the finest detail levels to perturb (1 = only finest).
    #   high_perturb_noise_range – (min_σ, max_σ) of Gaussian noise added to all detail coeffs.
    #
    "dwt_wavelet":                  "db4",          # try: "haar", "db4", "sym4", "coif2"
    "dwt_level":                    3,              # try: 2, 3, 4
    "dwt_soft_threshold_sigma":     0.3,            # try: 0.1, 0.3, 0.5
    "dwt_zero_out_ratio":           0.3,            # try: 0.2, 0.3, 0.5
    "dwt_finest_levels":            1,              # try: 1, 2
    "dwt_high_perturb_noise_range": (0.03, 0.08),  # try: (0.01, 0.05), (0.05, 0.15)
    "dwt_band_scale_approx_range":  (0.9,  1.1),
    "dwt_band_scale_detail_range":  (0.6,  1.4),

    # ── Augmentation views ────────────────────────────────────────────────────
    #
    # global_crops → seen by BOTH student and teacher (teacher only processes these).
    #                Keep crop_ratio = 1.0 for stable teacher targets.
    #
    # local_crops  → seen by the student only.
    #
    # Each entry is a dict:
    #   "type"       str | list[str]  – if list, one is drawn at random per sample
    #   "crop_ratio" float            – fraction of seq_len to keep (1.0 = no crop)
    #   Per-crop overrides of any dwt_* key above are also accepted.
    #
    # ── Available DWT aug types ───────────────────────────────────────────────
    #
    #  "dwt_soft_threshold"  — DWT → soft-threshold all detail coefficients → IDWT.
    #      Removes small-magnitude high-freq components; keeps dominant structure.
    #      Threshold per level = soft_threshold_sigma * max(|coeffs at that level|).
    #      → best for teacher (global views): produces clean, consistent targets.
    #      Override:  soft_threshold_sigma  (default from dwt_soft_threshold_sigma above)
    #      Example:   {"type": "dwt_soft_threshold", "crop_ratio": 1.0, "soft_threshold_sigma": 0.3}
    #
    #  "dwt_zero_out_detail" — DWT → randomly zero out zero_out_ratio of coefficients
    #      in the finest `finest_levels` detail arrays → IDWT.
    #      Drops stochastic fine-scale features; forces invariance to high-freq detail.
    #      → good student local view (type a).
    #      Overrides: zero_out_ratio, finest_levels
    #      Example:   {"type": "dwt_zero_out_detail", "crop_ratio": 1.0, "zero_out_ratio": 0.3, "finest_levels": 1}
    #
    #  "dwt_high_perturb"    — DWT → add Gaussian noise to ALL detail coefficients → IDWT.
    #      Corrupts detail at all scales while preserving approximation.
    #      → good student local view (type b).
    #      Override:  high_perturb_noise_range
    #      Example:   {"type": "dwt_high_perturb", "crop_ratio": 1.0}
    #
    #  "dwt_low_pass"        — DWT → zero all detail coefficients → IDWT (maximally smooth).
    #      Example:   {"type": "dwt_low_pass", "crop_ratio": 1.0}
    #
    #  "dwt_band_scale"      — DWT → randomly scale each frequency band → IDWT.
    #      Example:   {"type": "dwt_band_scale", "crop_ratio": 1.0}
    #
    # ─────────────────────────────────────────────────────────────────────────

    # 2 global views — teacher sees only these.
    # Both use soft thresholding with different strengths so the teacher
    # produces targets from slightly different levels of denoising.
    "global_crops": [
        {"type": "dwt_soft_threshold", "crop_ratio": 1.0, "soft_threshold_sigma": 0.2},
        {"type": "dwt_soft_threshold", "crop_ratio": 1.0, "soft_threshold_sigma": 0.4},
    ],

    # 8 local views — student sees these + the 2 global views above.
    # Each randomly applies either (a) zero-out-detail or (b) noise-on-detail,
    # with mild variation in the key hyperparameter across views so the student
    # sees a spectrum of perturbation strengths.
    "local_crops": [
        # ── type (a): stochastic zeroing of finest-level detail coefficients ──
        {"type": "dwt_zero_out_detail", "crop_ratio": 1.0, "zero_out_ratio": 0.2, "finest_levels": 1},
        {"type": "dwt_zero_out_detail", "crop_ratio": 1.0, "zero_out_ratio": 0.3, "finest_levels": 1},
        {"type": "dwt_zero_out_detail", "crop_ratio": 1.0, "zero_out_ratio": 0.4, "finest_levels": 1},
        {"type": "dwt_zero_out_detail", "crop_ratio": 1.0, "zero_out_ratio": 0.5, "finest_levels": 2},
        # ── type (b): Gaussian noise on all detail coefficients ──────────────
        {"type": "dwt_high_perturb",    "crop_ratio": 1.0, "high_perturb_noise_range": (0.02, 0.05)},
        {"type": "dwt_high_perturb",    "crop_ratio": 1.0, "high_perturb_noise_range": (0.03, 0.08)},
        {"type": "dwt_high_perturb",    "crop_ratio": 1.0, "high_perturb_noise_range": (0.05, 0.12)},
        {"type": "dwt_high_perturb",    "crop_ratio": 1.0, "high_perturb_noise_range": (0.08, 0.15)},
    ],

    # ── Downstream: Forecasting ───────────────────────────────────────────────
    "pred_len": 96,
    "epochs_forecasting": 10,
    "lr_forecasting": 0.001,
    "min_lr_forecasting": 1e-5,
    "parms_for_training_forecasting": ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
    "parms_for_testing_forecasting":  ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],

    # ── Downstream: Classification ────────────────────────────────────────────
    "n_classes": 6,
    "epochs_classification": 50,
    "lr_classification": 0.001,
    "min_lr_classification": 1e-6,
    "batch_size_classification": 64,
    "seq_len_classification": 128,  # UCI HAR fixed window
    "c_in_classification": 9,       # UCI HAR sensor count

    # checkpoint to load for downstream tasks  (0 = random init)
    "path_num": 0,

    # ── Distributed ───────────────────────────────────────────────────────────
    "dist_url": "env://",
}
