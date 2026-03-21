# ETTm1 configuration for JEPA (P2P only — no VQ / semantic tokens)
# Data paths use "./" relative to the JEPA/ directory.

config = {
    "path_save": "./output_model/JEPA/",
    "lr": 8e-5,
    "end_lr": 5e-6,
    "num_epochs": 21,
    "ema_momentum": 0.996,
    "weight_decay": 3e-3,
    "lr_pred": 6e-5,
    "weight_decay_pred": 1e-4,

    # masking
    "mask_ratio": 0.2,
    "masking_type": "multi_block",
    "num_blocks": 3,

    # encoder  (matches PatchTST architecture)
    "encoder_embed_dim": 128,
    "nhead": 16,
    "num_encoder_layers": 5,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "qk_scale": None,
    "drop_rate": 0.00,
    "attn_drop_rate": 0.00,
    "kernel_size": 6,
    "encoder_kernel_size": 6,
    "embed_bias": True,
    "encoder_embed_bias": True,
    "patch_size": 16,
    "patch_size_forcasting": 16,

    # predictor
    "predictor_embed_dim": 64,
    "predictor_nhead": 4,
    "predictor_num_layers": 2,

    # ── Datasets ──────────────────────────────────────────────────────────────
    "pretrain_dataset": "ettm1",
    "forecast_dataset": None,

    # data
    "checkpoint_save": 5000,
    "checkpoint_print": 30,
    "ratio_patches": 32,
    "batch_size": 64,

    # loader
    "clip_grad": 2.0,
    "warmup_ratio": 0.50,
    "ipe_scale": 1.25,

    # loss weights
    "vigreg_var": 0.25,   # VICReg variance term coefficient
    "vigreg_cov": 0.25,   # VICReg covariance term coefficient
    "val_prec": 0.2,
    "test_prec": 0.1,

    # ── ETTm1: 7 variables ────────────────────────────────────────────────────
    "timestampcols": ["date"],
    "input_variables": [
        "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT",
    ],
    "path_data": [
        "./data/ETTm1.csv"
    ],
    "chunk_size": 128,

    # forecasting downstream
    "epoch_t": 250,
    "context_t": 24,
    "horizon_t": 4,
    "input_variables_forcasting": [
        ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
    ],
    "val_prec_forcasting": 0.1,
    "test_prec_forcasting": 0.1,
    "window_step_forecasting": 1,
    "timestampcols_forcasting": ["date"],
    "path_data_forcasting": ["./data/ETTm1.csv"],
    "patches_to_forcast": 8,
    "patches_size_forecasting": 32,
    "lr_forcasting": 1e-3,
    "affine_revin": True,

    # ── Forecasting modes ─────────────────────────────────────────────────────
    "forecasting_modes": ["zeroshot"],

    # ── Monash pretraining ────────────────────────────────────────────────────
    "pretrain_on_monash": True,
    "monash_data_dir": "../Monash",   # relative to JEPA/
    "monash_min_len": 512,
}
