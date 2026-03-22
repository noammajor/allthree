config = {
    # ── Datasets ──────────────────────────────────────────────────────────────
    # pretrain_dataset: any key from dataset_registry, or "monash" for Monash pretraining.
    # forecast_dataset: defaults to pretrain_dataset when None.
    "pretrain_dataset":  "ettm1",
    "forecast_dataset":  None,

    # ── Monash pretraining ────────────────────────────────────────────────────
    # Set pretrain_on_monash=True (or pretrain_dataset="monash") to pretrain on
    # all Monash .tsf files instead of a single CSV.
    "pretrain_on_monash": False,
    "monash_data_dir":    "../Monash",   # relative to PatchTST_self_supervised/
    "monash_min_len":     512,           # skip series shorter than this

    # ── Input ─────────────────────────────────────────────────────────────────
    "context_points": 512,    # sequence length fed to the encoder
    "target_points":  96,     # forecast horizon (used during fine-tuning)
    "features":       "M",    # "M" = multivariate, "S" = univariate

    # ── Patch ─────────────────────────────────────────────────────────────────
    "patch_len": 12,
    "stride":    12,

    # ── Model ─────────────────────────────────────────────────────────────────
    "n_layers":     3,
    "n_heads":      16,
    "d_model":      128,
    "d_ff":         512,
    "dropout":      0.2,
    "head_dropout": 0.2,

    # ── Pretraining ───────────────────────────────────────────────────────────
    "mask_ratio":        0.4,
    "n_epochs_pretrain": 10,
    "batch_size":        64,
    "revin":             True,   # reversible instance normalization

    # ── Misc ──────────────────────────────────────────────────────────────────
    "pretrained_model_id": 1,
    "model_type":          "based_model",
    "num_workers":         0,
}
