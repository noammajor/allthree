"""
Unified training + forecasting runner for:
  - dino      (TSDINOALT 4)
  - jepa      (Discrete_JEPA / DiscreteJEPA)
  - patchtst  (PatchTST_self_supervised)

Usage
-----
  python Train_and_downstream.py --model dino
  python Train_and_downstream.py --model jepa  --skip_train true
  python Train_and_downstream.py --model patchtst

Colab
-----
  !python Train_and_downstream.py --model dino
  or call run(model="dino", skip_train=False) directly after importing.
"""

import os, sys, copy, argparse
import subprocess
from types import SimpleNamespace
from pathlib import Path

# Make sure the project root (where dataset_registry.py lives) is importable
_PROJECT_ROOT = str(Path(__file__).parent.resolve())
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataset_registry import get_dataset_info


# ── helpers ──────────────────────────────────────────────────────────────────

def _add_path(p):
    """Prepend p to sys.path if not already present."""
    p = str(Path(p).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)

def _config_to_dino_args(cfg):
    """
    Convert the DINO config dict (TSDINOALT 4/config.py) into the
    SimpleNamespace that train_TS_DINO / test_run expect.
    """
    local_crops = cfg.get("local_crops", [])
    global_crops = cfg.get("global_crops", [])

    args = SimpleNamespace(
        # ── task ──────────────────────────────────────────────────────────
        task                        = cfg.get("task", "dino"),
        test_only                   = cfg.get("test_only", False),
        seed                        = cfg.get("seed", 0),
        output_dir                  = cfg.get("output_dir", "./checkpoints"),
        saveckp_freq                = cfg.get("saveckp_freq", 10),

        # ── data ──────────────────────────────────────────────────────────
        data_path                   = cfg.get("data_path", "UCI HAR Dataset"),
        data_path_forecast_training = cfg.get("data_path_forecast_training", ""),
        data_path_forecast_test     = cfg.get("data_path_forecast_test", ""),
        data_path_classification    = cfg.get("data_path_classification", "UCI HAR Dataset"),
        num_workers                 = cfg.get("num_workers", 0),
        batch_size_per_gpu          = cfg.get("batch_size_per_gpu", 64),

        # ── model architecture ────────────────────────────────────────────
        c_in                        = cfg.get("c_in", 7),
        patch_len                   = cfg.get("patch_len", 12),
        step_size                   = cfg.get("step_size", 12),
        num_patches                 = cfg.get("num_patches", 32),
        n_layers                    = cfg.get("n_layers", 5),
        n_heads                     = cfg.get("n_heads", 16),
        embed_dim                   = cfg.get("embed_dim", 128),
        d_ff                        = cfg.get("d_ff", 512),
        dropout                     = cfg.get("dropout", 0.1),
        head_dropout                = cfg.get("head_dropout", 0.1),
        drop_path_rate              = cfg.get("drop_path_rate", 0.1),

        # ── DINO head ─────────────────────────────────────────────────────
        out_dim                     = cfg.get("out_dim", 20000),
        use_bn_in_head              = cfg.get("use_bn_in_head", False),
        norm_last_layer             = cfg.get("norm_last_layer", True),

        # ── DINO loss / temperatures ──────────────────────────────────────
        warmup_teacher_temp         = cfg.get("warmup_teacher_temp", 0.04),
        teacher_temp                = cfg.get("teacher_temp", 0.04),
        warmup_teacher_temp_epochs  = cfg.get("warmup_teacher_temp_epochs", 0),

        # ── EMA teacher ───────────────────────────────────────────────────
        momentum_teacher            = cfg.get("momentum_teacher", 0.9995),

        # ── optimizer ─────────────────────────────────────────────────────
        optimizer                   = cfg.get("optimizer", "adamw"),
        lr                          = cfg.get("lr", 0.0005),
        min_lr                      = cfg.get("min_lr", 1e-6),
        warmup_epochs               = cfg.get("warmup_epochs", 10),
        weight_decay                = cfg.get("weight_decay", 0.04),
        weight_decay_end            = cfg.get("weight_decay_end", 0.4),
        clip_grad                   = cfg.get("clip_grad", 3.0),
        use_fp16                    = cfg.get("use_fp16", False),
        freeze_last_layer           = cfg.get("freeze_last_layer", 1),

        # ── training schedule ─────────────────────────────────────────────
        epochs                      = cfg.get("epochs", 100),

        # ── augmentation (derived from crop specs) ────────────────────────
        # local_crops_number  = crop ratio of the first local crop
        # transformation_group_size = total number of local crops
        local_crops_number          = local_crops[0]["crop_ratio"] if local_crops else 0.5,
        transformation_group_size   = len(local_crops) if local_crops else 2,

        # ── distributed (defaults for single-GPU / CPU) ───────────────────
        dist_url                    = cfg.get("dist_url", "env://"),
        gpu                         = None,
        rank                        = 0,
        world_size                  = 1,
        dist_backend                = "nccl",

        # ── downstream: forecasting ───────────────────────────────────────
        pred_len                            = cfg.get("pred_len", 96),
        epochs_forecasting                  = cfg.get("epochs_forecasting", 10),
        lr_forecasting                      = cfg.get("lr_forecasting", 0.001),
        min_lr_forecasting                  = cfg.get("min_lr_forecasting", 1e-5),
        parms_for_training_forecasting      = cfg.get("parms_for_training_forecasting", []),
        parms_for_testing_forecasting       = cfg.get("parms_for_testing_forecasting", []),
        path_num                            = cfg.get("path_num", 0),

        # ── downstream: classification ────────────────────────────────────
        n_classes                   = cfg.get("n_classes", 6),
        epochs_classification       = cfg.get("epochs_classification", 50),
        lr_classification           = cfg.get("lr_classification", 0.001),
        min_lr_classification       = cfg.get("min_lr_classification", 1e-6),
        batch_size_classification   = cfg.get("batch_size_classification", 64),
        seq_len_classification      = cfg.get("seq_len_classification", 128),
        c_in_classification         = cfg.get("c_in_classification", 9),
    )
    return args


# ── DINO ──────────────────────────────────────────────────────────────────────

def run_dino(skip_train: bool = False,
             pretrain_dataset: str = None,
             forecast_dataset: str = None):
    dino_dir = Path(__file__).parent / "TSDINOALT 4"
    _add_path(dino_dir)

    from config import config as dino_cfg
    import main as dino_main

    dino_cfg = dict(dino_cfg)
    pretrain_on_monash = dino_cfg.get('pretrain_on_monash', False)

    # Resolve forecast dataset (always needed for downstream)
    forecast_dataset = forecast_dataset or dino_cfg.get("forecast_dataset")
    if pretrain_on_monash:
        # No pretrain CSV needed; derive c_in from forecast dataset
        if forecast_dataset is None:
            raise ValueError("forecast_dataset must be set when pretrain_on_monash=True")
        ds_fore = get_dataset_info(forecast_dataset)
        dino_cfg["c_in"] = ds_fore["c_in"]
        # Resolve monash_data_dir relative to dino_dir
        monash_dir = dino_cfg.get('monash_data_dir', '../Monash')
        if not os.path.isabs(monash_dir):
            dino_cfg['monash_data_dir'] = str((dino_dir / monash_dir).resolve())
        print("\n" + "="*60)
        print(f"  MODEL: DINO  (TSDINOALT 4)")
        print(f"  pretrain: Monash ({dino_cfg['monash_data_dir']})   forecast: {forecast_dataset}")
        print("="*60)
    else:
        pretrain_dataset = pretrain_dataset or dino_cfg.get("pretrain_dataset")
        forecast_dataset = forecast_dataset or pretrain_dataset
        if pretrain_dataset is None:
            raise ValueError("pretrain_dataset not set — specify via run() or config.py")
        ds_pre  = get_dataset_info(pretrain_dataset)
        ds_fore = get_dataset_info(forecast_dataset)
        dino_cfg["data_path"] = ds_pre["csv_path"]
        dino_cfg["c_in"]      = ds_pre["c_in"]
        print("\n" + "="*60)
        print(f"  MODEL: DINO  (TSDINOALT 4)")
        print(f"  pretrain: {pretrain_dataset}   forecast: {forecast_dataset}")
        print("="*60)

    dino_cfg["data_path_forecast_training"]    = ds_fore["csv_path"]
    dino_cfg["data_path_forecast_test"]        = ds_fore["csv_path"]
    dino_cfg["parms_for_training_forecasting"] = ds_fore["columns"]
    dino_cfg["parms_for_testing_forecasting"]  = ds_fore["columns"]

    args = _config_to_dino_args(dino_cfg)

    # Resolve data paths relative to dino_dir so they work from any CWD
    # (skip if already absolute — e.g. injected from dataset_registry)
    for attr in ('data_path', 'data_path_forecast_training',
                 'data_path_forecast_test', 'data_path_classification'):
        val = getattr(args, attr, '')
        if val and not os.path.isabs(val):
            setattr(args, attr, str(dino_dir / val))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── pretraining ──────────────────────────────────────────────────────────
    if not skip_train:
        print("\n[DINO] Starting pretraining …")
        dino_main.train_TS_DINO(args)
    else:
        print("[DINO] Skipping pretraining.")

    # ── forecasting downstream ────────────────────────────────────────────────
    print("\n[DINO] Running forecasting downstream task …")
    ckpt_freq = dino_cfg.get("saveckp_freq", 10)
    n_epochs   = dino_cfg.get("epochs", 100)
    checkpoints = [0] + list(range(ckpt_freq, n_epochs + 1, ckpt_freq))
    for ckpt in checkpoints:
        args.path_num = ckpt
        print(f"\n  → checkpoint {ckpt} ({'random init' if ckpt == 0 else f'epoch {ckpt}'})")
        dino_main.test_run(args)


# ── Discrete JEPA ─────────────────────────────────────────────────────────────

def _resolve_jepa_path(p: str, jepa_dir: Path) -> str:
    """Return *p* as-is if absolute, otherwise resolve relative to *jepa_dir*."""
    if os.path.isabs(p):
        return p
    return str((jepa_dir / p.lstrip('./').lstrip('/')).resolve())


def run_jepa(skip_train: bool = False,
             pretrain_dataset: str = None,
             forecast_dataset: str = None):
    jepa_dir = Path(__file__).parent / "Discrete_JEPA"
    _add_path(jepa_dir)

    import torch
    from config_files.config_pretrain import config
    from data_loaders.data_puller import (DataPullerDJepa, ForcastingDataPullerDescrete,
                                          MonashDataPullerJEPA)
    from Discrete_JEPA.Discrete_Jepa import DiscreteJEPA

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = dict(config)
    pretrain_on_monash = config.get('pretrain_on_monash', False)

    # Resolve forecast dataset (always needed for downstream)
    forecast_dataset = forecast_dataset or config.get("forecast_dataset")
    if pretrain_on_monash:
        if forecast_dataset is None:
            raise ValueError("forecast_dataset must be set when pretrain_on_monash=True")
        ds_fore = get_dataset_info(forecast_dataset)
        # Resolve monash_data_dir relative to jepa_dir
        monash_dir = config.get('monash_data_dir', '../Monash')
        if not os.path.isabs(monash_dir):
            config['monash_data_dir'] = str((jepa_dir / monash_dir).resolve())
        print("\n" + "="*60)
        print(f"  MODEL: Discrete JEPA")
        print(f"  pretrain: Monash ({config['monash_data_dir']})   forecast: {forecast_dataset}")
        print("="*60)
    else:
        pretrain_dataset = pretrain_dataset or config.get("pretrain_dataset")
        forecast_dataset = forecast_dataset or pretrain_dataset
        if pretrain_dataset is None:
            raise ValueError("pretrain_dataset not set — specify via run() or config_pretrain.py")
        ds_pre  = get_dataset_info(pretrain_dataset)
        ds_fore = get_dataset_info(forecast_dataset)
        n_groups = len(ds_pre["jepa_groups"])
        config["path_data"]       = [_resolve_jepa_path(ds_pre["csv_path"], jepa_dir)] * n_groups
        config["timestampcols"]   = [ds_pre["timestamp_col"]] * n_groups
        config["input_variables"] = ds_pre["jepa_groups"]
        print("\n" + "="*60)
        print(f"  MODEL: Discrete JEPA")
        print(f"  pretrain: {pretrain_dataset}   forecast: {forecast_dataset}")
        print("="*60)

    # Always set forecasting paths from forecast dataset
    config["path_data_forcasting"]       = [_resolve_jepa_path(ds_fore["csv_path"], jepa_dir)]
    config["timestampcols_forcasting"]   = [ds_fore["timestamp_col"]]
    config["input_variables_forcasting"] = [ds_fore["columns"]]

    # ── data ─────────────────────────────────────────────────────────────────
    print("\n[JEPA] Loading datasets …")
    if pretrain_on_monash:
        train_dataset = MonashDataPullerJEPA(config, which='train')
        val_dataset   = MonashDataPullerJEPA(config, which='val')
        test_dataset  = MonashDataPullerJEPA(config, which='test')
    else:
        train_dataset = DataPullerDJepa(
            data_paths         = config["path_data"],
            patch_size         = config["patch_size"],
            batch_size         = config["batch_size"],
            ratio_patches      = config["ratio_patches"],
            mask_ratio         = config["mask_ratio"],
            masking_type       = config["masking_type"],
            num_semantic_tokens= config["num_semantic_tokens"],
            input_variables    = config["input_variables"],
            timestamp_cols     = config["timestampcols"],
            type_data          = "train",
            val_prec           = config["val_prec"],
            test_prec          = config["test_prec"],
            stride             = config.get("stride", None),
            num_blocks         = config.get("num_blocks", 1),
        )
        val_dataset  = copy.copy(train_dataset); val_dataset.which  = "val"
        test_dataset = copy.copy(train_dataset); test_dataset.which = "test"

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=config["batch_size"], shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=config["batch_size"], shuffle=False)
    input_dim    = len(train_loader.dataset[0][0][0])

    forecasting_data = ForcastingDataPullerDescrete(config)
    val_fc   = copy.copy(forecasting_data); val_fc.which  = "val";  val_fc.rebuild()
    test_fc  = copy.copy(forecasting_data); test_fc.which = "test"; test_fc.rebuild()
    train_loader_fc = torch.utils.data.DataLoader(forecasting_data, batch_size=config["batch_size"], shuffle=True)
    val_loader_fc   = torch.utils.data.DataLoader(val_fc,           batch_size=config["batch_size"], shuffle=True)
    test_loader_fc  = torch.utils.data.DataLoader(test_fc,          batch_size=config["batch_size"], shuffle=False)

    # ── model ─────────────────────────────────────────────────────────────────
    model = DiscreteJEPA(
        config            = config,
        input_dim         = input_dim,
        num_patches       = len(train_loader.dataset[0][0]),
        steps_per_epoch   = len(train_loader),
        train_loader      = train_loader,
        val_loader        = val_loader,
        test_loader       = test_loader,
        forcasting_train  = train_loader_fc,
        forcasting_val    = val_loader_fc,
        forcasting_test   = test_loader_fc,
    )

    # ── pretraining ───────────────────────────────────────────────────────────
    if not skip_train:
        print("\n[JEPA] Starting pretraining …")
        model.train_and_evaluate()
    else:
        print("[JEPA] Skipping pretraining.")

    # ── forecasting downstream ────────────────────────────────────────────────
    modes = config.get("forecasting_modes", ["zeroshot"])
    print(f"\n[JEPA] Running forecasting downstream task … modes={modes}")
    _MODE_MAP = {
        "zeroshot":   "forcasting_zeroshot",
        "finetuning": "finetuning_forecasting",
        "predictor":  "predictor_forecasting",
    }
    for epoch in range(200, 2301, 100):
        print(f"\n  → checkpoint epoch {epoch}")
        for mode in modes:
            method_name = _MODE_MAP.get(mode)
            if method_name is None:
                print(f"  [JEPA] Unknown forecasting mode '{mode}', skipping.")
                continue
            getattr(model, method_name)(f"_epoch{epoch}")


# ── PatchTST ──────────────────────────────────────────────────────────────────

def run_patchtst(skip_train: bool = False, pretrain_dataset: str = None, forecast_dataset: str = None):
    print("\n" + "="*60)
    print("  MODEL: PatchTST (self-supervised)")
    print("="*60)

    patchtst_dir = str((Path(__file__).parent / "PatchTST_self_supervised").resolve())

    # ── pretraining ───────────────────────────────────────────────────────────
    if not skip_train:
        _pretrain_dset = pretrain_dataset or "ettm1"
        print(f"\n[PatchTST] Starting pretraining on {_pretrain_dset} …")
        result = subprocess.run(
            [sys.executable, "patchtst_pretrain.py",
             "--dset_pretrain", _pretrain_dset,
             "--n_epochs_pretrain", "10",
             "--d_ff", "512"],
            cwd=patchtst_dir,
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print("[PatchTST] Pretraining exited with errors.")
            print(result.stderr)
            return
    else:
        _pretrain_dset = pretrain_dataset or "ettm1"
        print("[PatchTST] Skipping pretraining.")

    # ── forecasting downstream ────────────────────────────────────────────────
    _forecast_dset = forecast_dataset or _pretrain_dset
    # Reconstruct the pretrained model path using the same naming convention as patchtst_pretrain.py
    pretrained_model_path = os.path.join(
        patchtst_dir,
        "saved_models", _pretrain_dset, "masked_patchtst", "based_model",
        "patchtst_pretrained_cw512_patch12_stride12_epochs-pretrain10_mask0.4_model1.pth"
    )
    print(f"\n[PatchTST] Running forecasting fine-tuning on {_forecast_dset} …")
    result = subprocess.run(
        [sys.executable, "patchtst_finetune.py",
         "--dset_finetune", _forecast_dset,
         "--is_finetune", "1",
         "--d_ff", "512",
         "--pretrained_model", pretrained_model_path],
        cwd=patchtst_dir,
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("[PatchTST] Forecasting fine-tuning exited with errors.")
        print(result.stderr)


# ── entry point ───────────────────────────────────────────────────────────────

RUNNERS = {
    "dino":     run_dino,
    "jepa":     run_jepa,
    "patchtst": run_patchtst,
}

def run(model: str, skip_train: bool = False,
        pretrain_dataset: str = None,
        forecast_dataset: str = None):
    """
    Call this directly from a notebook:
        from Train_and_downstream import run
        run(model="dino", pretrain_dataset="ettm1", skip_train=False)
        run(model="dino", pretrain_dataset="ettm1", forecast_dataset="etth1", skip_train=False)

    forecast_dataset defaults to pretrain_dataset when not set.
    Available datasets: ettm1, etth1, etth2, ettm2, weather, electricity, traffic
    """
    model = model.lower()
    if model not in RUNNERS:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(RUNNERS)}")
    RUNNERS[model](skip_train=skip_train,
                   pretrain_dataset=pretrain_dataset,
                   forecast_dataset=forecast_dataset)


if __name__ == "__main__":
    from dataset_registry import DATASETS as _DATASETS
    parser = argparse.ArgumentParser(description="Unified training + forecasting runner")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(RUNNERS),
        help="Which model to run: dino | jepa | patchtst",
    )
    parser.add_argument(
        "--pretrain_dataset", type=str, required=True,
        choices=list(_DATASETS),
        help=f"Dataset for pretraining. Available: {list(_DATASETS)}",
    )
    parser.add_argument(
        "--forecast_dataset", type=str, default=None,
        choices=list(_DATASETS),
        help="Dataset for forecasting downstream (defaults to pretrain_dataset).",
    )
    parser.add_argument(
        "--skip_train", type=str, default="false",
        choices=["true", "false"],
        help="Skip pretraining and go straight to forecasting (true | false)",
    )
    args = parser.parse_args()
    run(model=args.model,
        skip_train=args.skip_train.lower() == "true",
        pretrain_dataset=args.pretrain_dataset,
        forecast_dataset=args.forecast_dataset)
