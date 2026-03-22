


import glob
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
import sys
import os as _os

# Root directory containing ETT CSVs — resolved relative to this file so it works
# from any working directory (Colab, local, etc.)
_ETT_DATA_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'Discrete_JEPA', 'data')
) + _os.sep

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *


# ── Monash TSF reader & Dataset ───────────────────────────────────────────────

def _read_tsf_series(path):
    """Read a Monash .tsf file. Returns a list of 1-D numpy float32 arrays."""
    found_data = False
    series_list = []
    with open(path, 'r', encoding='cp1252') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('@data'):
                found_data = True
            elif not line.startswith('@') and found_data:
                vals_str = line.split(':')[-1].split(',')
                vals = []
                for v in vals_str:
                    v = v.strip()
                    vals.append(np.nan if v == '?' else float(v))
                if vals:
                    series_list.append(np.array(vals, dtype=np.float32))
    return series_list


class Dataset_Monash(Dataset):
    """
    Monash pretraining dataset for PatchTST.

    Loads all .tsf files from *monash_data_dir*, filters by *min_len*, and
    builds non-overlapping windows of length *context_points*.  Each item
    returns (x, x) — identical input and target — so PatchMaskCB can create
    masked patches from xb and reconstruct them.

    Splits are per-series: train 80%, val 10%, test 10%.
    """

    def __init__(self, monash_data_dir, context_points, min_len=512,
                 val_ratio=0.1, test_ratio=0.1, split='train'):
        self.context_points = context_points
        self._windows = []
        for fpath in sorted(glob.glob(_os.path.join(monash_data_dir, '*.tsf'))):
            for series in _read_tsf_series(fpath):
                if len(series) < min_len or np.isnan(series).any():
                    continue
                T = len(series)
                train_end = int(T * (1 - val_ratio - test_ratio))
                val_end   = int(T * (1 - test_ratio))
                if split == 'train':
                    seg = series[:train_end]
                elif split == 'val':
                    seg = series[train_end:val_end]
                else:
                    seg = series[val_end:]
                if len(seg) < context_points:
                    continue
                # Standardize with train-split statistics
                mu  = float(series[:train_end].mean())
                sig = float(series[:train_end].std()) + 1e-8
                seg = (seg - mu) / sig
                # Non-overlapping windows
                n_windows = len(seg) // context_points
                for i in range(n_windows):
                    self._windows.append(seg[i * context_points:(i + 1) * context_points].copy())
        print(f"[Dataset_Monash] split={split}  windows={len(self._windows)}")

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, idx):
        x = torch.tensor(self._windows[idx], dtype=torch.float32).unsqueeze(-1)  # [T, 1]
        return x, x


DSETS = ['ettm1', 'ettm2', 'etth1', 'etth2', 'electricity',
         'traffic', 'illness', 'weather', 'exchange', 'monash',
        ]

def get_dls(params):

    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params, 'use_time_features'): params.use_time_features = False

    if params.dset == 'monash':
        monash_dir = getattr(params, 'monash_data_dir', '../Monash')
        if not _os.path.isabs(monash_dir):
            monash_dir = _os.path.normpath(
                _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), monash_dir))
        dls = DataLoaders(
            datasetCls=Dataset_Monash,
            dataset_kwargs={
                'monash_data_dir': monash_dir,
                'context_points':  params.context_points,
                'min_len':         getattr(params, 'monash_min_len', 512),
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )
        dls.vars = 1
        dls.len  = params.context_points
        dls.c    = params.context_points
        return dls

    if params.dset == 'ettm1':
        root_path = _ETT_DATA_DIR
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_minute,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )


    elif params.dset == 'ettm2':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_minute,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'etth1':
        root_path = _ETT_DATA_DIR
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_hour,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )


    elif params.dset == 'etth2':
        root_path = _ETT_DATA_DIR
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_hour,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )


    elif params.dset == 'electricity':
        root_path = _ETT_DATA_DIR
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'electricity.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'traffic':
        root_path = _ETT_DATA_DIR
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'traffic.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'weather':
        root_path = _ETT_DATA_DIR
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'weather.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'illness':
        root_path = '/data/datasets/public/illness/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'national_illness.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'exchange':
        root_path = '/data/datasets/public/exchange_rate/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'exchange_rate.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls



if __name__ == "__main__":
    class Params:
        dset= 'etth2'
        context_points= 384
        target_points= 96
        batch_size= 64
        num_workers= 8
        with_ray= False
        features='M'
    params = Params
    dls = get_dls(params)
    for i, batch in enumerate(dls.valid):
        print(i, len(batch), batch[0].shape, batch[1].shape)
    breakpoint()
