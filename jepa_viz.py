"""
JEPA (P2P simple) Encoder Embedding Visualiser
================================================
Run from the allthree/ directory:

    python jepa_viz.py --ckpt models/JEPA_epoch_7.pt

Outputs PNGs to viz_output/jepa/
"""

import argparse
import os
import sys
from pathlib import Path

# Fix macOS segfault: PyTorch + sklearn OpenMP conflict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'JEPA'))
sys.path.insert(0, str(ROOT / 'Discrete_JEPA'))

OUT_DIR   = ROOT / 'viz_output' / 'jepa'
DATASET   = 'ettm1'
N_BATCHES = 10
BATCH_SZ  = 16


def load_config():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'config_jepa', ROOT / 'JEPA' / 'config_files' / 'config_jepa.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


def build_encoder(config, device):
    from JEPA.Encoder import Encoder
    return Encoder(
        num_patches    = config['ratio_patches'],
        dim_in         = config['patch_size'],
        embed_dim      = config['encoder_embed_dim'],
        nhead          = config['nhead'],
        num_layers     = config['num_encoder_layers'],
        mlp_ratio      = config['mlp_ratio'],
        drop_rate      = 0.0,
        attn_drop_rate = 0.0,
        pe             = 'sincos',
        learn_pe       = False,
        res_attention  = True,
    ).to(device)


def build_predictor(config, device):
    from JEPA.Predictors import JEPAPredictor
    return JEPAPredictor(
        num_patches = config['ratio_patches'],
        embed_dim   = config['encoder_embed_dim'],
        nhead       = config.get('predictor_nhead', 4),
        num_layers  = config.get('predictor_num_layers', 2),
        config      = config,
    ).to(device)


def make_loader(config):
    from torch.utils.data import DataLoader
    from data_loaders.data_puller import DataPullerDJepa

    data_paths = [str((ROOT / 'Discrete_JEPA' / p).resolve()) for p in config['path_data']]
    ds = DataPullerDJepa(
        data_paths          = data_paths,
        patch_size          = config['patch_size'],
        batch_size          = BATCH_SZ,
        ratio_patches       = config['ratio_patches'],
        mask_ratio          = config['mask_ratio'],
        masking_type        = config['masking_type'],
        num_semantic_tokens = 0,
        input_variables     = [config['input_variables']],
        timestamp_cols      = config['timestampcols'],
        type_data           = 'train',
        val_prec            = config['val_prec'],
        test_prec           = config['test_prec'],
        num_blocks          = config['num_blocks'],
    )
    return DataLoader(ds, batch_size=BATCH_SZ, shuffle=True, drop_last=True)


def collect_embeddings(encoder, ema_encoder, config, device, loader):
    """Collect full-sequence student/EMA embeddings for PCA, t-SNE, norm plots."""
    all_s, all_e, all_var = [], [], []
    encoder.eval(); ema_encoder.eval()

    with torch.no_grad():
        for i, (patches, *_) in enumerate(loader):
            if i >= N_BATCHES:
                break
            patches = patches.to(device)
            B, P, P_L, F = patches.shape
            all_s.append(encoder(patches)['data_patches'].cpu())
            all_e.append(ema_encoder(patches)['data_patches'].cpu())
            all_var.append(torch.arange(F).repeat(B))
            print(f'  batch {i+1}/{N_BATCHES}', end='\r')

    print()
    student = torch.cat(all_s, 0).numpy()
    ema     = torch.cat(all_e, 0).numpy()
    var_idx = torch.cat(all_var, 0).numpy()
    print(f'Patch embeddings: {student.shape}  (N×P×D)')
    return student, ema, var_idx


def collect_predictor_alignment(encoder, ema_encoder, predictor, config, device, loader):
    """
    Replicate the actual JEPA forward pass to measure real predictive alignment:
      - student encoder sees only context patches (masked input)
      - predictor predicts target patch embeddings
      - EMA encoder sees all patches; we keep only target positions
      - measure cosine_sim(predictor_output, ema_target) per absolute patch position
    Returns: cos_sims_by_pos  list of length num_patches, each entry = list of scalar sims
    """
    from mask_util import apply_mask

    num_patches = config['ratio_patches']
    cos_by_pos = [[] for _ in range(num_patches)]   # indexed by absolute patch position

    encoder.eval(); ema_encoder.eval(); predictor.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= N_BATCHES:
                break
            patches, masks, non_masks = batch[0], batch[1], batch[2]
            patches   = patches.to(device)
            masks     = masks.to(device)      # context indices [B, N_ctx]
            non_masks = non_masks.to(device)  # target indices  [B, N_tgt]

            # EMA encoder on full sequence → keep only target positions
            target_out = ema_encoder(patches)
            target_emb = apply_mask(target_out['data_patches'], non_masks)  # [B*F, N_tgt, D]

            # Student encoder on context patches only
            context_out = encoder(patches, mask=masks)
            context_emb = context_out['data_patches']                        # [B*F, N_ctx, D]

            # Predictor: predict target positions from context
            pred = predictor(context_emb, target_mask=non_masks)            # [B*F, N_tgt, D]

            # Cosine similarity per target patch
            pred_n   = F.normalize(pred,       dim=-1)  # [B*F, N_tgt, D]
            target_n = F.normalize(target_emb, dim=-1)  # [B*F, N_tgt, D]
            cos      = (pred_n * target_n).sum(-1)       # [B*F, N_tgt]

            # Accumulate by absolute patch position
            B, N_tgt = non_masks.shape
            BF = cos.shape[0]
            F_  = BF // B
            # non_masks is [B, N_tgt]; expand to [B*F, N_tgt] to match cos
            nm_exp = non_masks.unsqueeze(1).expand(-1, F_, -1).reshape(BF, N_tgt)
            for t in range(N_tgt):
                pos_ids  = nm_exp[:, t].cpu().tolist()   # absolute patch index for each sample
                sim_vals = cos[:, t].cpu().tolist()
                for pos, sim in zip(pos_ids, sim_vals):
                    cos_by_pos[pos].append(sim)

            print(f'  alignment batch {i+1}/{N_BATCHES}', end='\r')

    print()
    return cos_by_pos


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_pca(emb, out_dir):
    from sklearn.decomposition import PCA
    N, P, D = emb.shape
    flat = emb.reshape(-1, D)
    pca = PCA(n_components=min(50, D))
    pca.fit(flat)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(cumvar, marker='o', ms=3)
    axes[0].axhline(90, color='red', linestyle='--', label='90%')
    axes[0].set_xlabel('Component'); axes[0].set_ylabel('Cumulative variance (%)')
    axes[0].set_title('PCA — Cumulative Variance'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].bar(range(1, 21), pca.explained_variance_ratio_[:20] * 100)
    axes[1].set_xlabel('Component'); axes[1].set_ylabel('Variance (%)')
    axes[1].set_title('Top 20 PCA Components'); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'pca_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    dims90 = int(np.searchsorted(cumvar / 100, 0.90)) + 1
    print(f'[PCA] dims for 90%: {dims90}/{D}  top-1: {pca.explained_variance_ratio_[0]*100:.1f}%')


def plot_tsne(emb, var_idx, config, out_dir):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    N, P, D = emb.shape
    n_s = min(N, 500 // P)
    idx = np.random.choice(N, size=n_s, replace=False)
    flat = emb[idx].reshape(-1, D)
    pos_lbl = np.tile(np.arange(P), n_s)
    var_lbl = np.repeat(var_idx[idx], P)

    flat_pca = PCA(n_components=min(30, D)).fit_transform(flat)
    n_pts = flat_pca.shape[0]
    perp = min(30, n_pts // 4)
    print(f'[t-SNE] fitting {n_pts} points (perplexity={perp})…')
    emb2d = TSNE(n_components=2, perplexity=perp, n_iter=1000, random_state=42).fit_transform(flat_pca)

    # colour by patch position
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=pos_lbl, cmap='plasma', s=4, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Patch position (temporal)')
    ax.set_title('t-SNE — colour = temporal patch index')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_dir / 'tsne_position.png', dpi=150, bbox_inches='tight')
    plt.close()

    # colour by variable
    n_vars = int(var_lbl.max()) + 1
    cmap_v = matplotlib.colormaps.get_cmap('tab10').resampled(n_vars)
    var_names = config.get('input_variables', [f'v{i}' for i in range(n_vars)])
    fig, ax = plt.subplots(figsize=(8, 7))
    for v in range(n_vars):
        m = var_lbl == v
        ax.scatter(emb2d[m, 0], emb2d[m, 1], c=[cmap_v(v)], s=4, alpha=0.5,
                   label=var_names[v] if v < len(var_names) else f'v{v}')
    ax.legend(markerscale=3, fontsize=8)
    ax.set_title('t-SNE — colour = variable / channel')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_dir / 'tsne_variable.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[t-SNE] done.')


def plot_alignment(cos_by_pos, out_dir):
    """
    cos_by_pos: list[list[float]], indexed by absolute patch position.
    Plots mean ± std of predictor→EMA cosine similarity per patch position.
    Only positions that appeared as targets have data.
    """
    positions = [p for p, v in enumerate(cos_by_pos) if len(v) > 0]
    means     = np.array([np.mean(cos_by_pos[p]) for p in positions])
    stds      = np.array([np.std(cos_by_pos[p])  for p in positions])
    overall   = np.concatenate([cos_by_pos[p] for p in positions])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: mean ± std per patch position
    ax = axes[0]
    ax.plot(positions, means, marker='o', ms=4, label='mean cos sim')
    ax.fill_between(positions, means - stds, means + stds, alpha=0.2, label='±1 std')
    ax.axhline(overall.mean(), color='red', linestyle='--', label=f'global mean={overall.mean():.3f}')
    ax.set_xlabel('Patch position (absolute)')
    ax.set_ylabel('Cosine sim (predictor vs EMA target)')
    ax.set_title('Predictor → EMA alignment per patch position')
    ax.set_ylim(-0.1, 1.05); ax.legend(); ax.grid(True, alpha=0.3)

    # Right: histogram of all cosine similarities
    ax2 = axes[1]
    ax2.hist(overall, bins=50, edgecolor='black', linewidth=0.3)
    ax2.axvline(overall.mean(), color='red', linestyle='--', label=f'mean={overall.mean():.3f}')
    ax2.set_xlabel('Cosine similarity'); ax2.set_ylabel('Count')
    ax2.set_title('Distribution of predictor→EMA cosine similarities')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'student_ema_alignment.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Alignment] global mean cos sim (predictor vs EMA): {overall.mean():.4f}  std: {overall.std():.4f}')


def plot_norms(student, ema, out_dir):
    P  = student.shape[1]
    ns = np.linalg.norm(student, axis=-1)
    ne = np.linalg.norm(ema,     axis=-1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ns.mean(0), label='Student',    marker='o', ms=4)
    ax.plot(ne.mean(0), label='EMA target', marker='s', ms=4, linestyle='--')
    ax.fill_between(range(P), ns.mean(0) - ns.std(0), ns.mean(0) + ns.std(0), alpha=0.2)
    ax.set_xlabel('Patch position'); ax.set_ylabel('L2 norm')
    ax.set_title('Embedding norm per patch position')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'embedding_norms.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_sim_heatmap(emb, out_dir):
    mean_emb = emb.mean(0)
    norm = mean_emb / (np.linalg.norm(mean_emb, axis=-1, keepdims=True) + 1e-8)
    sim  = norm @ norm.T
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Patch position'); ax.set_ylabel('Patch position')
    ax.set_title('Cosine similarity between mean patch embeddings')
    plt.tight_layout()
    plt.savefig(out_dir / 'patch_similarity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[Heatmap] saved patch_similarity.png')


# ─────────────────────────────────────────────────────────────────────────────

def main(ckpt_path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    config      = load_config()
    encoder     = build_encoder(config, device)
    ema_encoder = build_encoder(config, device)
    predictor   = build_predictor(config, device)

    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt['encoder'])
    ema_key = 'target_encoder' if 'target_encoder' in ckpt else 'encoder_ema'
    ema_encoder.load_state_dict(ckpt[ema_key])
    predictor.load_state_dict(ckpt['predictor'])
    print(f'Loaded: {ckpt_path}')

    loader = make_loader(config)

    print('Collecting full-sequence embeddings (PCA / t-SNE / norms)…')
    student, ema, var_idx = collect_embeddings(encoder, ema_encoder, config, device, loader)

    print('Collecting predictor alignment (masked forward pass)…')
    cos_by_pos = collect_predictor_alignment(encoder, ema_encoder, predictor, config, device, loader)

    print('Generating plots…')
    plot_pca(student, OUT_DIR)
    plot_tsne(student, var_idx, config, OUT_DIR)
    plot_alignment(cos_by_pos, OUT_DIR)
    plot_norms(student, ema, OUT_DIR)
    plot_sim_heatmap(student, OUT_DIR)
    print(f'\nAll plots saved to: {OUT_DIR}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()
    main(Path(args.ckpt))
