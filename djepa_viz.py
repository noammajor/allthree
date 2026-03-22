"""
Discrete JEPA Encoder + VQ Codebook Embedding Visualiser
==========================================================
Run from the allthree/ directory:

    python djepa_viz.py --ckpt Discrete_JEPA/output_model/DiscreteJEPA/epoch10.pth

Outputs PNGs to viz_output/djepa/

Plots:
  Encoder (patch embeddings)
    1. PCA variance explained
    2. t-SNE coloured by patch position
    3. t-SNE coloured by variable
    4. P2P predictor → EMA alignment per target patch position
    5. S2P predictor → EMA alignment per target patch position
    6. Embedding norm per patch position
    7. Inter-patch cosine similarity heatmap

  Semantic tokens
    8.  t-SNE of semantic token embeddings
    9.  P2S predictor → EMA alignment per semantic token index
    10. Embedding norm per semantic token
    11. Inter-token cosine similarity heatmap

  VQ Codebook
    12. Code usage histogram — which codes fire and how often
    13. t-SNE of all codebook vectors (are codes spread out?)
    14. t-SNE of semantic token embeddings coloured by assigned code
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
sys.path.insert(0, str(ROOT / 'Discrete_JEPA'))

OUT_DIR   = ROOT / 'viz_output' / 'djepa'
DATASET   = 'ettm1'
N_BATCHES = 20
BATCH_SZ  = 32


def load_config():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'config_pretrain',
        ROOT / 'Discrete_JEPA' / 'config_files' / 'config_pretrain.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


def infer_arch(ckpt):
    """Read encoder architecture from checkpoint tensors, not from config."""
    enc = ckpt['encoder']
    ratio_patches       = enc['W_pos'].shape[0]
    embed_dim           = enc['W_pos'].shape[1]
    patch_size          = enc['W_P.weight'].shape[1]
    num_semantic_tokens = enc['semantic_tokens'].shape[1]
    num_layers          = max(int(k.split('.')[2])
                              for k in enc if k.startswith('transformer.layers.')) + 1
    vq_size             = ckpt['vector_quantizer']['_embedding.weight'].shape[0]
    return dict(ratio_patches=ratio_patches, embed_dim=embed_dim,
                patch_size=patch_size, num_semantic_tokens=num_semantic_tokens,
                num_layers=num_layers, vq_size=vq_size)


def build_encoder(arch, config, device):
    from Discrete_JEPA.Encoder import Encoder
    return Encoder(
        num_patches         = arch['ratio_patches'],
        dim_in              = arch['patch_size'],
        embed_dim           = arch['embed_dim'],
        nhead               = config['nhead'],
        num_layers          = arch['num_layers'],
        mlp_ratio           = config['mlp_ratio'],
        drop_rate           = 0.0,
        attn_drop_rate      = 0.0,
        pe                  = 'sincos',
        learn_pe            = False,
        res_attention       = True,
        num_semantic_tokens = arch['num_semantic_tokens'],
    ).to(device)


def build_vq(arch, config, device):
    from Discrete_JEPA.VQ import VectorQuantizer
    return VectorQuantizer(
        num_embeddings  = arch['vq_size'],
        embedding_dim   = arch['embed_dim'],
        commitment_cost = config['commitment_cost'],
    ).to(device)


def build_predictor(arch, config, device):
    from Discrete_JEPA.Predictors import DiscreteJEPAPredictor
    return DiscreteJEPAPredictor(
        num_semantic_tokens = arch['num_semantic_tokens'],
        embed_dim           = arch['embed_dim'],
        config              = config,
    ).to(device)


def make_loader(arch, config):
    from torch.utils.data import DataLoader
    from data_loaders.data_puller import DataPullerDJepa

    data_paths = [str((ROOT / 'Discrete_JEPA' / p).resolve()) for p in config['path_data']]
    ds = DataPullerDJepa(
        data_paths          = data_paths,
        patch_size          = arch['patch_size'],
        batch_size          = BATCH_SZ,
        ratio_patches       = arch['ratio_patches'],
        mask_ratio          = config['mask_ratio'],
        masking_type        = config['masking_type'],
        num_semantic_tokens = arch['num_semantic_tokens'],
        input_variables     = [config['input_variables']],
        timestamp_cols      = config['timestampcols'],
        type_data           = 'train',
        val_prec            = config['val_prec'],
        test_prec           = config['test_prec'],
        num_blocks          = config['num_blocks'],
    )
    return DataLoader(ds, batch_size=BATCH_SZ, shuffle=True, drop_last=True)


def collect_embeddings(encoder, ema_encoder, vq, config, arch, device, loader):
    all_patch_s, all_patch_e  = [], []
    all_sem_s,   all_sem_e    = [], []
    all_codes, all_var        = [], []

    encoder.eval(); ema_encoder.eval(); vq.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= N_BATCHES:
                break
            patches = batch[0].to(device)
            B, P, P_L, F = patches.shape

            out_s = encoder(patches)
            out_e = ema_encoder(patches)

            all_patch_s.append(out_s['data_patches'].cpu())
            all_patch_e.append(out_e['data_patches'].cpu())
            all_sem_s.append(out_s['quantized_semantic'].cpu())
            all_sem_e.append(out_e['quantized_semantic'].cpu())

            # quantize to get code assignments
            # VQ flattens [B*F, S, D] → indices [B*F*S, 1]; reshape to [B*F, S]
            sem = out_s['quantized_semantic']
            _, _, _, indices, _, _ = vq(sem)
            all_codes.append(indices.reshape(sem.shape[0], sem.shape[1]).cpu())

            all_var.append(torch.arange(F).repeat(B))
            print(f'  batch {i+1}/{N_BATCHES}', end='\r')

    print()
    patch_s = torch.cat(all_patch_s, 0).numpy()
    patch_e = torch.cat(all_patch_e, 0).numpy()
    sem_s   = torch.cat(all_sem_s,   0).numpy()
    sem_e   = torch.cat(all_sem_e,   0).numpy()
    codes   = torch.cat(all_codes,   0).numpy()   # [N, S]
    var_idx = torch.cat(all_var,     0).numpy()
    print(f'Patch embeddings : {patch_s.shape}')
    print(f'Semantic tokens  : {sem_s.shape}')
    print(f'Code assignments : {codes.shape}')
    return patch_s, patch_e, sem_s, sem_e, codes, var_idx


def collect_predictor_alignment(encoder, ema_encoder, predictor, vq, arch, config, device, loader):
    """
    Replicates the actual DJEPA forward pass to measure real predictive alignment.

    For each task:
      P2P: context patches → predictor → predicted target patches vs EMA target patches
      S2P: semantic tokens → predictor → predicted target patches vs EMA target patches
      P2S: context patches → predictor → predicted semantic tokens vs EMA semantic tokens

    P2P and S2P: cosine sims accumulated per absolute patch position [0, num_patches).
    P2S: cosine sims accumulated per semantic token index [0, num_semantic_tokens).
    """
    from mask_util import apply_mask

    num_patches   = arch['ratio_patches']
    num_sem       = arch['num_semantic_tokens']
    cos_p2p = [[] for _ in range(num_patches)]
    cos_s2p = [[] for _ in range(num_patches)]
    cos_p2s = [[] for _ in range(num_sem)]

    encoder.eval(); ema_encoder.eval(); predictor.eval(); vq.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= N_BATCHES:
                break
            patches, masks, non_masks = batch[0], batch[1], batch[2]
            patches   = patches.to(device)
            masks     = masks.to(device)
            non_masks = non_masks.to(device)

            B, N_tgt = non_masks.shape

            # EMA encoder on full sequence
            target_out   = ema_encoder(patches)
            target_patch = apply_mask(target_out['data_patches'], non_masks)  # [B*F, N_tgt, D]
            target_sem   = target_out['quantized_semantic']                   # [B*F, S, D]
            # Quantize EMA semantic tokens (same VQ, no grad)
            _, target_sem, _, _, _, _ = vq(target_sem)

            # Student encoder on context patches only
            context_out = encoder(patches, mask=masks)
            ctx_patches = context_out['data_patches']          # [B*F, N_ctx, D]
            _, ctx_sem, _, _, _, _ = vq(context_out['quantized_semantic'])  # [B*F, S, D]

            BF = ctx_patches.shape[0]
            F_ = BF // B

            # ── P2P ───────────────────────────────────────────────────────────
            pred_p2p = predictor(ctx_patches, task='P2P', target_mask=non_masks)  # [B*F, N_tgt, D]
            p2p_n = F.normalize(pred_p2p,    dim=-1)
            tgt_n = F.normalize(target_patch, dim=-1)
            cos_p2p_vals = (p2p_n * tgt_n).sum(-1)  # [B*F, N_tgt]
            nm_exp = non_masks.unsqueeze(1).expand(-1, F_, -1).reshape(BF, N_tgt)
            for t in range(N_tgt):
                for pos, sim in zip(nm_exp[:, t].cpu().tolist(), cos_p2p_vals[:, t].cpu().tolist()):
                    cos_p2p[pos].append(sim)

            # ── S2P ───────────────────────────────────────────────────────────
            pred_s2p = predictor(ctx_sem, task='S2P', target_mask=non_masks)      # [B*F, N_tgt, D]
            s2p_n = F.normalize(pred_s2p,    dim=-1)
            cos_s2p_vals = (s2p_n * tgt_n).sum(-1)  # [B*F, N_tgt]
            for t in range(N_tgt):
                for pos, sim in zip(nm_exp[:, t].cpu().tolist(), cos_s2p_vals[:, t].cpu().tolist()):
                    cos_s2p[pos].append(sim)

            # ── P2S ───────────────────────────────────────────────────────────
            pred_p2s  = predictor(ctx_patches, task='P2S')                        # [B*F, S, D]
            p2s_n     = F.normalize(pred_p2s,   dim=-1)
            tsem_n    = F.normalize(target_sem,  dim=-1)
            cos_p2s_vals = (p2s_n * tsem_n).sum(-1)  # [B*F, S]
            for s in range(num_sem):
                cos_p2s[s].extend(cos_p2s_vals[:, s].cpu().tolist())

            print(f'  alignment batch {i+1}/{N_BATCHES}', end='\r')

    print()
    return cos_p2p, cos_s2p, cos_p2s


# ── Shared plot helpers ───────────────────────────────────────────────────────

def plot_pca(emb, title, fname, out_dir):
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
    axes[0].set_title(f'PCA — {title}'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].bar(range(1, 21), pca.explained_variance_ratio_[:20] * 100)
    axes[1].set_xlabel('Component'); axes[1].set_ylabel('Variance (%)')
    axes[1].set_title(f'Top 20 PCA — {title}'); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()
    dims90 = int(np.searchsorted(cumvar / 100, 0.90)) + 1
    print(f'[PCA/{title}] dims for 90%: {dims90}  top-1: {pca.explained_variance_ratio_[0]*100:.1f}%')


def plot_tsne_positions(emb, var_idx, config, title_prefix, fname_prefix, out_dir):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    N, P, D = emb.shape
    n_s = min(N, 2000 // P)
    idx = np.random.choice(N, size=n_s, replace=False)
    flat    = emb[idx].reshape(-1, D)
    pos_lbl = np.tile(np.arange(P), n_s)
    var_lbl = np.repeat(var_idx[idx], P)

    flat_pca = PCA(n_components=min(50, D)).fit_transform(flat)
    print(f'[t-SNE/{title_prefix}] fitting…')
    emb2d = TSNE(n_components=2, perplexity=40, n_iter=1000,
                 random_state=42).fit_transform(flat_pca)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=pos_lbl, cmap='plasma', s=4, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Position index')
    ax.set_title(f't-SNE — {title_prefix} — colour = position')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_dir / f'{fname_prefix}_tsne_position.png', dpi=150, bbox_inches='tight')
    plt.close()

    n_vars = int(var_lbl.max()) + 1
    cmap_v = matplotlib.colormaps.get_cmap('tab10').resampled(n_vars)
    var_names = config.get('input_variables', [f'v{i}' for i in range(n_vars)])
    fig, ax = plt.subplots(figsize=(8, 7))
    for v in range(n_vars):
        m = var_lbl == v
        ax.scatter(emb2d[m, 0], emb2d[m, 1], c=[cmap_v(v)], s=4, alpha=0.5,
                   label=var_names[v] if v < len(var_names) else f'v{v}')
    ax.legend(markerscale=3, fontsize=8)
    ax.set_title(f't-SNE — {title_prefix} — colour = variable')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_dir / f'{fname_prefix}_tsne_variable.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[t-SNE/{title_prefix}] done.')
    return emb2d, idx   # return so VQ plot can reuse


def plot_predictor_alignment(cos_by_pos, title, fname, out_dir, x_label='Position'):
    """
    cos_by_pos: list[list[float]], indexed by position (patch index or semantic token index).
    Plots mean ± std of predictor→EMA cosine similarity per position + overall histogram.
    """
    positions = [p for p, v in enumerate(cos_by_pos) if len(v) > 0]
    means     = np.array([np.mean(cos_by_pos[p]) for p in positions])
    stds      = np.array([np.std(cos_by_pos[p])  for p in positions])
    overall   = np.concatenate([cos_by_pos[p] for p in positions])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    ax.plot(positions, means, marker='o', ms=4, label='mean cos sim')
    ax.fill_between(positions, means - stds, means + stds, alpha=0.2, label='±1 std')
    ax.axhline(overall.mean(), color='red', linestyle='--', label=f'global mean={overall.mean():.3f}')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Cosine sim (predictor vs EMA target)')
    ax.set_title(f'Predictor → EMA alignment — {title}')
    ax.set_ylim(-0.1, 1.05); ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(overall, bins=50, edgecolor='black', linewidth=0.3)
    ax2.axvline(overall.mean(), color='red', linestyle='--', label=f'mean={overall.mean():.3f}')
    ax2.set_xlabel('Cosine similarity'); ax2.set_ylabel('Count')
    ax2.set_title(f'Distribution — {title}')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Alignment/{title}] mean={overall.mean():.4f}  std={overall.std():.4f}')


def plot_norms(student, ema, title, fname, out_dir):
    P  = student.shape[1]
    ns = np.linalg.norm(student, axis=-1)
    ne = np.linalg.norm(ema,     axis=-1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ns.mean(0), label='Student',    marker='o', ms=4)
    ax.plot(ne.mean(0), label='EMA target', marker='s', ms=4, linestyle='--')
    ax.fill_between(range(P), ns.mean(0) - ns.std(0), ns.mean(0) + ns.std(0), alpha=0.2)
    ax.set_xlabel('Position'); ax.set_ylabel('L2 norm')
    ax.set_title(f'Embedding norms — {title}')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sim_heatmap(emb, title, fname, out_dir):
    mean_emb = emb.mean(0)
    norm = mean_emb / (np.linalg.norm(mean_emb, axis=-1, keepdims=True) + 1e-8)
    sim  = norm @ norm.T
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Position'); ax.set_ylabel('Position')
    ax.set_title(f'Mean embedding cosine similarity — {title}')
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()


# ── VQ-specific plots ─────────────────────────────────────────────────────────

def plot_code_usage(codes, codebook_size, out_dir):
    """Histogram of how often each code fires across all batches."""
    flat_codes = codes.flatten()   # [N * S]
    counts = np.bincount(flat_codes, minlength=codebook_size)
    active = (counts > 0).sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # full histogram
    axes[0].bar(range(codebook_size), counts, width=1.0)
    axes[0].set_xlabel('Code index'); axes[0].set_ylabel('Usage count')
    axes[0].set_title(f'Code usage histogram  ({active}/{codebook_size} active = {active/codebook_size*100:.1f}%)')

    # sorted usage (Zipf view)
    sorted_counts = np.sort(counts)[::-1]
    axes[1].plot(sorted_counts, marker='o', ms=2)
    axes[1].set_xlabel('Code rank'); axes[1].set_ylabel('Usage count (sorted)')
    axes[1].set_title('Code usage — sorted (Zipf view)')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'vq_code_usage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[VQ] active codes: {active}/{codebook_size}  ({active/codebook_size*100:.1f}%)')
    top5 = np.argsort(counts)[::-1][:5]
    print(f'[VQ] top-5 codes: {top5.tolist()} with counts {counts[top5].tolist()}')


def plot_codebook_tsne(codebook_vectors, codes, sem_s, out_dir):
    """
    Two plots:
    1. t-SNE of all codebook vectors — are they spread out?
    2. t-SNE of semantic token embeddings coloured by their assigned code.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    K, D = codebook_vectors.shape
    active_mask = np.zeros(K, dtype=bool)
    used_codes  = np.unique(codes.flatten())
    active_mask[used_codes] = True

    # ── Plot 1: codebook vectors ──
    print('[t-SNE/codebook] fitting codebook vectors…')
    cb_pca  = PCA(n_components=min(50, D)).fit_transform(codebook_vectors)
    cb_2d   = TSNE(n_components=2, perplexity=min(30, K-1), n_iter=1000,
                   random_state=42).fit_transform(cb_pca)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(cb_2d[~active_mask, 0], cb_2d[~active_mask, 1],
               c='lightgrey', s=15, alpha=0.5, label='Dead codes')
    ax.scatter(cb_2d[active_mask, 0],  cb_2d[active_mask, 1],
               c='red', s=30, label=f'Active ({active_mask.sum()})')
    ax.legend()
    ax.set_title(f'Codebook vectors — t-SNE  ({active_mask.sum()}/{K} active)')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_dir / 'vq_codebook_tsne.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot 2: semantic tokens coloured by assigned code ──
    N, S, D2 = sem_s.shape
    n_s = min(N, 2000 // S)
    idx = np.random.choice(N, size=n_s, replace=False)
    flat    = sem_s[idx].reshape(-1, D2)
    flat_codes_sub = codes[idx].flatten()   # [n_s * S]

    flat_pca = PCA(n_components=min(50, D2)).fit_transform(flat)
    print('[t-SNE/sem+codes] fitting…')
    emb2d = TSNE(n_components=2, perplexity=40, n_iter=1000,
                 random_state=42).fit_transform(flat_pca)

    # colour by code index (only active ones get distinct colours)
    active_codes_list = sorted(used_codes.tolist())
    n_active = len(active_codes_list)
    code_to_color = {c: i for i, c in enumerate(active_codes_list)}
    color_idx = np.array([code_to_color[c] for c in flat_codes_sub])
    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(n_active)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=color_idx, cmap='tab20',
                    vmin=0, vmax=n_active, s=4, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Assigned code (rank among active)')
    ax.set_title(f'Semantic token embeddings — colour = VQ code assignment\n({n_active} active codes)')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_dir / 'vq_semantic_by_code.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[VQ t-SNE] done.')


# ─────────────────────────────────────────────────────────────────────────────

def main(ckpt_path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    config      = load_config()
    ckpt        = torch.load(ckpt_path, map_location=device)
    arch        = infer_arch(ckpt)
    print(f'Arch from ckpt: embed_dim={arch["embed_dim"]}  layers={arch["num_layers"]}  '
          f'patches={arch["ratio_patches"]}  sem_tokens={arch["num_semantic_tokens"]}  '
          f'vq_size={arch["vq_size"]}')
    encoder     = build_encoder(arch, config, device)
    ema_encoder = build_encoder(arch, config, device)
    vq          = build_vq(arch, config, device)
    predictor   = build_predictor(arch, config, device)

    encoder.load_state_dict(ckpt['encoder'])
    ema_key = 'target_encoder' if 'target_encoder' in ckpt else 'encoder_ema'
    ema_encoder.load_state_dict(ckpt[ema_key])
    vq.load_state_dict(ckpt['vector_quantizer'])
    predictor.load_state_dict(ckpt['predictor'])
    print(f'Loaded: {ckpt_path}')

    loader = make_loader(arch, config)

    print('Collecting full-sequence embeddings (PCA / t-SNE / norms)…')
    patch_s, patch_e, sem_s, sem_e, codes, var_idx = collect_embeddings(
        encoder, ema_encoder, vq, config, arch, device, loader)

    print('Collecting predictor alignment (masked forward pass)…')
    cos_p2p, cos_s2p, cos_p2s = collect_predictor_alignment(
        encoder, ema_encoder, predictor, vq, arch, config, device, loader)

    print('\nGenerating plots…')

    # ── Patch embedding plots ──
    plot_pca(patch_s, 'Patch embeddings', 'pca_patches.png', OUT_DIR)
    plot_tsne_positions(patch_s, var_idx, config, 'Patches', 'patch', OUT_DIR)
    plot_predictor_alignment(cos_p2p, 'P2P (patch→patch)', 'alignment_p2p.png', OUT_DIR,
                             x_label='Target patch position')
    plot_predictor_alignment(cos_s2p, 'S2P (semantic→patch)', 'alignment_s2p.png', OUT_DIR,
                             x_label='Target patch position')
    plot_norms(patch_s, patch_e, 'Patches', 'norms_patches.png', OUT_DIR)
    plot_sim_heatmap(patch_s, 'Patches', 'similarity_patches.png', OUT_DIR)

    # ── Semantic token plots ──
    plot_pca(sem_s, 'Semantic tokens', 'pca_semantic.png', OUT_DIR)
    plot_tsne_positions(sem_s, var_idx, config, 'Semantic tokens', 'semantic', OUT_DIR)
    plot_predictor_alignment(cos_p2s, 'P2S (patch→semantic)', 'alignment_p2s.png', OUT_DIR,
                             x_label='Semantic token index')
    plot_norms(sem_s, sem_e, 'Semantic tokens', 'norms_semantic.png', OUT_DIR)
    plot_sim_heatmap(sem_s, 'Semantic tokens', 'similarity_semantic.png', OUT_DIR)

    # ── VQ codebook plots ──
    codebook_vectors = vq._embedding.weight.detach().cpu().numpy()   # [K, D]
    plot_code_usage(codes, arch['vq_size'], OUT_DIR)
    plot_codebook_tsne(codebook_vectors, codes, sem_s, OUT_DIR)

    print(f'\nAll plots saved to: {OUT_DIR}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()
    main(Path(args.ckpt))
