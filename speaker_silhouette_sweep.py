"""
Speaker Silhouette Sweep Analysis

Loads pre-computed features from run_silhouette_plot_comparison and measures
speaker-wise silhouette scores as the number of speakers increases (2..20).

For each N:
  - Multiple random trials selecting N speakers
  - 1:1 nat/syn balanced sampling per speaker
  - Compute silhouette_score with speaker labels (cosine metric)

Also generates:
  - All-speakers silhouette diagram (side-by-side: baseline vs ours)
  - Top-5 speakers silhouette diagram (side-by-side: baseline vs ours)

Usage:
    python speaker_silhouette_sweep.py \
        --data_dir results/ssl_clustering \
        --output_dir results/ssl_clustering \
        --n_trials 10
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse


# =============================================================================
# Balanced sampling helpers
# =============================================================================
def balanced_sample_by_speaker(speaker_ids, labels, selected_speakers=None, random_state=42):
    """
    Sample equal number of samples per speaker with 1:1 natural/synthetic ratio.

    Args:
        speaker_ids: array of speaker IDs
        labels: array of labels (0=natural, 1=synthetic)
        selected_speakers: list of speakers to include (None = all)
        random_state: random seed

    Returns:
        sampled_indices: list of sampled indices
        n_per_class: number of samples per class per speaker
    """
    rng = np.random.RandomState(random_state)
    speaker_ids = np.array(speaker_ids)
    labels = np.array(labels)

    if selected_speakers is None:
        selected_speakers = sorted(np.unique(speaker_ids))

    speaker_nat = {}
    speaker_syn = {}
    for spk in selected_speakers:
        spk_mask = speaker_ids == spk
        speaker_nat[spk] = np.where(spk_mask & (labels == 0))[0]
        speaker_syn[spk] = np.where(spk_mask & (labels == 1))[0]

    n_per_class = min(
        min(len(speaker_nat[spk]), len(speaker_syn[spk]))
        for spk in selected_speakers
    )

    sampled_indices = []
    for spk in selected_speakers:
        sampled_nat = rng.choice(speaker_nat[spk], n_per_class, replace=False)
        sampled_syn = rng.choice(speaker_syn[spk], n_per_class, replace=False)
        sampled_indices.extend(sampled_nat.tolist() + sampled_syn.tolist())

    return np.array(sampled_indices), n_per_class


# =============================================================================
# Silhouette diagram plotting
# =============================================================================
def plot_silhouette_by_speaker(ax, silhouette_vals, speaker_ids, color_map,
                                speakers_to_show=None, alpha=0.7):
    """
    Plot silhouette diagram with speaker clusters.
    """
    unique_speakers = sorted(np.unique(speaker_ids))
    if speakers_to_show is not None:
        show_set = set(speakers_to_show)
        unique_speakers = [s for s in unique_speakers if s in show_set]

    y_lower = 10
    for spk in unique_speakers:
        mask = speaker_ids == spk
        cluster_values = np.sort(silhouette_vals[mask])
        size_cluster = len(cluster_values)
        y_upper = y_lower + size_cluster

        spk_mean = np.mean(cluster_values)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, cluster_values,
            alpha=alpha, color=color_map[spk]
        )

        ax.text(-0.05, y_lower + 0.5 * size_cluster,
                f'{spk} ({spk_mean:.3f})',
                fontsize=8, va='center', ha='right')

        y_lower = y_upper + 10

    return y_lower


# =============================================================================
# Speaker sweep analysis
# =============================================================================
def run_speaker_sweep(baseline_features, ours_features, labels, speakers,
                      n_trials=10, random_state=42):
    """
    Sweep N = 2..max_speakers: measure speaker silhouette for baseline and ours.

    Returns:
        list of dicts with per-N results
    """
    unique_speakers = sorted(np.unique(speakers))
    n_total = len(unique_speakers)

    results = []
    for n_spk in range(2, n_total + 1):
        baseline_scores = []
        ours_scores = []

        for trial in range(n_trials):
            trial_rng = np.random.RandomState(random_state + trial)

            # For n_spk == n_total, use all speakers (no randomness)
            if n_spk == n_total:
                selected = list(unique_speakers)
            else:
                selected = list(trial_rng.choice(unique_speakers, n_spk, replace=False))

            # Balanced 1:1 nat/syn per speaker
            idx, n_per_class = balanced_sample_by_speaker(
                speakers, labels,
                selected_speakers=selected,
                random_state=random_state + trial
            )

            if n_per_class == 0 or len(idx) < 2 * n_spk:
                continue

            b_sil = silhouette_score(
                baseline_features[idx], speakers[idx], metric='cosine'
            )
            o_sil = silhouette_score(
                ours_features[idx], speakers[idx], metric='cosine'
            )

            baseline_scores.append(b_sil)
            ours_scores.append(o_sil)

        if not baseline_scores:
            continue

        b_mean, b_std = np.mean(baseline_scores), np.std(baseline_scores)
        o_mean, o_std = np.mean(ours_scores), np.std(ours_scores)

        results.append({
            'n_speakers': n_spk,
            'baseline_mean': b_mean,
            'baseline_std': b_std,
            'ours_mean': o_mean,
            'ours_std': o_std,
            'n_trials': len(baseline_scores),
            'n_per_class': n_per_class,
        })

        marker = " << ours < baseline" if o_mean < b_mean else ""
        print(f"  N={n_spk:2d}: Baseline={b_mean:.4f}+/-{b_std:.4f}, "
              f"Ours={o_mean:.4f}+/-{o_std:.4f}{marker}")

    return results


# =============================================================================
# Plotting
# =============================================================================
def plot_sweep(results_df, output_dir):
    """Plot speaker sweep: N speakers vs silhouette score."""
    n_spk = results_df['n_speakers'].values
    b_mean = results_df['baseline_mean'].values
    b_std = results_df['baseline_std'].values
    o_mean = results_df['ours_mean'].values
    o_std = results_df['ours_std'].values

    fig, ax = plt.subplots(figsize=(10, 6))

    # Baseline
    ax.plot(n_spk, b_mean, 'o-', color='tab:blue', label='Baseline (WavLM last layer)', linewidth=2)
    ax.fill_between(n_spk, b_mean - b_std, b_mean + b_std, color='tab:blue', alpha=0.15)

    # Ours
    ax.plot(n_spk, o_mean, 's-', color='tab:orange', label='Ours (WavLM L8+L22, SI)', linewidth=2)
    ax.fill_between(n_spk, o_mean - o_std, o_mean + o_std, color='tab:orange', alpha=0.15)

    # Find crossover point (ours < baseline)
    crossover = None
    for i in range(len(n_spk)):
        if o_mean[i] < b_mean[i]:
            crossover = n_spk[i]
            break

    if crossover is not None:
        ax.axvline(x=crossover, color='red', linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f'Crossover at N={crossover}')

    ax.set_xlabel('Number of Speakers', fontsize=12)
    ax.set_ylabel('Speaker Silhouette Score (cosine)', fontsize=12)
    ax.set_title('Speaker Silhouette vs Number of Speakers', fontsize=14)
    ax.set_xticks(n_spk)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        path = output_dir / f'silhouette_speaker_sweep.{ext}'
        plt.savefig(path, dpi=150 if ext == 'png' else 300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'silhouette_speaker_sweep.png'}")


def plot_speaker_diagrams(baseline_features, ours_features, labels, speakers,
                          output_dir, n_speaker_components=5):
    """Generate all-speakers and top-5 silhouette diagrams."""
    unique_spks = sorted(np.unique(speakers))

    # Balanced 1:1 nat/syn for all speakers
    idx, n_per_class = balanced_sample_by_speaker(speakers, labels)
    print(f"  Speaker diagrams: {len(unique_spks)} speakers, "
          f"{n_per_class} nat + {n_per_class} syn per speaker, {len(idx)} total")

    bal_baseline = baseline_features[idx]
    bal_ours = ours_features[idx]
    bal_speakers = speakers[idx]

    # Compute per-sample silhouette
    print("  Computing per-sample speaker silhouette (cosine)...")
    baseline_sil = silhouette_samples(bal_baseline, bal_speakers, metric='cosine')
    ours_sil = silhouette_samples(bal_ours, bal_speakers, metric='cosine')

    baseline_mean = np.mean(baseline_sil)
    ours_mean = np.mean(ours_sil)
    print(f"  Baseline speaker silhouette: {baseline_mean:.4f}")
    print(f"  Ours speaker silhouette:     {ours_mean:.4f}")

    # Per-speaker breakdown
    spk_scores = {}
    print(f"\n  {'Speaker':<12} {'Baseline':>10} {'Ours':>10}")
    print(f"  {'-'*34}")
    for spk in unique_spks:
        mask = bal_speakers == spk
        b = np.mean(baseline_sil[mask])
        o = np.mean(ours_sil[mask])
        spk_scores[spk] = {'baseline': b, 'ours': o}
        print(f"  {spk:<12} {b:>10.4f} {o:>10.4f}")

    # Color map
    color_map = {spk: plt.cm.tab20(i) for i, spk in enumerate(unique_spks)}

    # x-axis limits
    x_min = min(baseline_sil.min(), ours_sil.min()) - 0.05
    x_max = max(baseline_sil.max(), ours_sil.max()) + 0.05

    # === All speakers plot ===
    print("\n  Generating all-speakers silhouette plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 16), sharey=False)

    plot_silhouette_by_speaker(ax1, baseline_sil, bal_speakers, color_map=color_map)
    ax1.axvline(x=baseline_mean, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Mean ({baseline_mean:.4f})')
    ax1.set_xlabel('Silhouette Coefficient (cosine)', fontsize=11)
    ax1.set_ylabel('Samples (sorted by speaker)', fontsize=11)
    ax1.set_title('Baseline (WavLM last layer)', fontsize=12)
    ax1.set_xlim(x_min, x_max)
    ax1.set_yticks([])
    ax1.legend(loc='lower right', fontsize=10)

    plot_silhouette_by_speaker(ax2, ours_sil, bal_speakers, color_map=color_map)
    ax2.axvline(x=ours_mean, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Mean ({ours_mean:.4f})')
    ax2.set_xlabel('Silhouette Coefficient (cosine)', fontsize=11)
    ax2.set_ylabel('Samples (sorted by speaker)', fontsize=11)
    ax2.set_title(f'Ours (WavLM L8+L22, SI n={n_speaker_components})', fontsize=12)
    ax2.set_xlim(x_min, x_max)
    ax2.set_yticks([])
    ax2.legend(loc='lower right', fontsize=10)

    plt.suptitle('Speaker-wise Silhouette Plot (All Speakers)', fontsize=14, y=1.01)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(output_dir / f'silhouette_speaker_all.{ext}',
                    dpi=150 if ext == 'png' else 300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'silhouette_speaker_all.png'}")

    # === 5 speakers where ours loses most vs baseline ===
    top5 = sorted(spk_scores,
                  key=lambda s: spk_scores[s]['baseline'] - spk_scores[s]['ours'],
                  reverse=True)[:5]
    print(f"\n  5 speakers where Ours loses most vs Baseline (baseline - ours):")
    for spk in top5:
        diff = spk_scores[spk]['baseline'] - spk_scores[spk]['ours']
        print(f"    {spk}: Baseline={spk_scores[spk]['baseline']:.4f}, "
              f"Ours={spk_scores[spk]['ours']:.4f}, diff={diff:+.4f}")

    top5_set = set(top5)
    top5_mask = np.array([s in top5_set for s in bal_speakers])
    top5_baseline_sil = baseline_sil[top5_mask]
    top5_ours_sil = ours_sil[top5_mask]
    top5_speakers = bal_speakers[top5_mask]
    top5_b_mean = np.mean(top5_baseline_sil)
    top5_o_mean = np.mean(top5_ours_sil)

    print(f"\n  Generating top-5 speakers silhouette plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=False)

    plot_silhouette_by_speaker(ax1, top5_baseline_sil, top5_speakers,
                                color_map=color_map, speakers_to_show=top5)
    ax1.axvline(x=top5_b_mean, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Mean ({top5_b_mean:.4f})')
    ax1.set_xlabel('Silhouette Coefficient (cosine)', fontsize=11)
    ax1.set_ylabel('Samples (sorted by speaker)', fontsize=11)
    ax1.set_title('Baseline (WavLM last layer)', fontsize=12)
    ax1.set_xlim(x_min, x_max)
    ax1.set_yticks([])
    ax1.legend(loc='lower right', fontsize=10)

    plot_silhouette_by_speaker(ax2, top5_ours_sil, top5_speakers,
                                color_map=color_map, speakers_to_show=top5)
    ax2.axvline(x=top5_o_mean, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Mean ({top5_o_mean:.4f})')
    ax2.set_xlabel('Silhouette Coefficient (cosine)', fontsize=11)
    ax2.set_ylabel('Samples (sorted by speaker)', fontsize=11)
    ax2.set_title(f'Ours (WavLM L8+L22, SI n={n_speaker_components})', fontsize=12)
    ax2.set_xlim(x_min, x_max)
    ax2.set_yticks([])
    ax2.legend(loc='lower right', fontsize=10)

    plt.suptitle('Speaker-wise Silhouette Plot (5 Speakers, Largest Drop)', fontsize=14, y=1.01)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(output_dir / f'silhouette_speaker_largest_drop5.{ext}',
                    dpi=150 if ext == 'png' else 300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'silhouette_speaker_largest_drop5.png'}")

    # Save per-speaker results CSV
    rows = [{'speaker': spk, 'baseline_silhouette': spk_scores[spk]['baseline'],
             'ours_silhouette': spk_scores[spk]['ours'], 'n_samples_per_class': n_per_class}
            for spk in unique_spks]
    pd.DataFrame(rows).to_csv(output_dir / 'silhouette_speaker_results.csv', index=False)
    print(f"  Saved: {output_dir / 'silhouette_speaker_results.csv'}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Speaker Silhouette Sweep Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing common_*.npz from run_silhouette_plot_comparison')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as data_dir)')
    parser.add_argument('--n_trials', type=int, default=10,
                        help='Number of random trials per speaker count')
    parser.add_argument('--n_speaker_components', type=int, default=5,
                        help='n_speaker_components used (for plot titles)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-computed features (.npz float16 -> float32)
    print("Loading pre-computed features...")
    baseline_features = np.load(data_dir / 'common_baseline_features.npz')['data'].astype(np.float32)
    ours_features = np.load(data_dir / 'common_ours_features.npz')['data'].astype(np.float32)
    labels = np.load(data_dir / 'common_labels.npz')['data']
    speakers = np.load(data_dir / 'common_speakers.npz', allow_pickle=True)['data']

    unique_spks = sorted(np.unique(speakers))
    print(f"  Samples: {len(labels)}, Speakers: {len(unique_spks)}")
    print(f"  Natural: {np.sum(labels == 0)}, Synthetic: {np.sum(labels == 1)}")
    print(f"  Baseline features: {baseline_features.shape}")
    print(f"  Ours features:     {ours_features.shape}")

    # === Speaker sweep ===
    print("\n" + "=" * 60)
    print("Speaker Silhouette Sweep (N = 2..{})".format(len(unique_spks)))
    print("=" * 60)

    sweep_results = run_speaker_sweep(
        baseline_features, ours_features, labels, speakers,
        n_trials=args.n_trials
    )

    sweep_df = pd.DataFrame(sweep_results)
    sweep_csv = output_dir / 'silhouette_speaker_sweep.csv'
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"\nSaved: {sweep_csv}")

    # Sweep plot
    plot_sweep(sweep_df, output_dir)

    # === Speaker silhouette diagrams (all + top5) ===
    print("\n" + "=" * 60)
    print("Speaker Silhouette Diagrams")
    print("=" * 60)

    plot_speaker_diagrams(
        baseline_features, ours_features, labels, speakers,
        output_dir, n_speaker_components=args.n_speaker_components
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
