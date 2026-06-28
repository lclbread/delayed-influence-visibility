"""Regenerate descriptive context figures and preprocessing counts.

This script supports the manuscript sections on data preprocessing, descriptor
correlations, follower cohorts, descriptive growth, and song-level decade
profiles. It does not test delayed reachability.
"""

from __future__ import annotations

import csv
import math
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np

from _paths import DATA_DIR, FIGURE_DIR


DATA = str(DATA_DIR)
FIG = str(FIGURE_DIR)

FEATURES_8 = [
    "danceability",
    "energy",
    "tempo",
    "loudness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
]

FEATURE_LABELS = [
    "Danceability",
    "Energy",
    "Tempo",
    "Loudness",
    "Acousticness",
    "Instrumentalness",
    "Liveness",
    "Speechiness",
]


def ffloat(value):
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def decade(year):
    return year // 10 * 10


def build_artist_vectors(artist_rows):
    valid, raw = [], []
    for row in artist_rows:
        values = [ffloat(row.get(feature)) for feature in FEATURES_8]
        if row.get("genre", "").strip() and all(v is not None and math.isfinite(v) for v in values):
            valid.append(row)
            raw.append(values)
    x_raw = np.array(raw, dtype=float)
    x_cap = np.clip(x_raw, x_raw.mean(0) - 3 * x_raw.std(0), x_raw.mean(0) + 3 * x_raw.std(0))
    denom = x_cap.max(0) - x_cap.min(0)
    denom[denom == 0] = 1
    x_norm = (x_cap - x_cap.min(0)) / denom
    return valid, x_norm


def main():
    os.makedirs(FIG, exist_ok=True)
    artist_rows = read_csv(os.path.join(DATA, "data_by_artist.csv"))
    influence_rows = read_csv(os.path.join(DATA, "influence_data.csv"))
    song_rows = read_csv(os.path.join(DATA, "full_music_data.csv"))
    valid_artists, x_norm = build_artist_vectors(artist_rows)

    artist_index = {row["artist_id"]: i for i, row in enumerate(valid_artists)}
    valid_genres = sorted({row["genre"].strip() for row in valid_artists if row["genre"].strip()})
    poprock_ids = [row["artist_id"] for row in valid_artists if row["genre"].strip() == "Pop/Rock"]
    reggae_ids = [row["artist_id"] for row in valid_artists if row["genre"].strip() == "Reggae"]

    seen_edges = set()
    time_reversed = 0
    poprock_edges = []
    poprock_time_edges = []
    follower_series = defaultdict(lambda: defaultdict(set))
    cohort_genres = ["Pop/Rock", "Reggae", "Jazz", "Classical"]

    for row in influence_rows:
        a, b = row["influencer_id"], row["follower_id"]
        if (a, b) in seen_edges:
            continue
        seen_edges.add((a, b))
        yi = ffloat(row.get("influencer_active_start"))
        yf = ffloat(row.get("follower_active_start"))
        if yi is not None and yf is not None and yf < yi:
            time_reversed += 1
        if row.get("follower_main_genre", "").strip() in cohort_genres and yf is not None:
            follower_series[row["follower_main_genre"].strip()][decade(int(yf))].add(b)
        if (
            row.get("influencer_main_genre", "").strip() == "Pop/Rock"
            and row.get("follower_main_genre", "").strip() == "Pop/Rock"
            and a in artist_index
            and b in artist_index
        ):
            poprock_edges.append((a, b))
            if yi is not None and yf is not None and yf >= yi:
                poprock_time_edges.append((a, b))

    print("Preprocessing counts:")
    print("Song records", len(song_rows))
    print("Artist records raw/used", len(artist_rows), len(valid_artists))
    print("Influence edges raw/unique", len(influence_rows), len(seen_edges))
    print("Time-reversed edges", time_reversed)
    print("Valid genres", len(valid_genres), valid_genres)
    print("Pop/Rock artists", len(poprock_ids))
    print("Reggae artists", len(reggae_ids))
    print("Pop/Rock edge pairs", len(poprock_edges))
    print("Time-respecting Pop/Rock pairs", len(poprock_time_edges))

    # Figure: Pop/Rock descriptor correlations.
    pr_matrix = x_norm[[artist_index[artist_id] for artist_id in poprock_ids], :]
    corr = np.corrcoef(pr_matrix, rowvar=False)
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_LABELS)))
    ax.set_xticklabels(FEATURE_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(FEATURE_LABELS)))
    ax.set_yticklabels(FEATURE_LABELS)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(
                j,
                i,
                f"{corr[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(corr[i, j]) >= 0.55 else "#222222",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
    ax.set_title("Pop/Rock artist-level acoustic descriptor correlations")
    fig.tight_layout()
    out = os.path.join(FIG, "figure2_heatmap.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)

    # Figure: unique listed followers by active-start decade.
    decades = sorted({d for values in follower_series.values() for d in values})
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for genre in cohort_genres:
        counts = [len(follower_series[genre].get(d, set())) for d in decades]
        ax.plot(decades, counts, marker="o", linewidth=1.8, label=genre)
    ax.set_xlabel("Follower active-start decade")
    ax.set_ylabel("Unique listed followers")
    ax.set_title("Listed follower cohorts by genre")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG, "figure_followers_empirical.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Saved", out)

    # Figure: descriptive decade-over-decade follower growth for Pop/Rock and Reggae.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for genre, color in [("Pop/Rock", "#1f77b4"), ("Reggae", "#d62728")]:
        counts = [len(follower_series[genre].get(d, set())) for d in decades]
        growth_decades, growth_values = [], []
        for prev, current, d in zip(counts[:-1], counts[1:], decades[1:]):
            if prev > 0:
                growth_decades.append(d)
                growth_values.append((current - prev) / prev * 100.0)
        ax.plot(growth_decades, growth_values, marker="o", linewidth=1.8, color=color, label=genre)
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.set_xlabel("Follower active-start decade")
    ax.set_ylabel("Decade-over-decade change (%)")
    ax.set_title("Descriptive listed-follower growth")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG, "figure_gei_empirical.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Saved", out)

    # Figure: Pop/Rock song-level descriptors by release decade.
    plot_features = ["danceability", "energy", "acousticness"]
    feature_labels = ["Danceability", "Energy", "Acousticness"]
    pop_songs = []
    for row in song_rows:
        year = ffloat(row.get("year"))
        if row.get("genre", "").strip() == "Pop/Rock" and year is not None:
            values = [ffloat(row.get(feature)) for feature in plot_features]
            if all(value is not None and math.isfinite(value) for value in values):
                pop_songs.append((decade(int(year)), values))
    song_decades = [d for d in sorted({d for d, _ in pop_songs}) if 1950 <= d <= 2010]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), sharex=True)
    for axis, feature_index in zip(axes, range(len(plot_features))):
        values_by_decade = [
            [values[feature_index] for d, values in pop_songs if d == target_decade]
            for target_decade in song_decades
        ]
        box = axis.boxplot(values_by_decade, patch_artist=True, labels=[str(d) for d in song_decades], showfliers=False)
        for patch in box["boxes"]:
            patch.set_facecolor("#9ecae1")
            patch.set_edgecolor("#3b6f8f")
        axis.set_title(feature_labels[feature_index])
        axis.set_xlabel("Release decade")
        axis.tick_params(axis="x", rotation=45)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Song-level value")
    fig.suptitle("Pop/Rock song-level descriptors by release decade", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG, "figure3_boxplot.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    main()
