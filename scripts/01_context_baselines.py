"""Recompute context, matched-baseline, and cross-genre analyses.

Run from the released code directory:
    python scripts/01_context_baselines.py

The script regenerates contextual figures and prints the matched-baseline,
cross-genre, downsampling, and sensitivity values reported in the manuscript.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
from collections import Counter, defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np

from _paths import DATA_DIR, FIGURE_DIR

DATA = str(DATA_DIR)
FIG = str(FIGURE_DIR)
SEED = 2026

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
FEATURES_11 = FEATURES_8 + ["valence", "key", "mode"]


def ffloat(value):
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_md5(path):
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


def build_vectors(artist_rows, features):
    valid, raw = [], []
    for row in artist_rows:
        values = [ffloat(row.get(feature)) for feature in features]
        if row.get("genre", "").strip() and all(v is not None and math.isfinite(v) for v in values):
            valid.append(row)
            raw.append(values)

    x_raw = np.array(raw, dtype=float)
    x_cap = np.clip(x_raw, x_raw.mean(0) - 3 * x_raw.std(0), x_raw.mean(0) + 3 * x_raw.std(0))
    denom = x_cap.max(0) - x_cap.min(0)
    denom[denom == 0] = 1
    x_norm = (x_cap - x_cap.min(0)) / denom

    vectors = {row["artist_id"]: x_norm[i] for i, row in enumerate(valid)}
    genres = {row["artist_id"]: row["genre"].strip() for row in valid}
    by_genre = defaultdict(list)
    for i, row in enumerate(valid):
        by_genre[row["genre"].strip()].append(i)
    return x_norm, valid, vectors, genres, by_genre


def cosine(vectors, a, b):
    va, vb = vectors[a], vectors[b]
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def decade(year):
    return year // 10 * 10


def degree_bin(value):
    if value <= 0:
        return 0
    if value == 1:
        return 1
    if value <= 3:
        return 2
    if value <= 7:
        return 3
    if value <= 15:
        return 4
    if value <= 31:
        return 5
    if value <= 63:
        return 6
    return 7


def main():
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs(FIG, exist_ok=True)

    artists = read_csv(os.path.join(DATA, "data_by_artist.csv"))
    influence = read_csv(os.path.join(DATA, "influence_data.csv"))
    x8, valid8, vec8, artist_genre, by_genre = build_vectors(artists, FEATURES_8)
    _, valid11, vec11, _, _ = build_vectors(artists, FEATURES_11)

    outdegree, indegree = Counter(), Counter()
    adj, rev = defaultdict(list), defaultdict(list)
    years, genre, names = {}, {}, {}
    edges, seen = [], set()
    reversed_edges = 0
    for row in influence:
        a, b = row["influencer_id"], row["follower_id"]
        if (a, b) in seen:
            continue
        seen.add((a, b))
        edges.append((a, b, row))
        outdegree[a] += 1
        indegree[b] += 1
        adj[a].append(b)
        rev[b].append(a)
        for id_key, name_key, genre_key, year_key in [
            ("influencer_id", "influencer_name", "influencer_main_genre", "influencer_active_start"),
            ("follower_id", "follower_name", "follower_main_genre", "follower_active_start"),
        ]:
            artist_id = row[id_key]
            names[artist_id] = row[name_key]
            genre[artist_id] = row[genre_key].strip()
            year = ffloat(row[year_key])
            if year is not None:
                years[artist_id] = int(year)
        yi, yf = ffloat(row["influencer_active_start"]), ffloat(row["follower_active_start"])
        if yi is not None and yf is not None and yf < yi:
            reversed_edges += 1

    def time_ok(a, b):
        return a in years and b in years and years[b] >= years[a]

    print("Input MD5 hashes:")
    for name in ["data_by_artist.csv", "influence_data.csv", "full_music_data.csv"]:
        print(name, file_md5(os.path.join(DATA, name)))
    print("Valid 8-feature artists:", len(valid8))
    print("Valid 11-feature artists:", len(valid11))
    print("Unique edges:", len(edges), "time-reversed edges:", reversed_edges)

    edge_set = {(a, b) for a, b, _ in edges}
    pr_nodes = [i for i, g in genre.items() if g == "Pop/Rock" and i in vec8 and i in years]
    pr_edges = [
        (a, b)
        for a, b, _ in edges
        if genre.get(a) == "Pop/Rock" and genre.get(b) == "Pop/Rock" and a in vec8 and b in vec8
    ]
    pr_time = [(a, b) for a, b in pr_edges if time_ok(a, b)]
    print("Pop/Rock complete same-genre edges:", len(pr_edges), "time-respecting:", len(pr_time))

    by_key = defaultdict(list)
    for artist_id in pr_nodes:
        by_key[(decade(years[artist_id]), degree_bin(outdegree[artist_id]), degree_bin(indegree[artist_id]))].append(
            artist_id
        )

    used_edges, matched_non_edges = [], []
    failures = 0
    for a, b in pr_time:
        ca = by_key[(decade(years[a]), degree_bin(outdegree[a]), degree_bin(indegree[a]))]
        cb = by_key[(decade(years[b]), degree_bin(outdegree[b]), degree_bin(indegree[b]))]
        match = None
        for _ in range(200):
            aa, bb = random.choice(ca), random.choice(cb)
            if aa != bb and (aa, bb) not in edge_set and time_ok(aa, bb):
                match = (aa, bb)
                break
        if match:
            used_edges.append((a, b))
            matched_non_edges.append(match)
        else:
            failures += 1

    edge_sims = np.array([cosine(vec8, a, b) for a, b in used_edges])
    base_sims = np.array([cosine(vec8, a, b) for a, b in matched_non_edges])
    diff = edge_sims - base_sims
    perm = [(diff * rng.choice([-1, 1], len(diff))).mean() for _ in range(3000)]
    p_value = (np.sum(np.abs(perm) >= abs(diff.mean())) + 1) / 3001
    by_influencer = defaultdict(list)
    for i, (a, _) in enumerate(used_edges):
        by_influencer[a].append(i)
    influencers = list(by_influencer)
    boot = []
    for _ in range(800):
        idx = []
        for influencer in rng.choice(influencers, len(influencers), replace=True):
            idx.extend(by_influencer[influencer])
        boot.append(diff[idx].mean())
    print(
        "Matched similarity:",
        len(diff),
        "failures",
        failures,
        "unique matched nonedges",
        len(set(matched_non_edges)),
        "edge mean",
        edge_sims.mean(),
        "baseline mean",
        base_sims.mean(),
        "diff",
        diff.mean(),
        "CI",
        np.percentile(boot, [2.5, 97.5]),
        "p",
        p_value,
    )

    pairs11 = [
        ((a, b), (aa, bb))
        for (a, b), (aa, bb) in zip(used_edges, matched_non_edges)
        if a in vec11 and b in vec11 and aa in vec11 and bb in vec11
    ]
    sims11 = np.array([cosine(vec11, a, b) - cosine(vec11, aa, bb) for (a, b), (aa, bb) in pairs11])
    print("11-feature sensitivity mean difference:", sims11.mean())

    extremeness = {}
    for artist_id, vector in vec8.items():
        g = artist_genre.get(artist_id)
        if g in by_genre:
            extremeness[artist_id] = float(np.linalg.norm(vector - x8[by_genre[g]].mean(0)))

    q90 = np.percentile([extremeness[i] for i in pr_nodes], 90)
    q80 = np.percentile([extremeness[i] for i in pr_nodes], 80)
    qin25 = np.percentile([indegree[i] for i in pr_nodes], 25)
    qout50 = np.percentile([outdegree[i] for i in pr_nodes], 50)
    zero_outdegree_outliers = [i for i in pr_nodes if outdegree[i] == 0 and indegree[i] <= qin25 and extremeness[i] >= q90]
    acoustic_subset = [
        i
        for i in pr_nodes
        if years[i] <= 1960 and indegree[i] <= qin25 and 0 < outdegree[i] <= qout50 and extremeness[i] >= q80
    ]
    late = {i for i in pr_nodes if years[i] in (1980, 1990)}

    def enumerate_paths(starts, maxlen=4):
        paths = []
        for start in starts:
            queue = deque([(start, [start])])
            while queue:
                node, path = queue.popleft()
                if len(path) > maxlen + 1:
                    continue
                if node != start and node in late and len(path) >= 3:
                    paths.append(path)
                    continue
                if len(path) == maxlen + 1:
                    continue
                for nxt in adj[node]:
                    if nxt in path or nxt not in vec8 or genre.get(nxt) != "Pop/Rock" or nxt not in years:
                        continue
                    if years[nxt] < years[node]:
                        continue
                    queue.append((nxt, path + [nxt]))
        return paths

    print("Zero-outdegree acoustic outliers:", len(zero_outdegree_outliers))
    print("Acoustic-extremeness threshold sensitivity:")
    for q in [70, 80, 90, 95]:
        qext = np.percentile([extremeness[i] for i in pr_nodes], q)
        qsubset = [
            i
            for i in pr_nodes
            if years[i] <= 1960
            and indegree[i] <= qin25
            and 0 < outdegree[i] <= qout50
            and extremeness[i] >= qext
        ]
        qpaths = enumerate_paths(qsubset)
        qsims = np.array(
            [np.mean([cosine(vec8, path[i], path[i + 1]) for i in range(len(path) - 1)]) for path in qpaths]
        )
        print(
            q,
            "starts",
            len(qsubset),
            "reachable share",
            len({p[0] for p in qpaths}) / float(len(qsubset)) if qsubset else float("nan"),
            "paths",
            len(qpaths),
            "endpoints",
            len({p[-1] for p in qpaths}),
            "mean sim",
            float(qsims.mean()) if len(qsims) else float("nan"),
        )

    # Mahalanobis-style extremeness checks whether correlated descriptors drive seed selection.
    pr_vecs = np.array([vec8[i] for i in pr_nodes])
    cov = np.cov(pr_vecs, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-6
    inv_cov = np.linalg.pinv(cov)
    pr_mean = pr_vecs.mean(0)
    mahal = {i: float(np.sqrt((vec8[i] - pr_mean).dot(inv_cov).dot(vec8[i] - pr_mean))) for i in pr_nodes}
    mahal_q80 = np.percentile([mahal[i] for i in pr_nodes], 80)
    mahal_subset = [
        i
        for i in pr_nodes
        if years[i] <= 1960 and indegree[i] <= qin25 and 0 < outdegree[i] <= qout50 and mahal[i] >= mahal_q80
    ]
    mahal_paths = enumerate_paths(mahal_subset)
    print(
        "Mahalanobis extremeness sensitivity:",
        "starts",
        len(mahal_subset),
        "overlap with Euclidean acoustic subset",
        len(set(mahal_subset) & set(acoustic_subset)),
        "reachable share",
        len({p[0] for p in mahal_paths}) / float(len(mahal_subset)) if mahal_subset else float("nan"),
        "paths",
        len(mahal_paths),
        "endpoints",
        len({p[-1] for p in mahal_paths}),
    )

    # Cross-genre comparison used in Table 6 and Figure 6.
    series = defaultdict(lambda: defaultdict(set))
    first_observed = defaultdict(lambda: 9999)
    for a, b, row in edges:
        for genre_key, year_key in [
            ("influencer_main_genre", "influencer_active_start"),
            ("follower_main_genre", "follower_active_start"),
        ]:
            g = row[genre_key].strip()
            year = ffloat(row[year_key])
            if g and year is not None:
                first_observed[g] = min(first_observed[g], int(year))
        g = row["follower_main_genre"].strip()
        year = ffloat(row["follower_active_start"])
        if g and year is not None:
            series[g][decade(int(year))].add(row["follower_id"])

    genre_rows = []
    for g, idxs in by_genre.items():
        if len(idxs) < 20 or not series[g]:
            continue
        counts = {d: len(s) for d, s in series[g].items()}
        peak = max(counts, key=counts.get)
        delay = peak - first_observed[g]
        dispersion = float(np.mean(np.linalg.norm(x8[idxs] - x8[idxs].mean(0), axis=1)))
        nodes = [i for i in genre if genre.get(i) == g and i in vec8 and i in years]
        by_decade = defaultdict(list)
        for node in nodes:
            by_decade[decade(years[node])].append(node)
        genre_edges = [
            (a, b)
            for a, b, _ in edges
            if genre.get(a) == g and genre.get(b) == g and a in vec8 and b in vec8 and time_ok(a, b)
        ]
        edge_advantage = float("nan")
        if len(genre_edges) >= 20:
            sample = genre_edges if len(genre_edges) <= 3000 else random.sample(genre_edges, 3000)
            edge_vals, base_vals = [], []
            for a, b in sample:
                ca, cb = by_decade.get(decade(years[a]), []), by_decade.get(decade(years[b]), [])
                match = None
                for _ in range(100):
                    if not ca or not cb:
                        break
                    aa, bb = random.choice(ca), random.choice(cb)
                    if aa != bb and (aa, bb) not in edge_set and time_ok(aa, bb):
                        match = (aa, bb)
                        break
                if match:
                    edge_vals.append(cosine(vec8, a, b))
                    base_vals.append(cosine(vec8, *match))
            if base_vals:
                edge_advantage = float(np.mean(edge_vals) - np.mean(base_vals))
        genre_rows.append((g, len(idxs), first_observed[g], peak, delay, counts[peak], dispersion, len(genre_edges), edge_advantage))

    print("Cross-genre rows:")
    for row in sorted(genre_rows, key=lambda r: -r[5]):
        print(row)

    print("Pop/Rock artist-size downsampling:")
    pr_by_decade = defaultdict(list)
    for node in pr_nodes:
        pr_by_decade[decade(years[node])].append(node)
    pr_node_set = set(pr_nodes)
    pr_intra_time_edges = [(a, b) for a, b in pr_time if a in pr_node_set and b in pr_node_set]
    for target_n in [142, 208, 408, 677]:
        peak_counts, dispersions, edge_counts = [], [], []
        for _ in range(500):
            sample = set(rng.choice(pr_nodes, target_n, replace=False))
            decade_counts = Counter(decade(years[i]) for i in sample)
            peak_counts.append(max(decade_counts.values()))
            sample_vecs = np.array([vec8[i] for i in sample])
            dispersions.append(float(np.mean(np.linalg.norm(sample_vecs - sample_vecs.mean(0), axis=1))))
            edge_counts.append(sum(1 for a, b in pr_intra_time_edges if a in sample and b in sample))
        print(
            target_n,
            "peak count median/IQR",
            float(np.median(peak_counts)),
            np.percentile(peak_counts, [25, 75]).tolist(),
            "dispersion median/IQR",
            float(np.median(dispersions)),
            np.percentile(dispersions, [25, 75]).tolist(),
            "same-genre time edges median/IQR",
            float(np.median(edge_counts)),
            np.percentile(edge_counts, [25, 75]).tolist(),
        )

    comparable = [row for row in genre_rows if not math.isnan(row[8])]
    plot_rows = sorted(comparable, key=lambda r: -r[4])
    labels = [row[0] for row in plot_rows]
    y_pos = np.arange(len(labels))
    colors = ["#d62728" if label == "Pop/Rock" else "#7f7f7f" for label in labels]
    fig, axes = plt.subplots(1, 3, figsize=(10, 5.5), sharey=True)
    axes[0].barh(y_pos, [row[4] for row in plot_rows], color=colors)
    axes[1].barh(y_pos, [row[6] for row in plot_rows], color=colors)
    axes[2].barh(y_pos, [row[8] for row in plot_rows], color=colors)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    for ax, title, xlabel in zip(
        axes,
        ["Delay", "Dispersion", "Edge advantage"],
        ["Years", "Mean distance", "Cosine difference"],
    ):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.axvline(0, color="0.2", lw=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "figure6_genre_comparison.png"), dpi=300)
    plt.close()

    # Dot plot replacing the previous radar chart.
    genres_for_plot = ["Pop/Rock", "Reggae", "Jazz", "Classical"]
    y = np.arange(len(FEATURES_8))
    markers = ["o", "s", "^", "D"]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#6f4ba3"]
    plt.figure(figsize=(8, 5.8))
    for gi, g in enumerate(genres_for_plot):
        vals = x8[by_genre[g]].mean(0)
        se = x8[by_genre[g]].std(0) / np.sqrt(len(by_genre[g]))
        plt.errorbar(vals, y + (gi - 1.5) * 0.16, xerr=1.96 * se, fmt=markers[gi], color=colors[gi], capsize=2,
                     label=f"{g} (n={len(by_genre[g])})", markersize=4)
    plt.yticks(y, FEATURES_8)
    plt.xlabel("Mean normalized artist-level descriptor (95% CI)")
    plt.gca().invert_yaxis()
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "figure4_dotplot.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
