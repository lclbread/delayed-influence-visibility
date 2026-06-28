"""Recompute low-degree delayed-visibility and temporal rewiring analyses.

This script recomputes the added analyses directly from the ICM 2021 CSV files.
It keeps the main population as early low-degree Pop/Rock artists; acoustic
extremeness is treated as a descriptive subset rather than as the primary
contrast.
"""

from __future__ import annotations

import csv
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


def ffloat(value):
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_vectors(artist_rows):
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

    vectors = {row["artist_id"]: x_norm[i] for i, row in enumerate(valid)}
    genres = {row["artist_id"]: row["genre"].strip() for row in valid}
    by_genre = defaultdict(list)
    for i, row in enumerate(valid):
        by_genre[row["genre"].strip()].append(i)
    return x_norm, vectors, genres, by_genre


def decade(year):
    return year // 10 * 10


def summarize_paths(label, starts, paths, years):
    endpoints = {p[-1] for p in paths}
    intermediaries = Counter(x for p in paths for x in p[1:-1])
    per_start_paths = Counter(p[0] for p in paths)
    per_start_endpoints = defaultdict(set)
    for p in paths:
        per_start_endpoints[p[0]].add(p[-1])
    length_counts = Counter(len(p) - 1 for p in paths)
    late_paths = [p for p in paths if years[p[-1]] in (1980, 1990)]
    late_endpoints = {p[-1] for p in late_paths}
    first_lags = []
    for start in starts:
        reachable_decades = [years[p[-1]] for p in late_paths if p[0] == start]
        if reachable_decades:
            first_lags.append(min(reachable_decades) - years[start])
    return {
        "label": label,
        "starts": len(starts),
        "reachable_share": len(per_start_paths) / float(len(starts)) if starts else float("nan"),
        "paths": len(paths),
        "endpoints": len(endpoints),
        "intermediaries": len(intermediaries),
        "median_paths": float(np.median([per_start_paths.get(s, 0) for s in starts])) if starts else float("nan"),
        "median_endpoints": float(np.median([len(per_start_endpoints.get(s, set())) for s in starts])) if starts else float("nan"),
        "late_reachable_share": len({p[0] for p in late_paths}) / float(len(starts)) if starts else float("nan"),
        "late_paths": len(late_paths),
        "late_endpoints": len(late_endpoints),
        "median_first_lag": float(np.median(first_lags)) if first_lags else float("nan"),
        "iqr_first_lag": np.percentile(first_lags, [25, 75]).tolist() if first_lags else [float("nan"), float("nan")],
        "length_counts": dict(sorted(length_counts.items())),
    }


def summarize_reach(label, starts, reached_by_decade, years):
    late_decades = [1980, 1990]
    late_by_start = defaultdict(set)
    all_by_start = defaultdict(set)
    for d, per_start in reached_by_decade.items():
        for start, endpoints in per_start.items():
            all_by_start[start].update(endpoints)
            if d in late_decades:
                late_by_start[start].update(endpoints)
    first_lags = []
    for start, endpoints in late_by_start.items():
        if endpoints:
            first_lags.append(min(years[e] for e in endpoints) - years[start])
    late_endpoints = set()
    for endpoints in late_by_start.values():
        late_endpoints.update(endpoints)
    all_endpoints = set()
    for endpoints in all_by_start.values():
        all_endpoints.update(endpoints)
    return {
        "label": label,
        "starts": len(starts),
        "reachable_share_all_windows": len(all_by_start) / float(len(starts)) if starts else float("nan"),
        "unique_endpoints_all_windows": len(all_endpoints),
        "median_endpoints_all_windows": float(np.median([len(all_by_start.get(s, set())) for s in starts])) if starts else float("nan"),
        "late_reachable_share": len(late_by_start) / float(len(starts)) if starts else float("nan"),
        "late_endpoints": len(late_endpoints),
        "median_late_endpoints": float(np.median([len(late_by_start.get(s, set())) for s in starts])) if starts else float("nan"),
        "median_first_lag": float(np.median(first_lags)) if first_lags else float("nan"),
        "iqr_first_lag": np.percentile(first_lags, [25, 75]).tolist() if first_lags else [float("nan"), float("nan")],
    }


def gini(values):
    arr = np.array(values, dtype=float)
    if len(arr) == 0 or arr.sum() == 0:
        return float("nan")
    arr = np.sort(arr)
    index = np.arange(1, len(arr) + 1)
    return float(np.sum((2 * index - len(arr) - 1) * arr) / (len(arr) * arr.sum()))


def main():
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs(FIG, exist_ok=True)

    artist_rows = read_csv(os.path.join(DATA, "data_by_artist.csv"))
    influence_rows = read_csv(os.path.join(DATA, "influence_data.csv"))
    x8, vec8, artist_genre, by_genre = build_vectors(artist_rows)

    names, genre, years = {}, {}, {}
    outdegree, indegree = Counter(), Counter()
    edges, seen = [], set()
    for row in influence_rows:
        a, b = row["influencer_id"], row["follower_id"]
        if (a, b) in seen:
            continue
        seen.add((a, b))
        edges.append((a, b))
        outdegree[a] += 1
        indegree[b] += 1
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

    def time_ok(a, b):
        return a in years and b in years and years[b] >= years[a]

    pr_nodes = [i for i, g in genre.items() if g == "Pop/Rock" and i in vec8 and i in years]
    pr_edges = [
        (a, b)
        for a, b in edges
        if genre.get(a) == "Pop/Rock" and genre.get(b) == "Pop/Rock" and a in vec8 and b in vec8 and time_ok(a, b)
    ]
    qin25 = np.percentile([indegree[i] for i in pr_nodes], 25)
    qout50 = np.percentile([outdegree[i] for i in pr_nodes], 50)

    extremeness = {}
    for artist_id, vector in vec8.items():
        g = artist_genre.get(artist_id)
        if g in by_genre:
            extremeness[artist_id] = float(np.linalg.norm(vector - x8[by_genre[g]].mean(0)))
    pr_ext = [extremeness[i] for i in pr_nodes]
    q80 = np.percentile(pr_ext, 80)
    q50 = np.percentile(pr_ext, 50)

    low_degree = [
        i for i in pr_nodes if years[i] <= 1960 and indegree[i] <= qin25 and 0 < outdegree[i] <= qout50
    ]
    acoustic_subset = [i for i in low_degree if extremeness[i] >= q80]
    lower_ext_subset = [i for i in low_degree if extremeness[i] < q50]
    late = {i for i in pr_nodes if years[i] in (1980, 1990)}
    endpoint_decades = [1960, 1970, 1980, 1990, 2000]

    def make_adj(edge_list):
        adj = defaultdict(list)
        for a, b in edge_list:
            adj[a].append(b)
        return adj

    def enumerate_paths(starts, adj, target_decades=None, maxlen=4):
        paths = []
        target_decades = set(target_decades) if target_decades is not None else None
        for start in starts:
            queue = deque([(start, [start])])
            while queue:
                node, path = queue.popleft()
                if len(path) > maxlen + 1:
                    continue
                if node != start and len(path) >= 3:
                    if target_decades is None or years[node] in target_decades:
                        paths.append(path)
                if len(path) == maxlen + 1:
                    continue
                for nxt in adj[node]:
                    if nxt in path or nxt not in vec8 or genre.get(nxt) != "Pop/Rock" or nxt not in years:
                        continue
                    if years[nxt] < years[node]:
                        continue
                    queue.append((nxt, path + [nxt]))
        return paths

    real_adj = make_adj(pr_edges)
    def reachable_by_decade(starts, adj, target_decades, maxlen=4):
        reached = {d: defaultdict(set) for d in target_decades}
        for start in starts:
            queue = deque([(start, [start])])
            while queue:
                node, path = queue.popleft()
                if node != start and len(path) >= 3 and years[node] in reached:
                    reached[years[node]][start].add(node)
                if len(path) == maxlen + 1:
                    continue
                for nxt in adj[node]:
                    if nxt in path or nxt not in vec8 or genre.get(nxt) != "Pop/Rock" or nxt not in years:
                        continue
                    if years[nxt] < years[node]:
                        continue
                    queue.append((nxt, path + [nxt]))
        return reached

    real_reached_by_decade = reachable_by_decade(low_degree, real_adj, endpoint_decades)
    acoustic_reached_by_decade = reachable_by_decade(acoustic_subset, real_adj, endpoint_decades)
    lower_reached_by_decade = reachable_by_decade(lower_ext_subset, real_adj, endpoint_decades)

    def endpoint_concentration(label, starts, reached_by_decade):
        late_endpoints_by_start = defaultdict(set)
        for d in (1980, 1990):
            for start, endpoint_set in reached_by_decade[d].items():
                late_endpoints_by_start[start].update(endpoint_set)
        counts = [len(late_endpoints_by_start.get(start, set())) for start in starts]
        positive_counts = [count for count in counts if count > 0]
        total = sum(positive_counts)
        top_share = max(positive_counts) / float(total) if total else float("nan")
        hhi = sum((count / float(total)) ** 2 for count in positive_counts) if total else float("nan")
        print(
            "Late endpoint concentration",
            label,
            "starts",
            len(starts),
            "reachable_starts",
            len(positive_counts),
            "unique_start_endpoint_pairs",
            total,
            "top_start_share",
            top_share,
            "HHI",
            hhi,
            "Gini_all_starts",
            gini(counts),
            "median_positive",
            float(np.median(positive_counts)) if positive_counts else float("nan"),
        )
        return late_endpoints_by_start

    acoustic_late_by_start = endpoint_concentration(
        "acoustically distinctive subset", acoustic_subset, acoustic_reached_by_decade
    )
    lower_late_by_start = endpoint_concentration(
        "lower-extremeness subset", lower_ext_subset, lower_reached_by_decade
    )
    endpoint_concentration("early low-degree periphery", low_degree, real_reached_by_decade)
    acoustic_endpoints = set().union(*acoustic_late_by_start.values()) if acoustic_late_by_start else set()
    lower_endpoints = set().union(*lower_late_by_start.values()) if lower_late_by_start else set()
    endpoint_intersection = acoustic_endpoints & lower_endpoints
    print(
        "Late endpoint overlap acoustic_vs_lower",
        "intersection",
        len(endpoint_intersection),
        "acoustic_share",
        len(endpoint_intersection) / float(len(acoustic_endpoints)) if acoustic_endpoints else float("nan"),
        "lower_share",
        len(endpoint_intersection) / float(len(lower_endpoints)) if lower_endpoints else float("nan"),
        "jaccard",
        len(endpoint_intersection) / float(len(acoustic_endpoints | lower_endpoints))
        if acoustic_endpoints or lower_endpoints
        else float("nan"),
    )

    print("Early low-degree visibility", flush=True)
    print("Pop/Rock nodes", len(pr_nodes), "time-respecting Pop/Rock edges", len(pr_edges))
    print("Low-degree thresholds indegree<=", qin25, "0<outdegree<=", qout50)
    print("Low-degree pool", len(low_degree), "acoustic subset", len(acoustic_subset), "lower-ext subset", len(lower_ext_subset))
    for summary in [
        summarize_reach("early low-degree periphery", low_degree, real_reached_by_decade, years),
        summarize_reach("acoustically distinctive subset", acoustic_subset, acoustic_reached_by_decade, years),
        summarize_reach("lower-extremeness subset", lower_ext_subset, lower_reached_by_decade, years),
    ]:
        print(summary, flush=True)

    by_decade = {}
    for d in endpoint_decades:
        per_start_endpoints = real_reached_by_decade[d]
        unique_endpoints = set()
        for endpoint_set in per_start_endpoints.values():
            unique_endpoints.update(endpoint_set)
        by_decade[d] = {
            "unique_endpoints": len(unique_endpoints),
            "reachable_starts": len(per_start_endpoints),
            "median_endpoints_per_start": float(np.median([len(per_start_endpoints.get(s, set())) for s in low_degree])),
        }
    print("Visibility by endpoint decade:", flush=True)
    for d in endpoint_decades:
        print(d, by_decade[d], flush=True)

    observed_curve = np.array([by_decade[d]["unique_endpoints"] for d in endpoint_decades])
    observed_reachable = np.array([by_decade[d]["reachable_starts"] for d in endpoint_decades])
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(endpoint_decades, observed_curve, marker="o", color="#1f77b4", linewidth=2, label="unique endpoints")
    ax1.set_xlabel("Endpoint active-start decade")
    ax1.set_ylabel("Unique endpoints reached", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(endpoint_decades, observed_reachable, marker="s", color="#d62728", linewidth=1.8, label="reachable starts")
    ax2.set_ylabel("Reachable low-degree starts", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("Later reachability from early low-degree Pop/Rock artists")
    fig.tight_layout()
    observed_path = os.path.join(FIG, "figure_low_degree_visibility_observed.png")
    plt.savefig(observed_path, dpi=300)
    plt.close()
    print("Saved", observed_path, flush=True)

    # Directed degree-preserving temporal rewiring. Starting from the observed
    # Pop/Rock time-respecting edge list preserves exact in/out degrees. Swaps
    # are accepted only when both proposed edges remain time-respecting.
    def rewire_edges(edge_list, attempts_per_edge=8):
        current = list(edge_list)
        edge_set = set(current)
        n = len(current)
        attempts = attempts_per_edge * n
        accepted = 0
        for _ in range(attempts):
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            a, b = current[i]
            c, d = current[j]
            if len({a, b, c, d}) < 4:
                continue
            e1, e2 = (a, d), (c, b)
            if e1 in edge_set or e2 in edge_set or e1[0] == e1[1] or e2[0] == e2[1]:
                continue
            if not (time_ok(*e1) and time_ok(*e2)):
                continue
            edge_set.remove((a, b))
            edge_set.remove((c, d))
            edge_set.add(e1)
            edge_set.add(e2)
            current[i], current[j] = e1, e2
            accepted += 1
        return current, accepted

    null_rows = []
    null_curves = []
    null_iterations = int(os.environ.get("MUSIC_REWIRE_ITERATIONS", "1000"))
    rewire_attempts_per_edge = int(os.environ.get("MUSIC_REWIRE_ATTEMPTS_PER_EDGE", "3"))
    print("Null rewiring iterations", null_iterations, "attempts_per_edge", rewire_attempts_per_edge)
    for _ in range(null_iterations):
        rewired, accepted = rewire_edges(pr_edges, attempts_per_edge=rewire_attempts_per_edge)
        adj_null = make_adj(rewired)
        null_reached = reachable_by_decade(low_degree, adj_null, endpoint_decades)
        s = summarize_reach("null", low_degree, null_reached, years)
        s["accepted_swaps"] = accepted
        null_rows.append(s)
        curve = []
        for d in endpoint_decades:
            endpoints = set()
            for endpoint_set in null_reached[d].values():
                endpoints.update(endpoint_set)
            curve.append(len(endpoints))
        null_curves.append(curve)

    observed_late = summarize_reach("observed late", low_degree, real_reached_by_decade, years)
    print("Observed late-window low-degree summary:", observed_late)
    for metric in ["late_reachable_share", "late_endpoints", "median_late_endpoints", "median_first_lag"]:
        vals = np.array([row[metric] for row in null_rows], dtype=float)
        obs = observed_late[metric]
        if metric == "median_first_lag":
            p = (np.sum(vals >= obs) + 1) / float(null_iterations + 1)
        else:
            p = (np.sum(vals >= obs) + 1) / float(null_iterations + 1)
        print(
            "Null metric",
            metric,
            "observed",
            obs,
            "null median",
            float(np.nanmedian(vals)),
            "null 95%",
            np.nanpercentile(vals, [2.5, 97.5]).tolist(),
            "upper-tail p",
            p,
        )
    print("Null accepted swaps median/IQR", float(np.median([r["accepted_swaps"] for r in null_rows])), np.percentile([r["accepted_swaps"] for r in null_rows], [25, 75]).tolist())

    null_curves = np.array(null_curves)
    null_lo, null_hi = np.percentile(null_curves, [2.5, 97.5], axis=0)
    null_med = np.percentile(null_curves, 50, axis=0)

    plt.figure(figsize=(7, 4.2))
    plt.fill_between(endpoint_decades, null_lo, null_hi, color="#c9c9c9", alpha=0.65, label="rewired null 95% interval")
    plt.plot(endpoint_decades, null_med, color="#666666", linestyle="--", linewidth=1.5, label="rewired null median")
    plt.plot(endpoint_decades, observed_curve, marker="o", color="#1f77b4", linewidth=2, label="observed network")
    plt.xlabel("Endpoint active-start decade")
    plt.ylabel("Unique endpoints reached")
    plt.title("Early low-degree Pop/Rock reachability by endpoint decade")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(FIG, "figure_low_degree_visibility_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved", out_path)


if __name__ == "__main__":
    main()
