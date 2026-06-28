"""Recompute path-level, hub-mediation, and song-level descriptor checks.

The script recomputes targeted checks from the raw CSV files:
1. path-level acoustic continuity for early low-degree-to-late paths,
2. high-outdegree hub mediation share across the full enumerated path set,
3. song-level descriptor profiles of reachable late endpoints.
"""

from __future__ import annotations

import ast
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
REPERTOIRE_FEATURES = ["danceability", "energy", "acousticness", "loudness"]


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


def cosine(vectors, a, b):
    va, vb = vectors[a], vectors[b]
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def parse_artist_ids(value):
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return [str(parsed)]


def bootstrap_diff(a, b, rng, n=1000):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    diffs = []
    for _ in range(n):
        diffs.append(float(rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()))
    return np.percentile(diffs, [2.5, 97.5]).tolist()


def bootstrap_mean(values, rng, n=1000):
    values = np.array(values, dtype=float)
    means = []
    for _ in range(n):
        means.append(float(rng.choice(values, len(values), replace=True).mean()))
    return np.percentile(means, [2.5, 97.5]).tolist()


def gini(values):
    arr = np.array(values, dtype=float)
    if len(arr) == 0 or arr.sum() == 0:
        return float("nan")
    arr = np.sort(arr)
    index = np.arange(1, len(arr) + 1)
    return float((np.sum((2 * index - len(arr) - 1) * arr)) / (len(arr) * arr.sum()))


def main():
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs(FIG, exist_ok=True)

    artist_rows = read_csv(os.path.join(DATA, "data_by_artist.csv"))
    influence_rows = read_csv(os.path.join(DATA, "influence_data.csv"))
    song_rows = read_csv(os.path.join(DATA, "full_music_data.csv"))
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
    edge_set = set(pr_edges)
    adj = defaultdict(list)
    for a, b in pr_edges:
        adj[a].append(b)

    qin25 = np.percentile([indegree[i] for i in pr_nodes], 25)
    qout50 = np.percentile([outdegree[i] for i in pr_nodes], 50)
    low_degree = [i for i in pr_nodes if years[i] <= 1960 and indegree[i] <= qin25 and 0 < outdegree[i] <= qout50]
    pr_centroid = np.mean([vec8[i] for i in pr_nodes], axis=0)
    extremeness = {i: float(np.linalg.norm(vec8[i] - pr_centroid)) for i in pr_nodes}
    q80 = np.percentile([extremeness[i] for i in pr_nodes], 80)
    q50 = np.percentile([extremeness[i] for i in pr_nodes], 50)
    acoustic_subset = [i for i in low_degree if extremeness[i] >= q80]
    lower_ext_subset = [i for i in low_degree if extremeness[i] < q50]
    late_decades = {1980, 1990}

    def enumerate_paths(starts, maxlen=4, strict_year=False):
        paths = []
        for start in starts:
            queue = deque([(start, [start])])
            while queue:
                node, path = queue.popleft()
                if node != start and len(path) >= 3 and years[node] in late_decades:
                    paths.append(path)
                    continue
                if len(path) == maxlen + 1:
                    continue
                for nxt in adj[node]:
                    if nxt in path or nxt not in vec8 or genre.get(nxt) != "Pop/Rock" or nxt not in years:
                        continue
                    if strict_year and years[nxt] <= years[node]:
                        continue
                    if not strict_year and years[nxt] < years[node]:
                        continue
                    queue.append((nxt, path + [nxt]))
        return paths

    paths = enumerate_paths(low_degree)
    acoustic_paths = enumerate_paths(acoustic_subset)
    lower_ext_paths = enumerate_paths(lower_ext_subset)
    strict_paths = enumerate_paths(low_degree, strict_year=True)
    late_endpoints = {p[-1] for p in paths}
    print("Low-degree-to-late paths:", "low_degree", len(low_degree), "paths", len(paths))
    print(
        "Strictly increasing-year paths:",
        "starts",
        len(low_degree),
        "reachable",
        len({p[0] for p in strict_paths}),
        "share",
        len({p[0] for p in strict_paths}) / float(len(low_degree)),
        "paths",
        len(strict_paths),
        "late endpoints",
        len({p[-1] for p in strict_paths}),
    )

    def path_metrics(path):
        edge_vals = [cosine(vec8, path[i], path[i + 1]) for i in range(len(path) - 1)]
        return float(np.mean(edge_vals)), float(np.min(edge_vals)), cosine(vec8, path[0], path[-1])

    observed_metrics = np.array([path_metrics(path) for path in paths])

    nodes_by_key = defaultdict(list)
    for node in pr_nodes:
        nodes_by_key[(decade(years[node]), degree_bin(outdegree[node]), degree_bin(indegree[node]))].append(node)

    matched_metrics = []
    obs_matched = []
    matched_paths = []
    matched_count = 0
    for path in paths:
        pseudo = None
        keys = [(decade(years[n]), degree_bin(outdegree[n]), degree_bin(indegree[n])) for n in path]
        for _ in range(300):
            candidate = []
            ok = True
            for key in keys:
                pool = nodes_by_key.get(key, [])
                if not pool:
                    ok = False
                    break
                candidate.append(random.choice(pool))
            if not ok or len(set(candidate)) != len(candidate):
                continue
            bad_edge = False
            for i in range(len(candidate) - 1):
                if (candidate[i], candidate[i + 1]) in edge_set or years[candidate[i + 1]] < years[candidate[i]]:
                    bad_edge = True
                    break
            if not bad_edge:
                pseudo = candidate
                break
        if pseudo is not None:
            matched_count += 1
            matched_metrics.append(path_metrics(pseudo))
            obs_matched.append(path_metrics(path))
            matched_paths.append(path)
    matched_metrics = np.array(matched_metrics)
    obs_matched = np.array(obs_matched)

    print("Path-level matched pseudo-paths:", matched_count, "of", len(paths))
    metric_names = ["mean_edge_cosine", "min_edge_cosine", "start_endpoint_cosine"]
    for i, metric in enumerate(metric_names):
        diff = obs_matched[:, i] - matched_metrics[:, i]
        sign_perm = []
        for _ in range(3000):
            sign_perm.append(float((diff * rng.choice([-1, 1], len(diff))).mean()))
        p_value = (np.sum(np.abs(sign_perm) >= abs(diff.mean())) + 1) / 3001
        print(
            "Path metric",
            metric,
            "observed",
            float(obs_matched[:, i].mean()),
            "baseline",
            float(matched_metrics[:, i].mean()),
            "diff",
            float(diff.mean()),
            "sd_diff",
            float(diff.std(ddof=1)),
            "paired_dz",
            float(diff.mean() / diff.std(ddof=1)),
            "p",
            p_value,
        )

    print("Path metric grouped by start node:")
    for i, metric in enumerate(metric_names):
        by_start = defaultdict(list)
        for path, obs_row, pseudo_row in zip(matched_paths, obs_matched, matched_metrics):
            by_start[path[0]].append(float(obs_row[i] - pseudo_row[i]))
        grouped = np.array([np.mean(values) for values in by_start.values()], dtype=float)
        print(
            "Grouped path metric",
            metric,
            "starts",
            len(grouped),
            "mean_diff",
            float(grouped.mean()),
            "mean_diff_CI",
            bootstrap_mean(grouped, rng, n=1000),
            "median_diff",
            float(np.median(grouped)),
            "positive_start_share",
            float(np.mean(grouped > 0)),
        )

    def compact_reachability(starts, endpoint_decades, maxlen=4):
        reached_by_start = defaultdict(set)
        for start in starts:
            queue = deque([(start, [start])])
            while queue:
                node, path = queue.popleft()
                if node != start and len(path) >= 3 and years[node] in endpoint_decades:
                    reached_by_start[start].add(node)
                    continue
                if len(path) == maxlen + 1:
                    continue
                for nxt in adj[node]:
                    if nxt in path or nxt not in vec8 or genre.get(nxt) != "Pop/Rock" or nxt not in years:
                        continue
                    if years[nxt] < years[node]:
                        continue
                    queue.append((nxt, path + [nxt]))
        endpoints = set()
        for endpoint_set in reached_by_start.values():
            endpoints.update(endpoint_set)
        return len(reached_by_start) / float(len(starts)) if starts else float("nan"), len(endpoints)

    print("Compact threshold grid:")
    for active_cutoff, outdegree_max, maxlen, endpoint_decades in [
        (1950, 2, 4, {1980, 1990}),
        (1960, 1, 4, {1980, 1990}),
        (1960, 2, 3, {1980, 1990}),
        (1960, 2, 4, {1970, 1980}),
        (1960, 2, 4, {1980, 1990}),
        (1960, 3, 4, {1980, 1990}),
        (1970, 2, 4, {1980, 1990}),
    ]:
        starts = [
            i
            for i in pr_nodes
            if years[i] <= active_cutoff and indegree[i] <= qin25 and 0 < outdegree[i] <= outdegree_max
        ]
        share, endpoint_count = compact_reachability(starts, endpoint_decades, maxlen=maxlen)
        print(
            "Grid",
            "active<=",
            active_cutoff,
            "out<=",
            outdegree_max,
            "maxlen",
            maxlen,
            "window",
            sorted(endpoint_decades),
            "starts",
            len(starts),
            "reachable_share",
            share,
            "endpoints",
            endpoint_count,
        )

    hubs = [i for i in pr_nodes if years[i] == 1960 and outdegree[i] >= 50 and indegree[i] > 0]
    hub_path_counts = Counter()
    for path in paths:
        for node in set(path[1:-1]):
            if node in hubs:
                hub_path_counts[node] += 1
    beatles = next((i for i, n in names.items() if n == "The Beatles"), None)
    print("Hub mediation:", "hubs", len(hubs), "paths", len(paths), "paths with any hub", sum(1 for p in paths if any(n in hubs for n in p[1:-1])))
    print("Beatles mediated paths", hub_path_counts.get(beatles, 0), "share", hub_path_counts.get(beatles, 0) / float(len(paths)) if paths else 0)
    print("Top hub mediation rows:")
    for node, count in hub_path_counts.most_common(10):
        print(names[node], "out", outdegree[node], "in", indegree[node], "paths", count, "share", count / float(len(paths)))

    # Song-level descriptor profiles for reachable late endpoints.
    reachable_song_values = defaultdict(list)
    comparison_song_values = defaultdict(list)
    reachable_artist_song_values = defaultdict(lambda: defaultdict(list))
    comparison_artist_song_values = defaultdict(lambda: defaultdict(list))
    reachable_song_count = 0
    comparison_song_count = 0
    for row in song_rows:
        year = ffloat(row.get("year"))
        if year is None:
            continue
        year = int(year)
        if year < 1980 or year > 1999:
            continue
        if row.get("genre", "").strip() != "Pop/Rock":
            continue
        artist_ids = parse_artist_ids(row.get("artists_id", ""))
        is_reachable = any(artist_id in late_endpoints for artist_id in artist_ids)
        target = reachable_song_values if is_reachable else comparison_song_values
        has_all = True
        vals = {}
        for feature in REPERTOIRE_FEATURES:
            val = ffloat(row.get(feature))
            if val is None or not math.isfinite(val):
                has_all = False
                break
            vals[feature] = val
        if not has_all:
            continue
        if is_reachable:
            reachable_song_count += 1
        else:
            comparison_song_count += 1
        for feature, val in vals.items():
            target[feature].append(val)
        for artist_id in artist_ids:
            artist_target = reachable_artist_song_values if artist_id in late_endpoints else comparison_artist_song_values
            for feature, val in vals.items():
                artist_target[artist_id][feature].append(val)

    print("Repertoire songs:", "reachable", reachable_song_count, "comparison", comparison_song_count)
    repertoire_rows = []
    for feature in REPERTOIRE_FEATURES:
        a = reachable_song_values[feature]
        b = comparison_song_values[feature]
        ci = bootstrap_diff(a, b, rng, n=1000)
        diff = float(np.mean(a) - np.mean(b))
        pooled_var = ((len(a) - 1) * float(np.var(a, ddof=1)) + (len(b) - 1) * float(np.var(b, ddof=1))) / (
            len(a) + len(b) - 2
        )
        std_diff = diff / math.sqrt(pooled_var) if pooled_var > 0 else float("nan")
        repertoire_rows.append((feature, float(np.mean(a)), float(np.mean(b)), diff, ci, std_diff))
        print(
            "Repertoire",
            feature,
            "reachable",
            np.mean(a),
            "comparison",
            np.mean(b),
            "diff",
            diff,
            "std_diff",
            std_diff,
            "CI",
            ci,
        )

    print(
        "Repertoire artists:",
        "reachable",
        len(reachable_artist_song_values),
        "comparison",
        len(comparison_artist_song_values),
    )
    for feature in REPERTOIRE_FEATURES:
        a = [float(np.mean(values[feature])) for values in reachable_artist_song_values.values() if values[feature]]
        b = [float(np.mean(values[feature])) for values in comparison_artist_song_values.values() if values[feature]]
        ci = bootstrap_diff(a, b, rng, n=1000)
        print(
            "Artist-level descriptor profile",
            feature,
            "reachable",
            np.mean(a),
            "comparison",
            np.mean(b),
            "diff",
            np.mean(a) - np.mean(b),
            "CI",
            ci,
        )

    x = np.arange(len(REPERTOIRE_FEATURES))
    std_diffs = [row[5] for row in repertoire_rows]
    colors = ["#1f77b4" if value >= 0 else "#8c8c8c" for value in std_diffs]
    plt.figure(figsize=(7.2, 4.2))
    plt.axhline(0, color="#444444", linewidth=0.8)
    plt.bar(x, std_diffs, color=colors, width=0.56)
    plt.xticks(x, REPERTOIRE_FEATURES)
    plt.ylabel("Standardized mean difference\n(reachable - comparison)")
    plt.title("1980s/1990s descriptor profile of reachable late endpoints")
    plt.tight_layout()
    fig_path = os.path.join(FIG, "figure_reachable_endpoint_repertoire.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved", fig_path)


if __name__ == "__main__":
    main()
