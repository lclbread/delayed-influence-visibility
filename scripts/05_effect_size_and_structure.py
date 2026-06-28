"""Effect-size, acoustic-space, network-structure, and portability diagnostics.

This script prints the scale checks used to interpret the manuscript results:
paired standardized effects for matched acoustic comparisons, Pop/Rock graph
structure summaries, feature-space dimensionality checks, robustness summaries,
a song-level cosine benchmark, and a compact cross-genre path diagnostic using
the same relative workflow.
"""

from __future__ import annotations

import ast
import csv
import math
import random
import sys
from collections import Counter, defaultdict, deque

import numpy as np

from _paths import DATA_DIR

DATA = str(DATA_DIR)
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


def read_csv(name):
    with open(f"{DATA}/{name}", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_artist_ids(value):
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return [str(parsed)]


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
    return x_norm, valid, vectors


def cosine(vectors, a, b):
    va, vb = vectors[a], vectors[b]
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def cosine_rows(x, i, j):
    va, vb = x[i], x[j]
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return float("nan")
    return float(np.dot(va, vb) / denom)


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


def paired_dz(values):
    values = np.array(values, dtype=float)
    return float(values.mean() / values.std(ddof=1))


def euclidean(vectors, a, b):
    return float(np.linalg.norm(vectors[a] - vectors[b]))


def pca_summary(x):
    centered = x - x.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular_values ** 2 / (len(x) - 1)
    ratios = eigenvalues / eigenvalues.sum()
    effective_dimension = float((eigenvalues.sum() ** 2) / np.sum(eigenvalues ** 2))
    return ratios, effective_dimension


def decade_standardized_vectors(valid_rows, x_norm, years, genre_name="Pop/Rock"):
    id_to_index = {row["artist_id"]: i for i, row in enumerate(valid_rows)}
    groups = defaultdict(list)
    for row in valid_rows:
        artist_id = row["artist_id"]
        if row.get("genre", "").strip() == genre_name and artist_id in years:
            groups[decade(years[artist_id])].append(id_to_index[artist_id])

    global_mean = x_norm.mean(axis=0)
    global_std = x_norm.std(axis=0)
    global_std[global_std == 0] = 1
    result = {}
    for row in valid_rows:
        artist_id = row["artist_id"]
        if artist_id not in years:
            continue
        group = groups.get(decade(years[artist_id]), [])
        if len(group) >= 2:
            mean = x_norm[group].mean(axis=0)
            std = x_norm[group].std(axis=0)
            std[std == 0] = 1
        else:
            mean, std = global_mean, global_std
        result[artist_id] = (x_norm[id_to_index[artist_id]] - mean) / std
    return result


def core_numbers(nodes, undirected):
    remaining = set(nodes)
    degree = {node: len(undirected[node] & remaining) for node in nodes}
    core = {}
    current_core = 0
    while remaining:
        node = min(remaining, key=lambda item: degree[item])
        current_core = max(current_core, degree[node])
        core[node] = current_core
        remaining.remove(node)
        for nbr in undirected[node] & remaining:
            degree[nbr] -= 1
    return core


def strongly_connected_components(nodes, directed):
    sys.setrecursionlimit(max(10000, len(nodes) * 2))
    index = 0
    stack = []
    on_stack = set()
    indices = {}
    lowlink = {}
    components = []

    def visit(node):
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for nxt in directed[node]:
            if nxt not in indices:
                visit(nxt)
                lowlink[node] = min(lowlink[node], lowlink[nxt])
            elif nxt in on_stack:
                lowlink[node] = min(lowlink[node], indices[nxt])
        if lowlink[node] == indices[node]:
            component = []
            while True:
                item = stack.pop()
                on_stack.remove(item)
                component.append(item)
                if item == node:
                    break
            components.append(component)

    for node in nodes:
        if node not in indices:
            visit(node)
    return components


def reachable_from(starts, adjacency):
    reached = set(starts)
    queue = deque(starts)
    while queue:
        node = queue.popleft()
        for nxt in adjacency[node]:
            if nxt not in reached:
                reached.add(nxt)
                queue.append(nxt)
    return reached


def load_network(influence_rows):
    outdegree, indegree = Counter(), Counter()
    years, genre, edges, seen = {}, {}, [], set()
    follower_series = defaultdict(lambda: defaultdict(set))
    first_observed = defaultdict(lambda: 9999)

    for row in influence_rows:
        a, b = row["influencer_id"], row["follower_id"]
        if (a, b) in seen:
            continue
        seen.add((a, b))
        edges.append((a, b, row))
        outdegree[a] += 1
        indegree[b] += 1
        for id_key, genre_key, year_key in [
            ("influencer_id", "influencer_main_genre", "influencer_active_start"),
            ("follower_id", "follower_main_genre", "follower_active_start"),
        ]:
            artist_id = row[id_key]
            g = row[genre_key].strip()
            y = ffloat(row[year_key])
            genre[artist_id] = g
            if y is not None:
                years[artist_id] = int(y)
                first_observed[g] = min(first_observed[g], int(y))
        g = row["follower_main_genre"].strip()
        y = ffloat(row["follower_active_start"])
        if g and y is not None:
            follower_series[g][decade(int(y))].add(row["follower_id"])

    return edges, outdegree, indegree, years, genre, follower_series, first_observed


def main():
    random.seed(SEED)
    artists = read_csv("data_by_artist.csv")
    influence = read_csv("influence_data.csv")
    songs = read_csv("full_music_data.csv")

    x8, valid8, vec8 = build_vectors(artists, FEATURES_8)
    _, _, vec11 = build_vectors(artists, FEATURES_11)
    edges, outdegree, indegree, years, genre, follower_series, first_observed = load_network(influence)

    def time_ok(a, b):
        return a in years and b in years and years[b] >= years[a]

    edge_set = {(a, b) for a, b, _ in edges}
    pr_nodes = [i for i, g in genre.items() if g == "Pop/Rock" and i in vec8 and i in years]
    pr_edges = [
        (a, b)
        for a, b, _ in edges
        if genre.get(a) == "Pop/Rock" and genre.get(b) == "Pop/Rock" and a in vec8 and b in vec8 and time_ok(a, b)
    ]

    id_to_index = {row["artist_id"]: i for i, row in enumerate(valid8)}
    pr_indices = [id_to_index[node] for node in pr_nodes if node in id_to_index]
    pca_ratios, effective_dimension = pca_summary(x8[pr_indices])
    print("Pop/Rock eight-feature PCA:")
    print(
        "pc1",
        float(pca_ratios[0]),
        "pc2",
        float(pca_ratios[1]),
        "pc3",
        float(pca_ratios[2]),
        "pc4",
        float(pca_ratios[3]),
        "first3",
        float(np.sum(pca_ratios[:3])),
        "first4",
        float(np.sum(pca_ratios[:4])),
        "effective_dimension",
        effective_dimension,
    )

    by_key = defaultdict(list)
    for artist_id in pr_nodes:
        by_key[(decade(years[artist_id]), degree_bin(outdegree[artist_id]), degree_bin(indegree[artist_id]))].append(
            artist_id
        )

    used_edges, matched_non_edges = [], []
    for a, b in pr_edges:
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

    edge_sims = np.array([cosine(vec8, a, b) for a, b in used_edges])
    base_sims = np.array([cosine(vec8, a, b) for a, b in matched_non_edges])
    diff = edge_sims - base_sims
    print("Matched edge effect size:")
    print(
        "n",
        len(diff),
        "edge_mean",
        float(edge_sims.mean()),
        "baseline_mean",
        float(base_sims.mean()),
        "diff",
        float(diff.mean()),
        "sd_diff",
        float(diff.std(ddof=1)),
        "paired_dz",
        paired_dz(diff),
    )

    pairs11 = [
        ((a, b), (aa, bb))
        for (a, b), (aa, bb) in zip(used_edges, matched_non_edges)
        if a in vec11 and b in vec11 and aa in vec11 and bb in vec11
    ]
    diff11 = np.array([cosine(vec11, a, b) - cosine(vec11, aa, bb) for (a, b), (aa, bb) in pairs11])
    print("Matched edge 11-feature effect size:")
    print("n", len(diff11), "diff", float(diff11.mean()), "sd_diff", float(diff11.std(ddof=1)), "paired_dz", paired_dz(diff11))

    decade_z = decade_standardized_vectors(valid8, x8, years)
    edge_dist = np.array([euclidean(decade_z, a, b) for a, b in used_edges])
    base_dist = np.array([euclidean(decade_z, a, b) for a, b in matched_non_edges])
    dist_diff = base_dist - edge_dist
    print("Decade-standardized Euclidean edge distance:")
    print(
        "n",
        len(dist_diff),
        "edge_distance",
        float(edge_dist.mean()),
        "baseline_distance",
        float(base_dist.mean()),
        "baseline_minus_edge",
        float(dist_diff.mean()),
        "sd_diff",
        float(dist_diff.std(ddof=1)),
        "paired_dz",
        paired_dz(dist_diff),
    )

    graph_nodes = sorted(set(node for edge in pr_edges for node in edge))
    undirected = defaultdict(set)
    directed = defaultdict(list)
    for a, b in pr_edges:
        directed[a].append(b)
        undirected[a].add(b)
        undirected[b].add(a)

    seen, components = set(), []
    for node in graph_nodes:
        if node in seen:
            continue
        queue = deque([node])
        seen.add(node)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for nxt in undirected[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(component)

    lcc = max(components, key=len)
    lcc_set = set(lcc)
    total_distance = distance_count = diameter = 0
    for start in lcc:
        dist = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for nxt in undirected[current]:
                if nxt in lcc_set and nxt not in dist:
                    dist[nxt] = dist[current] + 1
                    queue.append(nxt)
        for distance in dist.values():
            if distance > 0:
                total_distance += distance
                distance_count += 1
                diameter = max(diameter, distance)

    directed_total = directed_count = directed_diameter = 0
    for start in graph_nodes:
        dist = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for nxt in directed[current]:
                if nxt not in dist:
                    dist[nxt] = dist[current] + 1
                    queue.append(nxt)
        for distance in dist.values():
            if distance > 0:
                directed_total += distance
                directed_count += 1
                directed_diameter = max(directed_diameter, distance)

    print("Pop/Rock time-respecting graph structure:")
    print(
        "nodes",
        len(graph_nodes),
        "edges",
        len(pr_edges),
        "density",
        len(pr_edges) / float(len(graph_nodes) * (len(graph_nodes) - 1)),
        "median_indegree",
        float(np.median([indegree[n] for n in graph_nodes])),
        "median_outdegree",
        float(np.median([outdegree[n] for n in graph_nodes])),
        "mean_degree",
        len(pr_edges) / float(len(graph_nodes)),
        "max_outdegree",
        max(outdegree[n] for n in graph_nodes),
        "weak_components",
        len(components),
        "largest_weak_component",
        len(lcc),
        "largest_weak_component_share",
        len(lcc) / float(len(graph_nodes)),
        "undirected_lcc_average_path",
        total_distance / float(distance_count),
        "undirected_lcc_diameter",
        diameter,
        "directed_reachable_pairs",
        directed_count,
        "directed_average_shortest_path",
        directed_total / float(directed_count),
        "directed_diameter",
        directed_diameter,
    )

    core = core_numbers(graph_nodes, defaultdict(set, {node: set(undirected[node]) for node in graph_nodes}))
    qin25 = np.percentile([indegree[i] for i in pr_nodes], 25)
    qout50 = np.percentile([outdegree[i] for i in pr_nodes], 50)
    low_degree = [i for i in pr_nodes if years[i] <= 1960 and indegree[i] <= qin25 and 0 < outdegree[i] <= qout50]
    low_core = [core[i] for i in low_degree if i in core]
    print("Pop/Rock k-core diagnostics:")
    print(
        "max_core",
        max(core.values()),
        "median_core_all_nodes",
        float(np.median(list(core.values()))),
        "low_degree_starts",
        len(low_degree),
        "median_core_low_degree",
        float(np.median(low_core)),
        "low_degree_core_iqr",
        np.percentile(low_core, [25, 75]).tolist(),
    )

    reverse_directed = defaultdict(list)
    for a, b in pr_edges:
        reverse_directed[b].append(a)
    sccs = strongly_connected_components(graph_nodes, directed)
    giant_scc = set(max(sccs, key=len))
    from_scc = reachable_from(giant_scc, directed)
    to_scc = reachable_from(giant_scc, reverse_directed)
    bow_in = to_scc - giant_scc
    bow_out = from_scc - giant_scc
    bow_other = set(graph_nodes) - giant_scc - bow_in - bow_out
    print("Pop/Rock bow-tie diagnostics:")
    print(
        "scc_count",
        len(sccs),
        "largest_scc",
        len(giant_scc),
        "largest_scc_share",
        len(giant_scc) / float(len(graph_nodes)),
        "in",
        len(bow_in),
        "out",
        len(bow_out),
        "other",
        len(bow_other),
    )

    names = {}
    for _, _, row in edges:
        names[row["influencer_id"]] = row.get("influencer_name", row["influencer_id"])
        names[row["follower_id"]] = row.get("follower_name", row["follower_id"])

    def reach_after_removal(removed):
        removed = set(removed)
        reached_by_start = defaultdict(set)
        path_count = 0
        for start in low_degree:
            if start in removed:
                continue
            queue = deque([(start, [start])])
            while queue:
                node, path = queue.popleft()
                if node != start and len(path) >= 3 and years[node] in (1980, 1990):
                    reached_by_start[start].add(node)
                    path_count += 1
                    continue
                if len(path) == 5:
                    continue
                for nxt in directed[node]:
                    if (
                        nxt in removed
                        or nxt in path
                        or nxt not in vec8
                        or genre.get(nxt) != "Pop/Rock"
                        or nxt not in years
                        or years[nxt] < years[node]
                    ):
                        continue
                    queue.append((nxt, path + [nxt]))
        endpoints = set()
        for endpoint_set in reached_by_start.values():
            endpoints.update(endpoint_set)
        return len(reached_by_start), len(endpoints), path_count

    hubs_1960 = [i for i in pr_nodes if years[i] == 1960 and outdegree[i] >= 50 and indegree[i] > 0]
    top_outdegree_hubs = sorted(hubs_1960, key=lambda item: outdegree[item], reverse=True)
    beatles = next((node for node in graph_nodes if names.get(node) == "The Beatles"), None)
    removal_sets = [
        ("none", []),
        ("Beatles", [beatles] if beatles else []),
        ("top5_1960_outdegree_hubs", top_outdegree_hubs[:5]),
        ("top10_1960_outdegree_hubs", top_outdegree_hubs[:10]),
        ("all_1960_outdegree_hubs", hubs_1960),
    ]
    print("Hub-removal reachability robustness:")
    for label, removed in removal_sets:
        reachable, endpoints, path_count = reach_after_removal(removed)
        removed_names = [names.get(node, node) for node in removed[:5]]
        print(
            label,
            "removed",
            len(removed),
            "sample_removed",
            removed_names,
            "reachable_starts",
            reachable,
            "reachable_share",
            reachable / float(len(low_degree)),
            "late_endpoints",
            endpoints,
            "paths",
            path_count,
        )

    song_rows, raw = [], []
    for row in songs:
        if row.get("genre", "").strip() != "Pop/Rock":
            continue
        y = ffloat(row.get("year"))
        artist_ids = parse_artist_ids(row.get("artists_id", ""))
        values = [ffloat(row.get(feature)) for feature in FEATURES_8]
        if y is None or not artist_ids or not all(v is not None and math.isfinite(v) for v in values):
            continue
        row["_year"] = int(y)
        row["_artist_ids"] = artist_ids
        song_rows.append(row)
        raw.append(values)

    x_raw = np.array(raw, dtype=float)
    x_cap = np.clip(x_raw, x_raw.mean(0) - 3 * x_raw.std(0), x_raw.mean(0) + 3 * x_raw.std(0))
    denom = x_cap.max(0) - x_cap.min(0)
    denom[denom == 0] = 1
    x_song = (x_cap - x_cap.min(0)) / denom

    songs_by_artist = defaultdict(list)
    songs_by_decade = defaultdict(list)
    for i, row in enumerate(song_rows):
        songs_by_decade[decade(row["_year"])].append(i)
        for artist_id in row["_artist_ids"]:
            songs_by_artist[artist_id].append(i)

    within_artist, same_decade_other = [], []
    artists_with_multiple_songs = [a for a, rows in songs_by_artist.items() if len(rows) >= 2]
    for _ in range(20000):
        artist_id = random.choice(artists_with_multiple_songs)
        i, j = random.sample(songs_by_artist[artist_id], 2)
        own_ids = set(song_rows[i]["_artist_ids"])
        match = None
        for _ in range(100):
            candidate = random.choice(songs_by_decade[decade(song_rows[i]["_year"])])
            if not (own_ids & set(song_rows[candidate]["_artist_ids"])):
                match = candidate
                break
        if match is not None:
            within_value = cosine_rows(x_song, i, j)
            other_value = cosine_rows(x_song, i, match)
            if math.isfinite(within_value) and math.isfinite(other_value):
                within_artist.append(within_value)
                same_decade_other.append(other_value)

    song_diff = np.array(within_artist) - np.array(same_decade_other)
    print("Song-level Pop/Rock benchmark:")
    print(
        "samples",
        len(song_diff),
        "within_artist_mean",
        float(np.mean(within_artist)),
        "same_decade_other_artist_mean",
        float(np.mean(same_decade_other)),
        "diff",
        float(song_diff.mean()),
        "sd_diff",
        float(song_diff.std(ddof=1)),
        "paired_dz",
        paired_dz(song_diff),
    )

    print("Compact cross-genre path diagnostic:")
    print("genre nodes edges first peak early_cut endpoint_window starts reachable_share endpoints")
    for g in ["Pop/Rock", "R&B;", "Jazz", "Country", "Electronic", "Reggae"]:
        nodes = [i for i in genre if genre.get(i) == g and i in vec8 and i in years]
        genre_edges = [
            (a, b)
            for a, b, _ in edges
            if genre.get(a) == g and genre.get(b) == g and a in vec8 and b in vec8 and time_ok(a, b)
        ]
        if len(nodes) < 20 or not genre_edges or not follower_series[g]:
            continue
        adj = defaultdict(list)
        for a, b in genre_edges:
            adj[a].append(b)
        counts = {d: len(s) for d, s in follower_series[g].items()}
        peak = max(counts, key=counts.get)
        endpoint_window = {peak - 10, peak}
        early_cut = first_observed[g] + 30
        qin = np.percentile([indegree[i] for i in nodes], 25)
        qout = np.percentile([outdegree[i] for i in nodes], 50)
        starts = [i for i in nodes if years[i] <= early_cut and indegree[i] <= qin and 0 < outdegree[i] <= qout]
        reached = defaultdict(set)
        for start in starts:
            queue = deque([(start, [start])])
            while queue:
                current, path = queue.popleft()
                if current != start and len(path) >= 3 and decade(years[current]) in endpoint_window:
                    reached[start].add(current)
                if len(path) == 5:
                    continue
                for nxt in adj[current]:
                    if nxt in path or nxt not in vec8 or genre.get(nxt) != g or nxt not in years:
                        continue
                    if years[nxt] < years[current]:
                        continue
                    queue.append((nxt, path + [nxt]))
        endpoints = set(endpoint for endpoint_set in reached.values() for endpoint in endpoint_set)
        reachable_share = sum(1 for start in starts if reached[start]) / float(len(starts)) if starts else float("nan")
        print(g, len(nodes), len(genre_edges), first_observed[g], peak, early_cut, sorted(endpoint_window), len(starts), reachable_share, len(endpoints))


if __name__ == "__main__":
    main()
