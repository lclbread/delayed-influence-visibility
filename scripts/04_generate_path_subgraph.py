"""Generate an illustrative Pop/Rock path subgraph for the manuscript.

The figure uses only observed, time-respecting Pop/Rock influence edges from
the ICM 2021 data. It is not used as an additional statistical test; it gives a
readable visual example of the path population analysed in the paper.
"""

from __future__ import annotations

import csv
import math
import os
import textwrap
from collections import Counter

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from _paths import DATA_DIR, FIGURE_DIR

DATA = str(DATA_DIR)
FIG = str(FIGURE_DIR)

SELECTED_PATHS = [
    ["The Chantays", "Rick Springfield", "Robbie Williams"],
    ["The Champs", "The Ventures", "Southern Culture on the Skids"],
    ["Joey Dee", "Ramones", "Soundgarden"],
    ["Mick Taylor", "Tom Verlaine", "The Shins"],
    ["The Champs", "The Trashmen", "The Stooges", "Pearl Jam"],
    ["The Champs", "The Trashmen", "The Stooges", "David Bowie", "Smashing Pumpkins"],
    ["Bill Justis", "Duane Eddy", "The Band", "The Beatles", "Tears for Fears"],
    ["Bill Justis", "Duane Eddy", "Johnny Kidd & the Pirates", "The Rolling Stones", "Pearl Jam"],
    ["Joey Dee", "Ramones", "Sleater-Kinney"],
    ["The Champs", "The Trashmen", "The Cramps", "The Flaming Lips"],
]

POSITIONS = {
    "Bill Justis": (0.4, 5.55),
    "The Champs": (0.4, 4.45),
    "Joey Dee": (0.4, 3.1),
    "The Chantays": (0.4, 1.85),
    "Mick Taylor": (0.4, 0.75),
    "Duane Eddy": (2.15, 5.55),
    "The Ventures": (2.15, 4.95),
    "The Trashmen": (2.15, 4.05),
    "Ramones": (2.15, 3.1),
    "Rick Springfield": (2.15, 1.85),
    "Tom Verlaine": (2.15, 0.75),
    "The Band": (3.95, 5.82),
    "Johnny Kidd & the Pirates": (3.95, 5.02),
    "The Stooges": (3.95, 4.05),
    "The Cramps": (3.95, 3.25),
    "The Beatles": (5.75, 5.82),
    "The Rolling Stones": (5.75, 5.02),
    "David Bowie": (5.75, 4.05),
    "Tears for Fears": (7.7, 5.82),
    "Southern Culture on the Skids": (7.7, 5.35),
    "Pearl Jam": (7.7, 4.65),
    "Smashing Pumpkins": (7.7, 3.95),
    "The Flaming Lips": (7.7, 3.25),
    "Sleater-Kinney": (7.7, 2.55),
    "Soundgarden": (7.7, 1.85),
    "Robbie Williams": (7.7, 1.15),
    "The Shins": (7.7, 0.45),
}

ROLE_COLORS = {
    "start": "#177E89",
    "intermediary": "#5B7FA6",
    "hub": "#C45746",
    "endpoint": "#D9A441",
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    os.makedirs(FIG, exist_ok=True)
    rows = read_csv(os.path.join(DATA, "influence_data.csv"))

    name_to_id = {}
    names = {}
    years = {}
    genres = {}
    edge_set = set()
    outdegree = Counter()
    indegree = Counter()

    for row in rows:
        a, b = row["influencer_id"], row["follower_id"]
        if (a, b) not in edge_set:
            edge_set.add((a, b))
            outdegree[a] += 1
            indegree[b] += 1
        for id_key, name_key, genre_key, year_key in [
            ("influencer_id", "influencer_name", "influencer_main_genre", "influencer_active_start"),
            ("follower_id", "follower_name", "follower_main_genre", "follower_active_start"),
        ]:
            artist_id = row[id_key]
            name = row[name_key]
            name_to_id[name.lower()] = artist_id
            names[artist_id] = name
            genres[artist_id] = row[genre_key].strip()
            years[artist_id] = int(float(row[year_key]))

    selected_edges = []
    for path in SELECTED_PATHS:
        ids = [name_to_id[name.lower()] for name in path]
        for a, b in zip(ids, ids[1:]):
            if (a, b) not in edge_set:
                raise RuntimeError("Selected path contains a missing observed edge: %s -> %s" % (names[a], names[b]))
            if genres[a] != "Pop/Rock" or genres[b] != "Pop/Rock":
                raise RuntimeError("Selected path is not wholly Pop/Rock: %s -> %s" % (names[a], names[b]))
            if years[b] < years[a]:
                raise RuntimeError("Selected path is not time-respecting: %s -> %s" % (names[a], names[b]))
            selected_edges.append((a, b))

    start_names = {path[0] for path in SELECTED_PATHS}
    endpoint_names = {path[-1] for path in SELECTED_PATHS}
    selected_names = set(POSITIONS)

    roles = {}
    for name in selected_names:
        artist_id = name_to_id[name.lower()]
        if name in start_names:
            roles[name] = "start"
        elif name in endpoint_names:
            roles[name] = "endpoint"
        elif outdegree[artist_id] >= 50 and indegree[artist_id] > 0:
            roles[name] = "hub"
        else:
            roles[name] = "intermediary"

    fig, ax = plt.subplots(figsize=(11.6, 7.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(-0.55, 8.55)
    ax.set_ylim(-0.25, 6.65)
    ax.axis("off")

    bands = [
        (-0.35, 1.15, "Early low-degree\npath origins"),
        (1.55, 2.75, "Observed\nintermediaries"),
        (3.45, 6.15, "High-outdegree\nmediators"),
        (7.05, 8.45, "1980s/1990s\nendpoints"),
    ]
    for x0, x1, label in bands:
        ax.axvspan(x0, x1, color="#F5F6F7", zorder=0)
        ax.text((x0 + x1) / 2, 6.42, label, ha="center", va="top", fontsize=11, color="#3D4248")

    edge_counter = Counter(selected_edges)
    unique_edges = list(dict.fromkeys(selected_edges))
    for idx, (a, b) in enumerate(unique_edges):
        aname, bname = names[a], names[b]
        start = POSITIONS[aname]
        end = POSITIONS[bname]
        dy = end[1] - start[1]
        rad = 0.07 if dy >= 0 else -0.07
        if edge_counter[(a, b)] > 1:
            rad *= 1.8
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.15,
            color="#68717A",
            alpha=0.56,
            connectionstyle="arc3,rad=%s" % rad,
            shrinkA=12,
            shrinkB=12,
            zorder=1,
        )
        ax.add_patch(arrow)

    for name, (x, y) in POSITIONS.items():
        artist_id = name_to_id[name.lower()]
        role = roles[name]
        size = 105 + 88 * math.log1p(outdegree[artist_id])
        if role == "start":
            size *= 0.78
        elif role == "endpoint":
            size *= 0.84
        ax.scatter(
            [x],
            [y],
            s=size,
            color=ROLE_COLORS[role],
            edgecolor="white",
            linewidth=1.35,
            zorder=3,
        )

    for name, (x, y) in POSITIONS.items():
        artist_id = name_to_id[name.lower()]
        role = roles[name]
        wrapped = textwrap.fill(name, width=13)
        if role == "hub":
            label = "%s\n%s, d+=%d" % (wrapped, years[artist_id], outdegree[artist_id])
        else:
            label = "%s\n%s" % (wrapped, years[artist_id])
        if role == "start":
            ha, dx = "right", -0.18
        elif role == "endpoint":
            ha, dx = "left", 0.18
        else:
            ha, dx = "center", 0.0
        text = ax.text(
            x + dx,
            y + (0.18 if role in {"hub", "intermediary"} else 0.0),
            label,
            ha=ha,
            va="center",
            fontsize=9.0,
            color="#20252A",
            zorder=4,
            linespacing=1.05,
        )
        text.set_path_effects([path_effects.withStroke(linewidth=3.6, foreground="white")])

    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ROLE_COLORS["start"],
               markeredgecolor="white", markersize=9, label="Early low-degree start"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ROLE_COLORS["intermediary"],
               markeredgecolor="white", markersize=9, label="Observed intermediary"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ROLE_COLORS["hub"],
               markeredgecolor="white", markersize=9, label="High-outdegree mediator"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ROLE_COLORS["endpoint"],
               markeredgecolor="white", markersize=9, label="Late endpoint"),
        Line2D([0], [0], color="#68717A", lw=1.2, label="Observed directed edge"),
    ]
    ax.legend(handles=legend_items, loc="lower left", bbox_to_anchor=(0.0, -0.02),
              ncol=3, frameon=False, fontsize=9, columnspacing=1.0, handletextpad=0.4)

    ax.text(
        8.35,
        -0.03,
        "Node size scales with log(1 + outdegree). Edges are observed and time-respecting.",
        ha="right",
        va="bottom",
        fontsize=8.6,
        color="#50575F",
    )

    output = os.path.join(FIG, "figure_path_subgraph.png")
    plt.tight_layout(pad=0.4)
    plt.savefig(output, dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
