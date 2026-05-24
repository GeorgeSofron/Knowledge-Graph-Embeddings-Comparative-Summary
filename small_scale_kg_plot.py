"""Generate a small-scale, presentation-friendly plot of the knowledge graph.

The script samples a local neighborhood around a seed drug and renders a
compact graph with typed node colors and relation labels.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


DEFAULT_INPUT = Path("drugbank_facts.txt")
DEFAULT_OUTPUT = Path("outputs_small_kg/small_knowledge_graph_plot.png")


def read_triples(path: Path) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            triples.append((h, r, t))
    return triples


def infer_node_type(node: str) -> str:
    if node.startswith("DB"):
        return "Drug"
    if node.startswith("P") or node.startswith("Q"):
        return "Protein"
    if node.startswith("SMP"):
        return "Pathway"
    if node.startswith("ATC"):
        return "ATC"
    if node.startswith("D"):
        return "Category"
    return "Other"


def pick_seed_drug(triples: list[tuple[str, str, str]], preferred_seed: str | None) -> str:
    if preferred_seed:
        return preferred_seed

    drugs = sorted({h for h, _, _ in triples if h.startswith("DB")})
    if "DB00001" in drugs:
        return "DB00001"
    if drugs:
        return drugs[0]

    return triples[0][0]


def sample_subgraph(
    triples: list[tuple[str, str, str]],
    seed: str,
    max_edges: int,
    max_neighbors_per_node: int,
    rng_seed: int,
) -> nx.DiGraph:
    rng = random.Random(rng_seed)
    outgoing: dict[str, list[tuple[str, str]]] = {}
    incoming: dict[str, list[tuple[str, str]]] = {}

    for h, r, t in triples:
        outgoing.setdefault(h, []).append((r, t))
        incoming.setdefault(t, []).append((r, h))

    g = nx.DiGraph()
    frontier = [seed]
    seen_nodes = {seed}

    while frontier and g.number_of_edges() < max_edges:
        current = frontier.pop(0)
        out_edges = outgoing.get(current, [])
        in_edges = incoming.get(current, [])

        rng.shuffle(out_edges)
        rng.shuffle(in_edges)

        sampled = out_edges[:max_neighbors_per_node] + in_edges[: max_neighbors_per_node // 2]
        for rel, nb in sampled:
            if len(out_edges) and (rel, nb) in out_edges:
                src, dst = current, nb
            else:
                src, dst = nb, current

            g.add_edge(src, dst, relation=rel)
            g.nodes[src]["kind"] = infer_node_type(src)
            g.nodes[dst]["kind"] = infer_node_type(dst)

            if src not in seen_nodes:
                seen_nodes.add(src)
                frontier.append(src)
            if dst not in seen_nodes:
                seen_nodes.add(dst)
                frontier.append(dst)

            if g.number_of_edges() >= max_edges:
                break

    return g


def draw_graph(g: nx.DiGraph, seed: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    color_map = {
        "Drug": "#d62728",
        "Protein": "#1f77b4",
        "Pathway": "#2ca02c",
        "ATC": "#9467bd",
        "Category": "#ff7f0e",
        "Other": "#7f7f7f",
    }

    pos = nx.spring_layout(g, seed=42, k=0.8)
    node_colors = [color_map.get(g.nodes[n].get("kind", "Other"), "#7f7f7f") for n in g.nodes()]

    plt.figure(figsize=(13, 9))
    nx.draw_networkx_edges(
        g,
        pos,
        edge_color="#888888",
        arrows=True,
        arrowsize=16,
        width=1.1,
        alpha=0.65,
        connectionstyle="arc3,rad=0.08",
    )

    nx.draw_networkx_nodes(
        g,
        pos,
        node_color=node_colors,
        node_size=520,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.95,
    )

    labels = {n: n if n == seed or g.nodes[n].get("kind") in {"Drug", "Protein", "Pathway"} else "" for n in g.nodes()}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8)

    edge_labels = {(u, v): d.get("relation", "") for u, v, d in g.edges(data=True)}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=6, rotate=False)

    for kind, color in color_map.items():
        plt.scatter([], [], c=color, label=kind, s=90)

    plt.title(f"Small-Scale DrugBank Knowledge Graph Sample (seed: {seed})", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small-scale KG plot from drugbank_facts.txt")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input TSV triple file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path")
    parser.add_argument("--seed-drug", type=str, default=None, help="Optional seed drug ID (e.g., DB00001)")
    parser.add_argument("--max-edges", type=int, default=42, help="Maximum edges in sampled subgraph")
    parser.add_argument("--max-neighbors", type=int, default=6, help="Max sampled neighbors per frontier node")
    parser.add_argument("--rng-seed", type=int, default=42, help="Random seed for deterministic sampling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    triples = read_triples(args.input)
    if not triples:
        raise ValueError(f"No valid triples found in: {args.input}")

    seed = pick_seed_drug(triples, args.seed_drug)
    graph = sample_subgraph(
        triples=triples,
        seed=seed,
        max_edges=max(10, args.max_edges),
        max_neighbors_per_node=max(2, args.max_neighbors),
        rng_seed=args.rng_seed,
    )

    if graph.number_of_edges() == 0:
        raise ValueError("Sampled graph is empty. Try increasing --max-edges or changing --seed-drug.")

    draw_graph(graph, seed=seed, out_path=args.output)
    print(f"Saved graph plot: {args.output}")
    print(f"Nodes: {graph.number_of_nodes()}  Edges: {graph.number_of_edges()}")


if __name__ == "__main__":
    main()
