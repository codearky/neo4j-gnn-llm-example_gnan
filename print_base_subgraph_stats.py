import argparse
import os
from collections import Counter
from typing import Any, Dict, Tuple

import pandas as pd
import torch


def load_base_subgraph(file_path: str) -> Dict[Any, Tuple[pd.DataFrame, pd.DataFrame]]:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        data = torch.load(file_path, weights_only=False)
    except TypeError:
        # For older torch versions where weights_only is not supported
        data = torch.load(file_path)
    if not isinstance(data, dict):
        raise ValueError("Expected a dict mapping -> (nodes_df, relationships_df)")
    return data


def compute_stats(base_subgraph: Dict[Any, Tuple[pd.DataFrame, pd.DataFrame]]) -> Dict[str, Any]:
    num_entries = len(base_subgraph)

    node_counts = []
    edge_counts = []

    global_unique_node_ids = set()
    relationship_type_counter: Counter = Counter()
    source_type_counter: Counter = Counter()
    target_type_counter: Counter = Counter()

    empty_node_entries = 0
    empty_edge_entries = 0

    for key, value in base_subgraph.items():
        try:
            nodes_df, relationships_df = value
        except Exception:
            raise ValueError(
                f"Entry {key} is not a tuple of (nodes_df, relationships_df). Got type: {type(value)}"
            )
        if not isinstance(nodes_df, pd.DataFrame) or not isinstance(relationships_df, pd.DataFrame):
            raise ValueError(
                f"Entry {key} values must be pandas DataFrames. Got: nodes={type(nodes_df)}, rels={type(relationships_df)}"
            )

        n_nodes = len(nodes_df)
        n_edges = len(relationships_df)

        node_counts.append(n_nodes)
        edge_counts.append(n_edges)

        if n_nodes == 0:
            empty_node_entries += 1
        if n_edges == 0:
            empty_edge_entries += 1

        if "nodeId" in nodes_df.columns:
            global_unique_node_ids.update(nodes_df["nodeId"].dropna().astype(int).tolist())

        if "relationshipType" in relationships_df.columns:
            relationship_type_counter.update(
                relationships_df["relationshipType"].dropna().astype(str).tolist()
            )

        if "sourceNodeType" in relationships_df.columns:
            source_type_counter.update(
                relationships_df["sourceNodeType"].dropna().astype(str).tolist()
            )
        if "targetNodeType" in relationships_df.columns:
            target_type_counter.update(
                relationships_df["targetNodeType"].dropna().astype(str).tolist()
            )

    def safe_min(values):
        return min(values) if values else 0

    def safe_max(values):
        return max(values) if values else 0

    def safe_avg(values):
        return (sum(values) / len(values)) if values else 0.0

    stats = {
        "num_entries": num_entries,
        "node_counts": {
            "min": safe_min(node_counts),
            "avg": safe_avg(node_counts),
            "max": safe_max(node_counts),
            "total": sum(node_counts),
            "empty_entries": empty_node_entries,
        },
        "edge_counts": {
            "min": safe_min(edge_counts),
            "avg": safe_avg(edge_counts),
            "max": safe_max(edge_counts),
            "total": sum(edge_counts),
            "empty_entries": empty_edge_entries,
        },
        "unique_nodes_global": len(global_unique_node_ids),
        "top_relationship_types": relationship_type_counter.most_common(10),
        "top_source_node_types": source_type_counter.most_common(10),
        "top_target_node_types": target_type_counter.most_common(10),
    }

    return stats


def print_stats(stats: Dict[str, Any], file_path: str) -> None:
    print(f"Base subgraph file: {file_path}")
    print("== Overview ==")
    print(f"Entries: {stats['num_entries']}")
    print(f"Unique nodes (global): {stats['unique_nodes_global']}")

    print("\n== Nodes per entry ==")
    nc = stats["node_counts"]
    print(f"min/avg/max: {nc['min']} / {nc['avg']:.2f} / {nc['max']}")
    print(f"total nodes across entries: {nc['total']}")
    print(f"entries with 0 nodes: {nc['empty_entries']}")

    print("\n== Edges per entry ==")
    ec = stats["edge_counts"]
    print(f"min/avg/max: {ec['min']} / {ec['avg']:.2f} / {ec['max']}")
    print(f"total edges across entries: {ec['total']}")
    print(f"entries with 0 edges: {ec['empty_entries']}")

    def print_top(title: str, items):
        print(f"\n== {title} ==")
        if not items:
            print("(none)")
            return
        for label, count in items:
            print(f"{label}: {count}")

    print_top("Top relationship types", stats["top_relationship_types"])
    print_top("Top source node types", stats["top_source_node_types"])
    print_top("Top target node types", stats["top_target_node_types"])


def main():
    # parser = argparse.ArgumentParser(description="Print statistics for a saved base subgraph .pt file")
    # parser.add_argument("file_path", help="Path to train_data_base_subgraph.pt")
    # args = parser.parse_args()

    base_subgraph = load_base_subgraph('/worxpace/neo4j-gnn-llm-example/base_subgraphs/v0/train_data_base_subgraph.pt')
    stats = compute_stats(base_subgraph)
    print_stats(stats, '/worxpace/neo4j-gnn-llm-example/base_subgraphs/v0/train_data_base_subgraph.pt')


if __name__ == "__main__":
    main() 