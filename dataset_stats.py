import argparse
import os
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from stark_qa import load_qa
from STaRKQADatasetGDS import STaRKQADataset


def compute_split_stats(dataset: STaRKQADataset) -> Dict[str, float]:
    node_counts: List[int] = []
    edge_counts: List[int] = []
    feature_dims: List[int] = []

    for i in range(len(dataset)):
        data_i = dataset[i]
        num_nodes = int(data_i.x.size(0)) if getattr(data_i, 'x', None) is not None else 0
        num_edges = int(data_i.edge_index.size(1)) if getattr(data_i, 'edge_index', None) is not None else 0
        feat_dim = int(data_i.x.size(1)) if getattr(data_i, 'x', None) is not None else 0

        node_counts.append(num_nodes)
        edge_counts.append(num_edges)
        feature_dims.append(feat_dim)

    node_counts_np = np.array(node_counts, dtype=np.int64)
    edge_counts_np = np.array(edge_counts, dtype=np.int64)
    with np.errstate(divide='ignore', invalid='ignore'):
        edges_per_node = np.where(node_counts_np > 0, edge_counts_np / node_counts_np, 0.0)

    stats = {
        'num_graphs': len(dataset),
        'nodes_avg': float(np.mean(node_counts_np)) if len(node_counts_np) else 0.0,
        'nodes_std': float(np.std(node_counts_np)) if len(node_counts_np) else 0.0,
        'nodes_min': int(np.min(node_counts_np)) if len(node_counts_np) else 0,
        'nodes_median': float(np.median(node_counts_np)) if len(node_counts_np) else 0.0,
        'nodes_max': int(np.max(node_counts_np)) if len(node_counts_np) else 0,
        'edges_avg': float(np.mean(edge_counts_np)) if len(edge_counts_np) else 0.0,
        'edges_std': float(np.std(edge_counts_np)) if len(edge_counts_np) else 0.0,
        'edges_min': int(np.min(edge_counts_np)) if len(edge_counts_np) else 0,
        'edges_median': float(np.median(edge_counts_np)) if len(edge_counts_np) else 0.0,
        'edges_max': int(np.max(edge_counts_np)) if len(edge_counts_np) else 0,
        'edges_per_node_avg': float(np.mean(edges_per_node)) if len(edges_per_node) else 0.0,
        'feature_dim_mode': int(np.bincount(np.array(feature_dims)).argmax()) if len(feature_dims) and np.any(feature_dims) else 0,
    }
    return stats


def print_stats(title: str, stats: Dict[str, float]):
    print(f"\n=== {title} ===")
    print(f"Graphs: {stats['num_graphs']}")
    print(f"Nodes  | avg {stats['nodes_avg']:.2f} | std {stats['nodes_std']:.2f} | min {stats['nodes_min']} | median {stats['nodes_median']:.2f} | max {stats['nodes_max']}")
    print(f"Edges  | avg {stats['edges_avg']:.2f} | std {stats['edges_std']:.2f} | min {stats['edges_min']} | median {stats['edges_median']:.2f} | max {stats['edges_max']}")
    print(f"Edges/Node avg: {stats['edges_per_node_avg']:.3f}")
    print(f"Feature dim (mode): {stats['feature_dim_mode']}")


def load_processed_split(root_path: str, raw_subset, retrieval_config_version: int, algo_config_version: int, split: str) -> STaRKQADataset:
    # This will only process if processed files are missing. Otherwise it will load from disk.
    ds = STaRKQADataset(
        root=root_path,
        raw_dataset=raw_subset,
        retrieval_config_version=retrieval_config_version,
        algo_config_version=algo_config_version,
        split=split,
        force_reload=False,
    )
    return ds


def main():
    parser = argparse.ArgumentParser(description='Compute statistics for processed STaRKQA GNN datasets (train/val/test).')
    parser.add_argument('--retrieval_config_version', type=int, required=True)
    parser.add_argument('--algo_config_version', type=int, required=True)
    parser.add_argument('--root_path', type=str, default=None, help='Override dataset root. Defaults to stark_qa_v{retrieval}_{algo}')
    args = parser.parse_args()

    qa_dataset = load_qa("prime")
    qa_raw_train = qa_dataset.get_subset('train')
    qa_raw_val = qa_dataset.get_subset('val')
    qa_raw_test = qa_dataset.get_subset('test')

    root_path = args.root_path or f"stark_qa_v{args.retrieval_config_version}_{args.algo_config_version}"

    if not os.path.isdir(root_path):
        print(f"Warning: root path '{root_path}' does not exist. If processed files are missing, this script may attempt to process and will require database access.")

    print(f"Loading processed datasets from: {root_path}")

    train_ds = load_processed_split(root_path, qa_raw_train, args.retrieval_config_version, args.algo_config_version, split='train')
    val_ds = load_processed_split(root_path, qa_raw_val, args.retrieval_config_version, args.algo_config_version, split='val')
    test_ds = load_processed_split(root_path, qa_raw_test, args.retrieval_config_version, args.algo_config_version, split='test')

    train_stats = compute_split_stats(train_ds)
    val_stats = compute_split_stats(val_ds)
    test_stats = compute_split_stats(test_ds)

    print_stats('Train', train_stats)
    print_stats('Val', val_stats)
    print_stats('Test', test_stats)


if __name__ == '__main__':
    main() 