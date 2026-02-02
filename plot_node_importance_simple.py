#!/usr/bin/env python3
"""
Simplified script to visualize node importance from processed test data.
Works without needing stark-qa installation.
"""

import argparse
import io
import os
import sys
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric import seed_everything
from torch_geometric.nn import GRetriever, TensorGNAN
from torch_geometric.nn.nlp import LLM


def load_params_dict(model, save_path):
    """Load model parameters from a checkpoint file."""
    state_dict = model.state_dict()
    checkpoint = torch.load(save_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    state_dict.update(checkpoint)
    model.load_state_dict(state_dict)
    return model


def get_node_importance(model, data, device='cpu'):
    """Compute node importance scores using GNAN."""
    data = data.to(device)
    with torch.no_grad():
        contrib = model.gnn.node_importance(data)
        scores = contrib.sum(dim=1)
    return scores.cpu().numpy()


def parse_node_info_from_desc(desc: str):
    """Parse node information from description CSV format."""
    sep = '\nsrc,edge_attr,dst\n'
    pos = desc.find(sep)
    nodes_csv = desc[:pos] if pos != -1 else desc

    try:
        nodes_df = pd.read_csv(io.StringIO(nodes_csv))
        if 'node_attr' in nodes_df.columns:
            node_attr_series = nodes_df['node_attr'].astype(str)
            names_series = node_attr_series.str.extract(r'^name:\s*(.*?),(?:\s*)description:\s*.*$')[0]
            descs_series = node_attr_series.str.extract(r'^name:\s*.*?,(?:\s*)description:\s*(.*)$')[0]
            names = names_series.fillna(node_attr_series).tolist()
            descriptions = descs_series.fillna('').tolist()
        else:
            fallback_series = nodes_df[nodes_df.columns[-1]].astype(str)
            names = fallback_series.tolist()
            descriptions = ['' for _ in names]

        return [{'name': n, 'description': d} for n, d in zip(names, descriptions)]
    except Exception as e:
        print(f"Warning: Could not parse node info: {e}")
        return []


def create_networkx_graph(data, node_info: List[dict], importance_scores: np.ndarray):
    """Create a NetworkX graph from PyG data."""
    G = nx.DiGraph()

    num_nodes = data.x.size(0)
    for i in range(num_nodes):
        node_name = node_info[i]['name'] if i < len(node_info) else f"Node {i}"
        node_desc = node_info[i]['description'] if i < len(node_info) else ""

        if len(node_desc) > 100:
            node_desc = node_desc[:100] + "..."

        G.add_node(i, name=node_name, description=node_desc, importance=float(importance_scores[i]))

    edge_index = data.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        G.add_edge(int(src), int(dst))

    return G


def plot_subgraph_with_importance(G, importance_scores, title, top_k=10, save_path=None):
    """Plot subgraph with node importance highlighted."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))

    top_k = min(top_k, len(importance_scores))
    top_indices = np.argsort(importance_scores)[-top_k:][::-1]

    norm_scores = importance_scores.copy()
    if norm_scores.max() > 0:
        norm_scores = norm_scores / norm_scores.max()

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Plot 1: Full graph
    ax1.set_title(f"{title}\nFull Subgraph (colored by importance)", fontsize=14, fontweight='bold')
    nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.3, arrows=True, arrowsize=10, edge_color='gray', width=1)
    node_colors = [norm_scores[i] for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, cmap=plt.cm.YlOrRd,
                          node_size=500, vmin=0, vmax=1, alpha=0.8)
    top_labels = {i: G.nodes[i]['name'][:20] for i in top_indices if i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=top_labels, ax=ax1, font_size=8, font_weight='bold')
    ax1.axis('off')

    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Node Importance', rotation=270, labelpad=20)

    # Plot 2: Top-k subgraph
    top_k_nodes = set(top_indices)
    extended_nodes = set(top_k_nodes)
    for node in top_k_nodes:
        if node in G:
            extended_nodes.update(G.predecessors(node))
            extended_nodes.update(G.successors(node))

    if len(extended_nodes) > top_k * 3:
        extended_list = sorted(extended_nodes, key=lambda x: importance_scores[x], reverse=True)[:top_k * 3]
        extended_nodes = set(extended_list)

    G_sub = G.subgraph(extended_nodes).copy()
    pos_sub = {k: pos[k] for k in G_sub.nodes()}

    ax2.set_title(f"Top-{top_k} Important Nodes + Neighbors", fontsize=14, fontweight='bold')
    nx.draw_networkx_edges(G_sub, pos_sub, ax=ax2, alpha=0.4, arrows=True, arrowsize=15, edge_color='gray', width=2)

    top_k_in_sub = [n for n in G_sub.nodes() if n in top_k_nodes]
    neighbor_nodes = [n for n in G_sub.nodes() if n not in top_k_nodes]

    if neighbor_nodes:
        nx.draw_networkx_nodes(G_sub, pos_sub, nodelist=neighbor_nodes, ax=ax2,
                              node_color='lightgray', node_size=400, alpha=0.6)

    if top_k_in_sub:
        top_k_colors = [norm_scores[i] for i in top_k_in_sub]
        nx.draw_networkx_nodes(G_sub, pos_sub, nodelist=top_k_in_sub, ax=ax2,
                              node_color=top_k_colors, cmap=plt.cm.YlOrRd, node_size=800,
                              vmin=0, vmax=1, alpha=1.0, edgecolors='black', linewidths=2)

    labels_sub = {i: G_sub.nodes[i]['name'][:20] for i in G_sub.nodes()}
    nx.draw_networkx_labels(G_sub, pos_sub, labels=labels_sub, ax=ax2, font_size=9, font_weight='bold')
    ax2.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")

    plt.close()
    return fig


def plot_importance_bar_chart(importance_scores, node_info, title, top_k=20, save_path=None):
    """Plot bar chart of top-k node importance scores."""
    top_k = min(top_k, len(importance_scores))
    top_indices = np.argsort(importance_scores)[-top_k:][::-1]
    top_scores = importance_scores[top_indices]

    node_names = []
    for idx in top_indices:
        if idx < len(node_info):
            name = node_info[idx]['name']
            if len(name) > 30:
                name = name[:27] + "..."
            node_names.append(f"{name} [{idx}]")
        else:
            node_names.append(f"Node {idx}")

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.YlOrRd(np.linspace(0.4, 1.0, top_k))

    y_pos = np.arange(top_k)
    bars = ax.barh(y_pos, top_scores, color=colors, alpha=0.8, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(node_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f' {score:.4f}',
               ha='left', va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved bar chart to: {save_path}")

    plt.close()
    return fig


def plot_example_analysis(model, data, example_idx, output_dir, top_k=10, device='cpu'):
    """Complete analysis and plotting for a single example."""
    importance_scores = get_node_importance(model, data, device)

    desc = data.desc if isinstance(data.desc, str) else data.desc[0]
    node_info = parse_node_info_from_desc(desc)

    question = data.question if isinstance(data.question, str) else data.question[0]
    label = data.label if isinstance(data.label, str) else data.label[0]

    clean_question = question.replace('Question: ', '').replace('Answer: ', '').strip()
    if len(clean_question) > 100:
        clean_question = clean_question[:100] + "..."

    print(f"\n{'='*80}")
    print(f"Example {example_idx}")
    print(f"{'='*80}")
    print(f"Question: {clean_question}")
    print(f"Answer: {label}")
    print(f"Nodes: {data.x.size(0)}, Edges: {data.edge_index.size(1)}")

    top_k_actual = min(top_k, len(importance_scores))
    top_indices = np.argsort(importance_scores)[-top_k_actual:][::-1]

    print(f"\nTop-{top_k_actual} Important Nodes:")
    print(f"{'-'*80}")
    for rank, idx in enumerate(top_indices, 1):
        score = importance_scores[idx]
        if idx < len(node_info):
            name = node_info[idx]['name']
            desc_text = node_info[idx]['description']
            if len(desc_text) > 80:
                desc_text = desc_text[:80] + "..."
            print(f"  {rank:2d}. [{idx:3d}] {name:40s} | Score: {score:.4f}")
            if desc_text:
                print(f"      {desc_text}")
        else:
            print(f"  {rank:2d}. [Node {idx}] Score: {score:.4f}")

    G = create_networkx_graph(data, node_info, importance_scores)

    graph_title = f"Example {example_idx}: {clean_question}"
    graph_save_path = os.path.join(output_dir, f"example_{example_idx}_subgraph.png")
    plot_subgraph_with_importance(G, importance_scores, graph_title, top_k=top_k, save_path=graph_save_path)

    bar_title = f"Example {example_idx}: Top-{top_k_actual} Node Importance"
    bar_save_path = os.path.join(output_dir, f"example_{example_idx}_importance_bars.png")
    plot_importance_bar_chart(importance_scores, node_info, bar_title, top_k=20, save_path=bar_save_path)

    summary_path = os.path.join(output_dir, f"example_{example_idx}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Example {example_idx} - Node Importance Analysis\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Question: {question}\n\n")
        f.write(f"Answer: {label}\n\n")
        f.write(f"Graph Statistics:\n")
        f.write(f"  - Number of nodes: {data.x.size(0)}\n")
        f.write(f"  - Number of edges: {data.edge_index.size(1)}\n")
        f.write(f"  - Max importance score: {importance_scores.max():.4f}\n")
        f.write(f"  - Min importance score: {importance_scores.min():.4f}\n")
        f.write(f"  - Mean importance score: {importance_scores.mean():.4f}\n\n")

        f.write(f"Top-{top_k_actual} Important Nodes:\n")
        f.write(f"{'-'*80}\n")
        for rank, idx in enumerate(top_indices, 1):
            score = importance_scores[idx]
            if idx < len(node_info):
                name = node_info[idx]['name']
                desc_text = node_info[idx]['description']
                f.write(f"\n{rank:2d}. Node Index: {idx}\n")
                f.write(f"    Name: {name}\n")
                f.write(f"    Importance Score: {score:.4f}\n")
                if desc_text:
                    f.write(f"    Description: {desc_text}\n")
            else:
                f.write(f"\n{rank:2d}. Node {idx}: Score {score:.4f}\n")

    print(f"\nSaved summary to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot node importance from trained GNAN model.')
    parser.add_argument('--algo_version', type=int, default=4, help='Algorithm config version')
    parser.add_argument('--retrieval_version', type=int, default=0, help='Retrieval config version')
    parser.add_argument('--g_retriever_version', type=int, default=0, help='G-Retriever config version')
    parser.add_argument('--llama_version', type=str, default='llama3.1-8b', help='LLaMA version')
    parser.add_argument('--num_examples', type=int, default=3, help='Number of examples to plot')
    parser.add_argument('--specific_indices', type=int, nargs='+', default=None, help='Specific indices to plot')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top important nodes')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')

    args = parser.parse_args()

    seed_everything(args.seed)

    root_path = f"stark_qa_v{args.retrieval_version}_{args.algo_version}"

    if args.model_path is None:
        model_name = f"{args.retrieval_version}_{args.algo_version}_{args.g_retriever_version}_gnn-llm-{args.llama_version}.pt"
        model_path = os.path.join(root_path, 'models', model_name)

        best_ckpt_name = f"{args.retrieval_version}_{args.algo_version}_{args.g_retriever_version}_gnn-llm-{args.llama_version}_best_val_loss_ckpt.pt"
        best_ckpt_path = os.path.join(root_path, 'models', best_ckpt_name)
        if os.path.exists(best_ckpt_path):
            model_path = best_ckpt_path
            print(f"Using best validation checkpoint: {model_path}")
        else:
            print(f"Using model: {model_path}")
    else:
        model_path = args.model_path
        print(f"Using specified model: {model_path}")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(root_path, 'visualizations',
                                       f'node_importance_algo{args.algo_version}_{timestamp}')

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput directory: {args.output_dir}")

    # Load test dataset directly from processed file
    print(f"\nLoading test dataset from {root_path}/processed/test_data.pt...")
    test_data_path = os.path.join(root_path, 'processed', 'test_data.pt')
    if not os.path.exists(test_data_path):
        print(f"Error: Test data not found: {test_data_path}")
        sys.exit(1)

    test_dataset = torch.load(test_data_path)
    print(f"Loaded {len(test_dataset)} test examples")

    # Create model
    print("\nInitializing model architecture...")
    gnn = TensorGNAN(
        in_channels=1536,
        hidden_channels=1536,
        out_channels=1536,
        n_layers=4,
        normalize_rho=True,
        feature_groups=[list(range(1536))],
    )

    if args.llama_version == 'tiny_llama':
        llm = LLM(model_name='TinyLlama/TinyLlama-1.1B-Chat-v0.1', num_params=1)
        model = GRetriever(llm=llm, gnn=gnn, mlp_out_channels=2048)
    elif args.llama_version == 'llama2-7b':
        llm = LLM(model_name='meta-llama/Llama-2-7b-chat-hf', num_params=7)
        model = GRetriever(llm=llm, gnn=gnn)
    elif args.llama_version == 'llama3.1-8b':
        llm = LLM(model_name='meta-llama/Llama-3.1-8B-Instruct', num_params=8)
        model = GRetriever(llm=llm, gnn=gnn)
    else:
        raise ValueError(f"Unknown llama_version: {args.llama_version}")

    print(f"Loading model weights from {model_path}...")
    model = load_params_dict(model, model_path)
    model.eval()
    device = next(model.gnn.parameters()).device
    print(f"Model loaded successfully on device: {device}")

    # Determine which examples to plot
    if args.specific_indices is not None:
        indices_to_plot = args.specific_indices
        print(f"\nPlotting specific indices: {indices_to_plot}")
    else:
        max_idx = len(test_dataset)
        np.random.seed(args.seed)
        indices_to_plot = np.random.choice(max_idx, size=min(args.num_examples, max_idx),
                                          replace=False).tolist()
        print(f"\nPlotting {len(indices_to_plot)} random examples: {indices_to_plot}")

    # Plot each example
    for idx in indices_to_plot:
        if idx >= len(test_dataset):
            print(f"\nWarning: Index {idx} is out of range (dataset has {len(test_dataset)} examples)")
            continue

        try:
            data = test_dataset[idx]
            plot_example_analysis(model, data, idx, args.output_dir, top_k=args.top_k, device=device)
        except Exception as e:
            print(f"\nError plotting example {idx}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"All plots saved to: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
