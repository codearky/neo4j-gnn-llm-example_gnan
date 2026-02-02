#!/usr/bin/env python3
"""
Example script showing how to use the node importance plotting functions programmatically.

This demonstrates how to integrate the visualization functions into your own analysis pipeline.
"""

import os
import torch
from dotenv import load_dotenv
from stark_qa import load_qa
from torch_geometric import seed_everything
from torch_geometric.nn import GRetriever, TensorGNAN
from torch_geometric.nn.nlp import LLM
from torch_geometric.transforms.gnan import PreprocessDistances

from STaRKQADatasetGDS import STaRKQADataset
from plot_node_importance import (
    load_params_dict,
    get_node_importance,
    parse_node_info_from_desc,
    create_networkx_graph,
    plot_subgraph_with_importance,
    plot_importance_bar_chart,
)


def main():
    """Example of using the visualization functions in your own code."""

    # Setup
    load_dotenv('db.env', override=True)
    seed_everything(42)

    # Configuration
    algo_version = 4
    retrieval_version = 0
    g_retriever_version = 0
    llama_version = 'llama3.1-8b'

    # Paths
    root_path = f"stark_qa_v{retrieval_version}_{algo_version}"
    model_name = f"{retrieval_version}_{algo_version}_{g_retriever_version}_gnn-llm-{llama_version}.pt"
    model_path = os.path.join(root_path, 'models', model_name)

    # Check for best checkpoint
    best_ckpt = model_path.replace('.pt', '_best_val_loss_ckpt.pt')
    if os.path.exists(best_ckpt):
        model_path = best_ckpt

    print(f"Loading model from: {model_path}")

    # Load test dataset
    qa_dataset = load_qa("prime")
    qa_raw_test = qa_dataset.get_subset('test')
    test_dataset = STaRKQADataset(
        root_path, qa_raw_test,
        retrieval_version, algo_version,
        split="test",
        transform=PreprocessDistances()
    )

    # Create model
    gnn = TensorGNAN(
        in_channels=1536,
        hidden_channels=1536,
        out_channels=1536,
        n_layers=4,
        normalize_rho=True,
        feature_groups=[list(range(1536))],
    )

    llm = LLM(
        model_name='meta-llama/Llama-3.1-8B-Instruct',
        num_params=8
    )

    model = GRetriever(llm=llm, gnn=gnn)
    model = load_params_dict(model, model_path)
    model.eval()

    device = next(model.gnn.parameters()).device
    print(f"Model loaded on device: {device}")

    # Select an example
    example_idx = 0
    data = test_dataset[example_idx]

    print(f"\nAnalyzing example {example_idx}...")
    print(f"Question: {data.question if isinstance(data.question, str) else data.question[0]}")

    # Get node importance scores
    importance_scores = get_node_importance(model, data, device)
    print(f"Computed importance for {len(importance_scores)} nodes")

    # Parse node information
    desc = data.desc if isinstance(data.desc, str) else data.desc[0]
    node_info = parse_node_info_from_desc(desc)

    # Print top-5 most important nodes
    top_5_indices = importance_scores.argsort()[-5:][::-1]
    print("\nTop-5 Most Important Nodes:")
    for rank, idx in enumerate(top_5_indices, 1):
        score = importance_scores[idx]
        name = node_info[idx]['name'] if idx < len(node_info) else f"Node {idx}"
        print(f"  {rank}. {name} (score: {score:.4f})")

    # Create visualizations
    output_dir = "example_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Create NetworkX graph
    G = create_networkx_graph(data, node_info, importance_scores)

    # Plot subgraph
    plot_subgraph_with_importance(
        G, importance_scores,
        title=f"Example {example_idx}",
        top_k=10,
        save_path=os.path.join(output_dir, "example_subgraph.png")
    )

    # Plot bar chart
    plot_importance_bar_chart(
        importance_scores, node_info,
        title=f"Example {example_idx}: Node Importance",
        top_k=15,
        save_path=os.path.join(output_dir, "example_importance_bars.png")
    )

    print(f"\nVisualizations saved to: {output_dir}/")
    print("  - example_subgraph.png")
    print("  - example_importance_bars.png")

    # You can also do custom analysis on the importance scores
    # For example, check if important nodes match the answer
    label = data.label if isinstance(data.label, str) else data.label[0]
    label_lower = label.lower()

    matching_nodes = []
    for idx in top_5_indices:
        if idx < len(node_info):
            node_name = node_info[idx]['name'].lower()
            if node_name in label_lower or label_lower in node_name:
                matching_nodes.append(node_info[idx]['name'])

    if matching_nodes:
        print(f"\nImportant nodes matching answer: {matching_nodes}")
    else:
        print("\nNo important nodes directly match the answer")

    # Advanced: You can integrate this into a larger pipeline
    # For example, comparing importance across different models,
    # or analyzing importance patterns for correct vs incorrect predictions


if __name__ == '__main__':
    main()
