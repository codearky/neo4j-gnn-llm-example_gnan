# Node Importance Visualization for GNAN Models

This documentation explains how to visualize node importance and subgraph structures from trained GNAN models.

## Overview

The visualization script (`plot_node_importance.py`) provides comprehensive visual analysis of which nodes the Graph Neural Network considers most important when answering questions. This helps understand:

- Which entities the model focuses on
- How node importance is distributed across the graph
- Whether important nodes align with correct answers
- The structure of the subgraph around important nodes

## Quick Start

### Basic Usage

```bash
# Plot 3 random examples from algorithm version 4
./run_plot_node_importance.sh 4 3

# Plot 5 random examples from algorithm version 3
./run_plot_node_importance.sh 3 5
```

### Advanced Usage

```bash
# Plot specific test set indices
python plot_node_importance.py --algo_version 4 --specific_indices 0 5 10 15

# Plot 10 examples with more top-k nodes highlighted
python plot_node_importance.py --algo_version 4 --num_examples 10 --top_k 20

# Use a specific model checkpoint
python plot_node_importance.py --algo_version 4 --model_path path/to/checkpoint.pt

# Specify custom output directory
python plot_node_importance.py --algo_version 4 --output_dir ./my_visualizations
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--algo_version` | int | 4 | Algorithm config version |
| `--retrieval_version` | int | 0 | Retrieval config version |
| `--g_retriever_version` | int | 0 | G-Retriever config version |
| `--llama_version` | str | llama3.1-8b | LLaMA model version |
| `--num_examples` | int | 3 | Number of random examples to plot |
| `--specific_indices` | int[] | None | Specific test indices to plot (e.g., `0 5 10`) |
| `--top_k` | int | 10 | Number of top important nodes to highlight |
| `--output_dir` | str | auto | Output directory for plots |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--model_path` | str | auto | Path to model checkpoint |

## Output Files

For each example, the script generates three files:

### 1. Subgraph Visualization (`example_N_subgraph.png`)

A side-by-side visualization showing:
- **Left panel**: Full subgraph with all nodes colored by importance (red = high, yellow = low)
- **Right panel**: Top-k important nodes plus their immediate neighbors for focused analysis

Features:
- Node colors indicate importance (heatmap from yellow to red)
- Edges show relationships between entities
- Labels show node names for important nodes
- Colorbar indicates importance scale

### 2. Importance Bar Chart (`example_N_importance_bars.png`)

A horizontal bar chart showing:
- Top-20 most important nodes ranked by score
- Node names and indices
- Exact importance scores
- Color gradient indicating relative importance

### 3. Text Summary (`example_N_summary.txt`)

A detailed text report containing:
- Question and answer
- Graph statistics (number of nodes/edges)
- Importance score statistics (min, max, mean)
- Complete list of top-k nodes with names, descriptions, and scores

## Example Workflow

### 1. Train a Model

```bash
python train.py \
    --llama_version llama3.1-8b \
    --algo_config_version 4 \
    --retrieval_config_version 0 \
    --g_retriever_config_version 0 \
    --num_gnn_layers 4 \
    --epochs 2
```

### 2. Visualize Node Importance

```bash
# Quick visualization of 5 examples
./run_plot_node_importance.sh 4 5

# Or plot specific interesting examples
python plot_node_importance.py --algo_version 4 --specific_indices 0 42 100
```

### 3. Analyze Results

Check the generated visualizations in the output directory:
```
stark_qa_v0_4/visualizations/node_importance_algo4_YYYYMMDD_HHMMSS/
├── example_0_subgraph.png
├── example_0_importance_bars.png
├── example_0_summary.txt
├── example_5_subgraph.png
├── example_5_importance_bars.png
├── example_5_summary.txt
└── ...
```

## Interpreting the Visualizations

### Node Colors in Subgraph

- **Dark Red**: Very high importance - the model strongly focuses on these nodes
- **Orange/Yellow**: Medium importance - moderately relevant nodes
- **Light Yellow**: Low importance - less relevant to the answer

### What to Look For

1. **Correct Predictions**: Do the top important nodes contain or relate to the answer?
2. **Incorrect Predictions**: Are important nodes misleading or off-topic?
3. **Graph Structure**: Are important nodes well-connected or isolated?
4. **Importance Distribution**: Is importance concentrated in a few nodes or spread out?

## Integration with Existing Analysis

This visualization tool complements the existing analysis script:

- `analyze_gnan_explanations.py`: Textual analysis and LLM explanations
- `plot_node_importance.py`: Visual analysis with graphs and charts

You can use both together:

```bash
# Run textual analysis
./run_gnan_explanation_analysis.sh 4

# Run visual analysis
./run_plot_node_importance.sh 4 5
```

## Troubleshooting

### Model Not Found

If you see "Model file not found", make sure:
1. You've trained a model for the specified algorithm version
2. The model checkpoint exists in `stark_qa_v{retrieval}_{algo}/models/`
3. You're using the correct version numbers

### Out of Memory

For large graphs, reduce:
- `--num_examples` to plot fewer examples
- `--top_k` to show fewer nodes

### Import Errors

Required packages:
```bash
pip install matplotlib networkx numpy pandas torch torch_geometric
```

## Tips for Best Results

1. **Start Small**: Begin with 3-5 examples to get a feel for the visualizations
2. **Compare Versions**: Plot the same indices across different algorithm versions
3. **Check Edge Cases**: Plot both correct and incorrect predictions for comparison
4. **High-Quality Output**: The PNG files are saved at 300 DPI for publication quality

## Related Scripts

- `train.py`: Train GNAN models
- `analyze_gnan_explanations.py`: Analyze predictions and generate explanations
- `run_gnan_explanation_analysis.sh`: Easy launcher for explanation analysis

## Citation

If you use these visualizations in your research, please cite the GNAN paper and repository.
