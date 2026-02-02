#!/bin/bash
# Script to visualize node importance from a trained GNAN model
#
# This script loads a trained GNAN model and creates visualizations showing
# which nodes the GNN considers most important for answering questions.
#
# Usage:
#   ./run_plot_node_importance.sh [algo_version] [num_examples]
#
# Examples:
#   ./run_plot_node_importance.sh 4 5        # Plot 5 random examples from algo4
#   ./run_plot_node_importance.sh 3 3        # Plot 3 random examples from algo3

ALGO_VERSION=${1:-4}  # Default to algo4
NUM_EXAMPLES=${2:-3}  # Default to 3 examples

# Load environment variables
if [ -f db.env ]; then
    export $(cat db.env | grep -v '^#' | xargs)
fi

echo "========================================"
echo "GNAN Node Importance Visualization"
echo "Algorithm version: ${ALGO_VERSION}"
echo "Number of examples: ${NUM_EXAMPLES}"
echo "========================================"

# Run the visualization script
python plot_node_importance.py \
    --algo_version ${ALGO_VERSION} \
    --retrieval_version 0 \
    --g_retriever_version 0 \
    --llama_version llama3.1-8b \
    --num_examples ${NUM_EXAMPLES} \
    --top_k 10

echo ""
echo "Visualization complete! Check the output directory printed above."
echo ""
echo "You can also plot specific examples by index:"
echo "  python plot_node_importance.py --algo_version ${ALGO_VERSION} --specific_indices 0 5 10"
