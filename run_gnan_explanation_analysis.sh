#!/bin/bash
# Script to run GNAN explanation analysis
#
# This script loads a trained GNAN model and analyzes predictions,
# extracting important nodes and optionally comparing with augmented context.
#
# Usage:
#   ./run_gnan_explanation_analysis.sh [algo_version]
#
# Examples:
#   ./run_gnan_explanation_analysis.sh 4     # Analyze algo4 (GNAN with merge mode)
#   ./run_gnan_explanation_analysis.sh 3     # Analyze algo3 (for comparison)

ALGO_VERSION=${1:-4}  # Default to algo4

# Load environment variables
if [ -f db.env ]; then
    export $(cat db.env | grep -v '^#' | xargs)
fi

echo "========================================"
echo "GNAN Explanation Analysis"
echo "Algorithm version: ${ALGO_VERSION}"
echo "========================================"

# Run the analysis with default settings:
# - 2 correct examples
# - 2 incorrect examples
# - Top 10 important nodes
# - Compare with/without augmented context
python analyze_gnan_explanations.py \
    --algo_version ${ALGO_VERSION} \
    --retrieval_version 0 \
    --g_retriever_version 0 \
    --llama_version llama3.1-8b \
    --num_correct 2 \
    --num_incorrect 2 \
    --topk_nodes 10 \
    --compare_augmented \
    --max_examples_to_scan 100

echo ""
echo "Analysis complete! Check the output files in stark_qa_v0_${ALGO_VERSION}/models/"



