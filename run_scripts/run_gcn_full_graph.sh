#!/bin/bash
# Run training with GCN (Graph Convolutional Network) using full graph context

python train.py --checkpointing \
--llama_version llama3.1-8b --retrieval_config_version 0 --g_retriever_config_version 0 --eval_batch_size 1 \
--num_gnn_layers 4 --algo_config_version 0 \
--gnn_type gcn \
--use_full_graph_context

