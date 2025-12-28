#!/bin/bash
# Run training with GAT (Graph Attention Network) using algo_config_version 3 with PCST context

python train.py --checkpointing \
--llama_version llama3.1-8b --retrieval_config_version 0 --g_retriever_config_version 0 --eval_batch_size 1 \
--num_gnn_layers 4 --algo_config_version 3 \
--gnn_type gat \
--include_pcst_desc_context

