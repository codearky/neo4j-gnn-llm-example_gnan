import argparse
import math
import os
import time
import io
import json
import pandas as pd

import torch
from dotenv import load_dotenv
from torch_geometric.loader import DataLoader


from stark_qa import load_qa
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch_geometric import seed_everything
from torch_geometric.nn import GAT, GRetriever
from torch_geometric.nn.nlp import LLM
from tqdm import tqdm

from compute_metrics import compute_metrics

from STaRKQADatasetGDS import STaRKQADataset
from STaRKQAVectorSearchDataset import STaRKQAVectorSearchDataset

# Global variable for intersection log file path
_intersection_log_file = None
_current_epoch = None
_current_step = None


def get_loss(model, batch, model_save_name) -> Tensor:
    if model_save_name.startswith('llm'):
        return model(batch.question, batch.label, batch.desc)
    else:
        # Optionally augment context with PCST nodes and/or GNAN top-k nodes
        desc_arg = batch.desc
        try:
            if (getattr(args, 'include_pcst_desc_context', False)
                    or getattr(args, 'topk_gnan_nodes_context', 0) > 0):
                desc_arg = build_augmented_desc(model, batch)
        except NameError:
            desc_arg = batch.desc
        # calls forward for GRetriever
        return model(batch, batch.question, batch.x, batch.edge_index, batch.batch,
                     batch.label, batch.edge_attr,  desc_arg)



def inference_step(model, batch, model_save_name):
    if model_save_name.startswith('llm'):
        return model.inference(batch.question, batch.desc)
    else:
        # Optionally augment context with PCST nodes and/or GNAN top-k nodes
        desc_arg = batch.desc
        try:
            if (getattr(args, 'include_pcst_desc_context', False)
                    or getattr(args, 'topk_gnan_nodes_context', 0) > 0):
                desc_arg = build_augmented_desc(model, batch)
        except NameError:
            desc_arg = batch.desc
        return model.inference(batch, batch.question, batch.x, batch.edge_index,
                               batch.batch, batch.edge_attr, desc_arg)



def save_params_dict(model, save_path):
    state_dict = model.state_dict()
    param_grad_dict = {
        k: v.requires_grad
        for (k, v) in model.named_parameters()
    }
    for k in list(state_dict.keys()):
        if k in param_grad_dict.keys() and not param_grad_dict[k]:
            del state_dict[k]  # Delete parameters that do not require gradient
    torch.save(state_dict, save_path)


def save_checkpoint(model, optimizer, epoch, step, generator_state, save_path):
    """Save a complete training checkpoint including optimizer state and training progress."""
    state_dict = model.state_dict()
    param_grad_dict = {
        k: v.requires_grad
        for (k, v) in model.named_parameters()
    }
    for k in list(state_dict.keys()):
        if k in param_grad_dict.keys() and not param_grad_dict[k]:
            del state_dict[k]  # Delete parameters that do not require gradient
    
    checkpoint = {
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'generator_state': generator_state,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, generator, checkpoint_path):
    """Load a training checkpoint and return the epoch and step to resume from."""
    checkpoint = torch.load(checkpoint_path)
    
    # Load model state
    state_dict = model.state_dict()
    state_dict.update(checkpoint['model_state_dict'])
    model.load_state_dict(state_dict)
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore generator state
    generator.set_state(checkpoint['generator_state'])
    
    return checkpoint['epoch'], checkpoint['step']


def load_params_dict(model, save_path):
    state_dict = model.state_dict()
    state_dict.update(torch.load(save_path)) #All weights might not be saved, eg when using LoRA.
    model.load_state_dict(state_dict)
    return model



def print_node_importance_examples(model, test_dataset, print_node_description: bool, num_examples: int = 3):
    # Skip if GNN doesn't support node_importance (e.g., GAT)
    if not hasattr(model, 'gnn') or not hasattr(model.gnn, 'node_importance'):
        print("\nNode importance not supported for this GNN model.")
        return
    
    print(f"\nNode importance on {num_examples} random test examples:")
    sample_indices = torch.randperm(len(test_dataset))[:num_examples].tolist()
    for idx in sample_indices:
        data_i = test_dataset[idx]
        data_i = data_i.to(model.llm.device)
        with torch.no_grad():
            contrib = model.gnn.node_importance(data_i)
            scores = contrib.sum(dim=1) 
            topk = min(10, scores.numel())
            top_vals, top_idx = torch.topk(scores, k=topk)

            # Generate model answer for this single example
            q_list = [data_i.question if isinstance(data_i.question, str) else (data_i.question[0] if len(data_i.question) > 0 else '')]
            desc_list = [data_i.desc if isinstance(data_i.desc, str) else (data_i.desc[0] if len(data_i.desc) > 0 else '')]
            batch_vec = getattr(data_i, 'batch', None)
            if batch_vec is None:
                batch_vec = torch.zeros(data_i.x.size(0), dtype=torch.long, device=data_i.x.device)
            pred_list = model.inference(
                data_i,
                q_list,
                data_i.x,
                data_i.edge_index,
                batch_vec,
                data_i.edge_attr,
                desc_list,
            )
            model_answer = pred_list[0] if isinstance(pred_list, (list, tuple)) and len(pred_list) > 0 else str(pred_list)

        # Parse node words from desc (nodes CSV concatenated with edges CSV)
        desc = data_i.desc
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
        except Exception:
            names = []
            descriptions = []

        q_str = data_i.question if isinstance(data_i.question, str) else (data_i.question[0] if len(data_i.question) > 0 else '')
        print(f"\nExample idx={idx}")
        print(f"Question: {q_str} {model_answer}")
        for rank in range(topk):
            j = int(top_idx[rank])
            base_name = names[j] if j < len(names) else f'<node {j}>'
            if print_node_description and j < len(descriptions) and descriptions[j]:
                display_text = f"{base_name} | {descriptions[j]}"
            else:
                display_text = base_name
            print(f"{rank + 1}. {display_text} | score={float(top_vals[rank]):.4f}")



def evaluate_with_permuted_topk_node_features(model, test_dataset, topk: int = 30):
    # Skip if GNN doesn't support node_importance (e.g., GAT)
    if not hasattr(model, 'gnn') or not hasattr(model.gnn, 'node_importance'):
        print("\nNode importance evaluation not supported for this GNN model.")
        return []
    
    print(f"\nEvaluating with permuted features among top-{topk} important nodes per graph...")
    eval_output = []
    progress_bar = tqdm(range(len(test_dataset)))
    model_device = model.llm.device if hasattr(model, 'llm') else next(model.parameters()).device
    for idx in range(len(test_dataset)):
        data_i = test_dataset[idx]
        data_i = data_i.to(model_device)
        with torch.no_grad():
            contrib = model.gnn.node_importance(data_i)
            scores = contrib.sum(dim=1)
            k = min(topk, scores.numel())
            if k > 1:
                top_vals, top_idx = torch.topk(scores, k=k)
                x_perm = data_i.x.clone()
                feat_dim = x_perm.size(1)
                for node_idx in top_idx.tolist():
                    dim_perm = torch.randperm(feat_dim, device=x_perm.device)
                    x_perm[node_idx] = data_i.x[node_idx, dim_perm]
            else:
                x_perm = data_i.x

            q_list = [data_i.question if isinstance(data_i.question, str) else (data_i.question[0] if len(data_i.question) > 0 else '')]
            desc_list = [data_i.desc if isinstance(data_i.desc, str) else (data_i.desc[0] if len(data_i.desc) > 0 else '')]
            batch_vec = getattr(data_i, 'batch', None)
            if batch_vec is None:
                batch_vec = torch.zeros(data_i.x.size(0), dtype=torch.long, device=data_i.x.device)
            orig_x = data_i.x
            data_i.x = x_perm
            pred_list = model.inference(
                data_i,
                q_list,
                x_perm,
                data_i.edge_index,
                batch_vec,
                data_i.edge_attr,
                desc_list,
            )
            data_i.x = orig_x
            pred = pred_list[0] if isinstance(pred_list, (list, tuple)) and len(pred_list) > 0 else str(pred_list)

        eval_output.append({
            'pred': [pred],
            'question': [q_list[0]],
            'desc': [desc_list[0]],
            'label': [data_i.label if isinstance(data_i.label, str) else (data_i.label[0] if len(data_i.label) > 0 else '')],
        })
        progress_bar.update(1)
    return eval_output


def train(
    num_epochs,
    hidden_channels,
    num_gnn_layers,
    batch_size,
    eval_batch_size,
    lr,
    llama_version,
    retrieval_config_version,
    algo_config_version,
    g_retriever_config_version,
    checkpointing=False,
    sys_prompt=None,
    num_gpus=None,
    print_node_description: bool = False,
    load_model_path: str | None = None,
):
    def adjust_learning_rate(param_group, LR, epoch):
        # Decay the learning rate with half-cycle cosine after warmup
        min_lr = 5e-6
        warmup_epochs = 1
        if epoch < warmup_epochs:
            lr = LR
        else:
            lr = min_lr + (LR - min_lr) * 0.5 * (
                    1.0 + math.cos(math.pi * (epoch - warmup_epochs) /
                                   (num_epochs - warmup_epochs)))
        param_group['lr'] = lr
        return lr

    global _intersection_log_file, _current_epoch, _current_step
    
    start_time = time.time()
    qa_dataset = load_qa("prime")
    qa_raw_train = qa_dataset.get_subset('train')
    qa_raw_val = qa_dataset.get_subset('val')
    qa_raw_test = qa_dataset.get_subset('test')
    seed_everything(42)

    print("Loading stark-qa prime train dataset...")
    t = time.time()

    if num_gnn_layers == 0:
        model_save_name = f'llm-{llama_version}'
    else:
        if args.freeze_llm:
            model_save_name = f'gnn-frozen-llm-{llama_version}'
        else:
            model_save_name = f'gnn-llm-{llama_version}'

    if model_save_name == f'llm-{llama_version}':
        root_path = f"stark_qa_vector_rag_{retrieval_config_version}"
        train_dataset = STaRKQAVectorSearchDataset(root_path, qa_raw_train, split="train")
        print(f'Finished loading train dataset in {time.time() - t} seconds.')
        print("Loading stark-qa prime val dataset...")
        val_dataset = STaRKQAVectorSearchDataset(root_path, qa_raw_val, split="val")
        print("Loading stark-qa prime test dataset...")
        test_dataset = STaRKQAVectorSearchDataset(root_path, qa_raw_test, split="test")
        os.makedirs(f'{root_path}/models', exist_ok=True)
    else:
        root_path = f"stark_qa_v{retrieval_config_version}_{algo_config_version}"
        train_dataset = STaRKQADataset(root_path, qa_raw_train, retrieval_config_version, algo_config_version, split="train")
        print(f'Finished loading train dataset in {time.time() - t} seconds.')
        print("Loading stark-qa prime val dataset...")
        val_dataset = STaRKQADataset(root_path, qa_raw_val, retrieval_config_version, algo_config_version, split="val")
        print("Loading stark-qa prime test dataset...")
        test_dataset = STaRKQADataset(root_path, qa_raw_test, retrieval_config_version, algo_config_version, split="test")
        os.makedirs(f'{root_path}/models', exist_ok=True)

    # Set up intersection log file
    _intersection_log_file = f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}_gnan_pcst_intersection.log'
    # Clear the log file if it exists
    if os.path.exists(_intersection_log_file):
        os.remove(_intersection_log_file)

    # Use generator with fixed seed for reproducible shuffling
    train_generator = torch.Generator()
    train_generator.manual_seed(42)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, pin_memory=True, shuffle=True,
                              generator=train_generator)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size,
                            drop_last=False, pin_memory=True, shuffle=False )
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size,
                             drop_last=False, pin_memory=True, shuffle=False)

    gnn = GAT(
        in_channels=1536,
        hidden_channels=hidden_channels,
        out_channels=1536,
        num_layers=num_gnn_layers,
        heads=4,
    )

    if llama_version == 'tiny_llama':
        llm = LLM(
            model_name='TinyLlama/TinyLlama-1.1B-Chat-v0.1',
            num_params=1
        )
    elif llama_version == 'llama2-7b':
        llm = LLM(
            model_name='meta-llama/Llama-2-7b-chat-hf',
            num_params=7
        )
    elif llama_version == 'llama3.1-8b':
        llm = LLM(
            model_name='meta-llama/Llama-3.1-8B-Instruct',
            num_params=8
        )


    if args.freeze_llm:
        for param in llm.parameters():
            param.requires_grad = False

    if model_save_name == f'llm-{llama_version}':
        model = llm
    else:
        if llama_version == 'tiny_llama':
            model = GRetriever(llm=llm, gnn=gnn, mlp_out_channels=2048)
        else:
            model = GRetriever(llm=llm, gnn=gnn)

    print(f"Model device is: {llm.device}")

    if load_model_path is not None:
        print(f"Loading pre-trained model from: {load_model_path}")
        model = load_params_dict(model, load_model_path)

    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {
            'params': params,
            'lr': lr,
            'weight_decay': 0.05
        },
    ], betas=(0.9, 0.95))
    grad_steps = 8

    # Check for existing checkpoint to resume training
    checkpoint_path = f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}_checkpoint_latest.pt'
    start_epoch = 0
    start_step = 0
    global_step = 0
    
    if os.path.exists(checkpoint_path) and load_model_path is None:
        print(f"Found checkpoint at {checkpoint_path}, resuming training...")
        start_epoch, start_step = load_checkpoint(model, optimizer, train_generator, checkpoint_path)
        # Calculate global step
        global_step = start_epoch * len(train_loader) + start_step
        print(f"Resuming from epoch {start_epoch}, step {start_step}, global_step {global_step}")

    best_epoch = 0
    best_val_loss = float('inf')
    if load_model_path is None:
        for epoch in range(start_epoch, num_epochs):
            _current_epoch = epoch  # Set global for logging
            model.train()
            epoch_loss = 0
            if epoch == start_epoch and start_step == 0:
                print(f"Total Preparation Time: {time.time() - start_time:2f}s")
                start_time = time.time()
                print("Training beginning...")
            epoch_str = f'Epoch: {epoch + 1}|{num_epochs}'
            loader = tqdm(train_loader, desc=epoch_str)

            for step, batch in enumerate(loader):
                _current_step = step  # Set global for logging
                # Skip already processed steps when resuming from checkpoint
                if epoch == start_epoch and step < start_step:
                    continue
                
                optimizer.zero_grad()
                loss = get_loss(model, batch, model_save_name)
                loss.backward()

                clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

                if (step + 1) % grad_steps == 0:
                    adjust_learning_rate(optimizer.param_groups[0], lr,
                                         step / len(train_loader) + epoch)

                optimizer.step()
                epoch_loss = epoch_loss + float(loss)

                if (step + 1) % grad_steps == 0:
                    lr = optimizer.param_groups[0]['lr']
                
                # Save checkpoint every 500 steps
                global_step += 1
                if global_step % 500 == 0:
                    print(f"\nSaving checkpoint at global step {global_step}...")
                    save_checkpoint(
                        model, 
                        optimizer, 
                        epoch, 
                        step + 1,  # Save next step to start from
                        train_generator.get_state(),
                        checkpoint_path
                    )


            train_loss = epoch_loss / len(train_loader)
            print(epoch_str + f', Train Loss: {train_loss:4f}')

            val_loss = 0
            model.eval()
            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    loss = get_loss(model, batch, model_save_name)
                    val_loss += loss.item()
                val_loss = val_loss / len(val_loader)
                print(epoch_str + f", Val Loss: {val_loss:4f}")
            if checkpointing and val_loss < best_val_loss:
                print("Checkpointing best model...")
                best_val_loss = val_loss
                best_epoch = epoch
                save_params_dict(model, f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}_best_val_loss_ckpt.pt')

    if llm.device.type != "cpu":
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

    if checkpointing and load_model_path is None and best_epoch != num_epochs - 1:
        print("Loading best checkpoint...")
        model = load_params_dict(
            model,
            f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}_best_val_loss_ckpt.pt',
        )

    model.eval()
    eval_output = []
    node_importance_data = [] if (num_gnn_layers > 0 and getattr(args, 'save_node_importance', False)) else None
    
    print("Final evaluation...")
    progress_bar_test = tqdm(range(len(test_loader)))
    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            pred_time = time.time()
            pred = inference_step(model, batch, model_save_name)
            print(f"Time to predict: {time.time() - pred_time:2f}s")
            eval_data = {
                'pred': pred,
                'question': batch.question,
                'desc': batch.desc,
                'label': batch.label
            }
            eval_output.append(eval_data)
            
            # Collect node importance scores if requested
            if node_importance_data is not None and hasattr(model, 'gnn') and hasattr(model.gnn, 'node_importance'):
                try:
                    device = next(model.gnn.parameters()).device
                    batch_on_device = batch.to(device)
                    contrib = model.gnn.node_importance(batch_on_device)
                    scores = contrib.sum(dim=1).detach().cpu().numpy()
                    batch_vec = getattr(batch_on_device, 'batch', None)
                    if batch_vec is None:
                        batch_vec = torch.zeros(scores.shape[0], dtype=torch.long)
                    else:
                        batch_vec = batch_vec.cpu()
                    
                    # Split scores by graph
                    num_graphs_in_batch = batch_vec.max().item() + 1 if batch_vec.numel() > 0 else 1
                    for gid in range(num_graphs_in_batch):
                        node_mask = (batch_vec == gid).nonzero(as_tuple=False).view(-1)
                        graph_scores = scores[node_mask].tolist()
                        node_importance_data.append({
                            'batch_idx': step,
                            'graph_idx': gid,
                            'node_scores': graph_scores,
                            'num_nodes': len(graph_scores)
                        })
                except Exception as e:
                    print(f"Warning: Failed to compute node importance for batch {step}: {e}")
        progress_bar_test.update(1)

    compute_metrics(eval_output)
    
    # Save node importance data if collected
    if node_importance_data is not None:
        importance_path = f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}_node_importance.json'
        with open(importance_path, 'w') as f:
            json.dump(node_importance_data, f, indent=2)
        print(f"Node importance scores saved to: {importance_path}")
    # Permuted-topk evaluation for GNAN models
    if num_gnn_layers > 0:
        # permuted_eval_output = evaluate_with_permuted_topk_node_features(model, test_dataset, topk=10)
        # print("\nPermuted-top-10 metrics:")
        compute_metrics(eval_output)

    print(f"Total Training Time: {time.time() - start_time:2f}s")
    save_params_dict(model, f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}.pt')
    torch.save(eval_output, f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}_eval_outs.pt')
    # if num_gnn_layers > 0:
    #     torch.save(permuted_eval_output, f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}_perm_top10_eval_outs.pt')


def _safe_listify(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _compute_and_print_pcst_gnan_intersection(batch, topk_indices_per_graph, log_file, epoch=None, step=None):
    """
    Compute intersection statistics between GNAN top-k nodes and PCST nodes and write to file.
    
    Args:
        batch: The batch containing pcst_nodes attribute
        topk_indices_per_graph: Dict mapping graph_id -> list of top-k node indices
        log_file: File path to write the intersection statistics
        epoch: Optional epoch number for logging
        step: Optional step/iteration number for logging
    """
    pcst_nodes_attr = getattr(batch, 'pcst_nodes', None)
    if pcst_nodes_attr is None:
        return
    
    # Handle different possible formats of pcst_nodes
    pcst_nodes_list = _safe_listify(pcst_nodes_attr)
    num_graphs = len(topk_indices_per_graph)
    
    # If pcst_nodes is a single flat list, broadcast it
    if len(pcst_nodes_list) != num_graphs and num_graphs > 0:
        pcst_nodes_list = [pcst_nodes_list for _ in range(num_graphs)]
    
    total_intersection = 0
    total_k = 0
    total_pcst = 0
    
    # Get graph structure info from batch
    batch_vec = getattr(batch, 'batch', None)
    edge_index = getattr(batch, 'edge_index', None)
    
    # Compute nodes and edges per graph
    graph_node_counts = {}
    graph_edge_counts = {}
    
    if batch_vec is not None:
        for gid in range(num_graphs):
            node_mask = (batch_vec == gid).nonzero(as_tuple=False).view(-1)
            graph_node_counts[gid] = node_mask.numel()
    else:
        # Single graph case
        graph_node_counts[0] = batch.x.size(0) if hasattr(batch, 'x') else 0
    
    if edge_index is not None and edge_index.numel() > 0:
        if batch_vec is not None:
            # Map edges to graphs based on source node
            src_nodes = edge_index[0]
            for gid in range(num_graphs):
                node_mask = (batch_vec == gid).nonzero(as_tuple=False).view(-1)
                node_set = set(node_mask.tolist())
                edge_mask = torch.tensor([int(src.item()) in node_set for src in src_nodes], dtype=torch.bool)
                graph_edge_counts[gid] = edge_mask.sum().item()
        else:
            # Single graph case
            graph_edge_counts[0] = edge_index.size(1)
    
    for gid in topk_indices_per_graph:
        if gid >= len(pcst_nodes_list):
            continue
            
        topk_indices = set(topk_indices_per_graph[gid])
        pcst_nodes = pcst_nodes_list[gid]
        
        # Convert pcst_nodes to set of indices (they should already be indices in the local graph)
        if isinstance(pcst_nodes, (list, tuple)):
            pcst_set = set(int(n) for n in pcst_nodes)
        else:
            pcst_set = {int(pcst_nodes)}
        
        intersection = topk_indices & pcst_set
        total_intersection += len(intersection)
        total_k += len(topk_indices)
        total_pcst += len(pcst_set)
    
    if total_k > 0 and total_pcst > 0:
        # Build log message with epoch/step info
        prefix = ""
        if epoch is not None:
            prefix = f"Epoch {epoch}"
            if step is not None:
                prefix += f", Step {step}"
        elif step is not None:
            prefix = f"Step {step}"
        
        if prefix:
            prefix += ": "
        
        # Build graph stats string
        graph_stats_parts = []
        for gid in sorted(graph_node_counts.keys()):
            num_nodes = graph_node_counts.get(gid, 0)
            num_edges = graph_edge_counts.get(gid, 0)
            graph_stats_parts.append(f"Graph {gid}: {num_nodes} nodes, {num_edges} edges")
        graph_stats_str = "; ".join(graph_stats_parts)
        
        log_msg = (f"{prefix}GNAN-PCST Intersection: {total_intersection}/{total_k} of top-k nodes, "
                   f"{total_intersection}/{total_pcst} of PCST nodes "
                   f"(k={total_k}, |PCST|={total_pcst})\n"
                   f"Graph stats: {graph_stats_str}\n")
        
        # Write to file
        with open(log_file, 'a') as f:
            f.write(log_msg)


def build_augmented_desc(model, batch):
    """
    Builds description for the model context.
    - If use_full_graph_context is True: returns original desc unchanged
    - If use_full_graph_context is False: creates NEW desc from PCST/GNAN content only
    """
    base_desc_list = _safe_listify(batch.desc)
    num_graphs = len(base_desc_list)

    # If using full graph context, return original desc unchanged
    use_full_graph = getattr(args, 'use_full_graph_context', False)
    if use_full_graph:
        return [d if isinstance(d, str) else str(d) for d in base_desc_list]

    # Otherwise, create NEW description from PCST/GNAN content only
    new_desc = [""] * num_graphs  # Start with empty descriptions
    
    add_pcst_desc = getattr(args, 'include_pcst_desc_context', False)
    topk = getattr(args, 'topk_gnan_nodes_context', 0)

    # Add PCST textual description if available
    if add_pcst_desc:
        pcst_desc_attr = getattr(batch, 'pcst_desc', None)
        if pcst_desc_attr is not None:
            pcst_desc_list = _safe_listify(pcst_desc_attr)
            for i in range(num_graphs):
                pcst_desc = pcst_desc_list[i] if i < len(pcst_desc_list) else ""
                if isinstance(pcst_desc, str) and pcst_desc:
                    if new_desc[i]:
                        new_desc[i] = f"{new_desc[i]}\n\nPCST_DESC:\n{pcst_desc}"
                    else:
                        new_desc[i] = f"PCST_DESC:\n{pcst_desc}"

    # Add GNAN top-k node indices if requested
    if topk and topk > 0 and hasattr(model, 'gnn') and hasattr(model.gnn, 'node_importance'):
        try:
            device =  next(model.gnn.parameters()).device
            data_for_importance = batch.to(device)
            topk_indices_per_graph = {}
            with torch.no_grad():
                contrib = model.gnn.node_importance(data_for_importance)
                scores = contrib.sum(dim=1)
                batch_vec = getattr(data_for_importance, 'batch', None)
                if batch_vec is None:
                    batch_vec = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
                for gid in range(num_graphs):
                    node_mask = (batch_vec == gid).nonzero(as_tuple=False).view(-1)
                    if node_mask.numel() == 0:
                        continue
                    local_scores = scores[node_mask]
                    k = min(int(topk), local_scores.numel())
                    top_vals, top_idx_local = torch.topk(local_scores, k=k)
                    top_idx_global = node_mask[top_idx_local]
                    # Node ID mapping is not available on Data; include node indices
                    idx_list = top_idx_global.detach().cpu().tolist()
                    topk_indices_per_graph[gid] = idx_list
                    idx_str = ','.join(str(int(x)) for x in idx_list)
                    if new_desc[gid]:
                        new_desc[gid] = f"{new_desc[gid]}\n\nIMPORTANT_NODES_FROM_GNN: {idx_str}"
                    else:
                        new_desc[gid] = f"IMPORTANT_NODES_FROM_GNN: {idx_str}"
            
            # Log intersection statistics with PCST nodes
            if _intersection_log_file is not None:
                _compute_and_print_pcst_gnan_intersection(
                    batch, topk_indices_per_graph, _intersection_log_file, 
                    epoch=_current_epoch, step=_current_step
                )
        except Exception:
            # If anything goes wrong, just return what we have so far
            return new_desc

    return new_desc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn_hidden_channels', type=int, default=1536)
    parser.add_argument('--num_gnn_layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--checkpointing', action='store_true')
    parser.add_argument('--llama_version', type=str, required=True)
    parser.add_argument('--retrieval_config_version', type=int, required=True)
    parser.add_argument('--algo_config_version', type=int, required=True)
    parser.add_argument('--g_retriever_config_version', type=int, required=True)
    parser.add_argument('--freeze_llm', type=bool, default=False)
    parser.add_argument('--print_node_description', action='store_true')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--include_pcst_desc_context', action='store_true',
                        help='Append PCST textual description to the context passed to GRetriever.')
    parser.add_argument('--topk_gnan_nodes_context', type=int, default=0,
                        help='Append top-k GNAN-important node indices per graph to the context. 0 to disable.')
    parser.add_argument('--save_node_importance', action='store_true',
                        help='Save all node importance scores from GNAN to JSON during final test evaluation.')
    parser.add_argument('--use_full_graph_context', action='store_true',
                        help='Use full graph in context without augmentation. Overrides PCST and GNAN augmentation flags.')
    args = parser.parse_args()
    load_dotenv('db.env', override=True)

    start_time = time.time()
    train(
        args.epochs,
        args.gnn_hidden_channels,
        args.num_gnn_layers,
        args.batch_size,
        args.eval_batch_size,
        args.lr,
        llama_version=args.llama_version,
        retrieval_config_version=args.retrieval_config_version,
        algo_config_version=args.algo_config_version,
        g_retriever_config_version=args.g_retriever_config_version,
        checkpointing=args.checkpointing,
        sys_prompt="help answer the user question as best as you can",
        num_gpus=1,
        print_node_description=args.print_node_description,
        load_model_path=args.load_model_path,
    )
    print(f"Total Time: {time.time() - start_time:2f}s")

