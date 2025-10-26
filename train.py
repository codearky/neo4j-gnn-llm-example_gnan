import argparse
import math
import os
import time
import io
import pandas as pd

import torch
from dotenv import load_dotenv
from torch_geometric.loader import DataLoader
from torch_geometric.loader.gnan_dataloader import GNANDataLoader


from stark_qa import load_qa
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch_geometric import seed_everything
from torch_geometric.nn import GRetriever, TensorGNAN
from torch_geometric.nn.nlp import LLM
from tqdm import tqdm

from compute_metrics import compute_metrics

from STaRKQADatasetGDS import STaRKQADataset
from STaRKQAVectorSearchDataset import STaRKQAVectorSearchDataset
from torch_geometric.transforms.gnan import PreprocessDistances


def get_loss(model, batch, model_save_name) -> Tensor:
    if model_save_name.startswith('llm'):
        return model(batch.question, batch.label, batch.desc)
    else:
        # calls forward for GRetriever
        return model(batch, batch.question, batch.x, batch.edge_index, batch.batch,
                     batch.label, batch.edge_attr, batch.desc)



def inference_step(model, batch, model_save_name):
    if model_save_name.startswith('llm'):
        return model.inference(batch.question, batch.desc)
    else:
        return model.inference(batch, batch.question, batch.x, batch.edge_index,
                               batch.batch, batch.edge_attr, batch.desc)



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


def load_params_dict(model, save_path):
    state_dict = model.state_dict()
    state_dict.update(torch.load(save_path)) #All weights might not be saved, eg when using LoRA.
    model.load_state_dict(state_dict)
    return model



def print_node_importance_examples(model, test_dataset, print_node_description: bool, num_examples: int = 3):
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
        train_dataset = STaRKQADataset(root_path, qa_raw_train, retrieval_config_version, algo_config_version, split="train", transform=PreprocessDistances())
        print(f'Finished loading train dataset in {time.time() - t} seconds.')
        print("Loading stark-qa prime val dataset...")
        val_dataset = STaRKQADataset(root_path, qa_raw_val, retrieval_config_version, algo_config_version, split="val", transform=PreprocessDistances())
        print("Loading stark-qa prime test dataset...")
        test_dataset = STaRKQADataset(root_path, qa_raw_test, retrieval_config_version, algo_config_version, split="test", transform=PreprocessDistances())
        os.makedirs(f'{root_path}/models', exist_ok=True)

    train_loader = GNANDataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, pin_memory=True, shuffle=True,)
    val_loader = GNANDataLoader(val_dataset, batch_size=eval_batch_size,
                            drop_last=False, pin_memory=True, shuffle=False )
    test_loader = GNANDataLoader(test_dataset, batch_size=eval_batch_size,
                             drop_last=False, pin_memory=True, shuffle=False)

    gnn = TensorGNAN(
        in_channels=1536,
        hidden_channels=hidden_channels,
        out_channels=1536,
        n_layers=num_gnn_layers,
        normalize_rho=True,
        feature_groups=[list(range(1536))],
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
    grad_steps = 2

    best_epoch = 0
    best_val_loss = float('inf')
    if load_model_path is None:
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            if epoch == 0:
                print(f"Total Preparation Time: {time.time() - start_time:2f}s")
                start_time = time.time()
                print("Training beginning...")
            epoch_str = f'Epoch: {epoch + 1}|{num_epochs}'
            loader = tqdm(train_loader, desc=epoch_str)

            for step, batch in enumerate(loader):
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
        progress_bar_test.update(1)

    compute_metrics(eval_output)
    # Permuted-topk evaluation for GNAN models
    # if num_gnn_layers > 0:
        # permuted_eval_output = evaluate_with_permuted_topk_node_features(model, test_dataset, topk=10)
        # print("\nPermuted-top-10 metrics:")
        # compute_metrics(eval_output)

    print(f"Total Training Time: {time.time() - start_time:2f}s")
    save_params_dict(model, f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}.pt')
    torch.save(eval_output, f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}_eval_outs.pt')
    if num_gnn_layers > 0:
        torch.save(permuted_eval_output, f'{root_path}/models/{retrieval_config_version}_{algo_config_version}_{g_retriever_config_version}_{model_save_name}_perm_top10_eval_outs.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn_hidden_channels', type=int, default=1536)
    parser.add_argument('--num_gnn_layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--checkpointing', action='store_true')
    parser.add_argument('--llama_version', type=str, required=True)
    parser.add_argument('--retrieval_config_version', type=int, required=True)
    parser.add_argument('--algo_config_version', type=int, required=True)
    parser.add_argument('--g_retriever_config_version', type=int, required=True)
    parser.add_argument('--freeze_llm', type=bool, default=False)
    parser.add_argument('--print_node_description', action='store_true')
    parser.add_argument('--load_model_path', type=str, default=None)
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

