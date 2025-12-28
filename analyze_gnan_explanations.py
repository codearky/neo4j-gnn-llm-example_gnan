#!/usr/bin/env python3
"""
Script to analyze GNAN model predictions and generate explanations.

This script loads a trained GNAN model, finds examples where it correctly/incorrectly
predicts answers, extracts important nodes from GNAN, and asks the model to explain
its predictions. It can also augment the context with important nodes.

Usage:
    python analyze_gnan_explanations.py --algo_version 4 --num_correct 2 --num_incorrect 2

Author: yoavk
"""

import argparse
import io
import json
import os
import re
import sys
from datetime import datetime
from typing import Optional

import pandas as pd
import torch
from dotenv import load_dotenv
from stark_qa import load_qa
from torch_geometric import seed_everything
from torch_geometric.nn import GRetriever, TensorGNAN
from torch_geometric.nn.nlp import LLM
from torch_geometric.transforms.gnan import PreprocessDistances
from tqdm import tqdm

from STaRKQADatasetGDS import STaRKQADataset


def load_params_dict(model, save_path):
    """Load model parameters from a checkpoint file."""
    state_dict = model.state_dict()
    checkpoint = torch.load(save_path, map_location='cpu')
    # Handle both full checkpoint and direct state dict formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    state_dict.update(checkpoint)
    model.load_state_dict(state_dict)
    return model


def parse_node_info_from_desc(desc: str):
    """
    Parse node information from the description CSV format.
    Returns a list of dicts with 'name' and 'description' for each node.
    """
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


def check_prediction_correctness(pred: str, label: str) -> dict:
    """
    Check if prediction is correct using various metrics.
    Returns dict with correctness info.
    """
    pred_items = pred.split('[/s]')[0].strip().split('|')
    label_items = label.split('|')
    
    # Clean up predictions
    pred_items = [p.strip().lower() for p in pred_items if p.strip()]
    label_items = [l.strip().lower() for l in label_items]
    
    matches = set(pred_items).intersection(set(label_items))
    
    exact_hit_at_1 = pred_items[0] in label_items if pred_items else False
    exact_hit_at_any = len(matches) > 0
    precision = len(matches) / len(set(pred_items)) if pred_items else 0
    recall = len(matches) / len(set(label_items)) if label_items else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Also check substring match
    try:
        substring_hit = len(re.findall(pred_items[0], label.lower())) > 0 if pred_items else False
    except:
        substring_hit = False
    
    return {
        'exact_hit_at_1': exact_hit_at_1,
        'exact_hit_at_any': exact_hit_at_any,
        'substring_hit': substring_hit,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'is_correct': exact_hit_at_any,  # Primary correctness criterion
        'pred_items': pred_items,
        'label_items': label_items,
        'matches': list(matches),
    }


def get_node_importance(model, data, topk: int = 10):
    """
    Compute node importance scores using GNAN.
    Returns topk indices and their scores.
    """
    device = next(model.gnn.parameters()).device
    data = data.to(device)
    
    with torch.no_grad():
        contrib = model.gnn.node_importance(data)
        scores = contrib.sum(dim=1)  # Sum over channels
        
        k = min(topk, scores.numel())
        top_vals, top_idx = torch.topk(scores, k=k)
        
    return top_idx.cpu().tolist(), top_vals.cpu().tolist()


def generate_explanation_prompt(question: str, prediction: str, label: str, 
                               important_nodes: Optional[list] = None,
                               include_nodes_in_context: bool = False) -> str:
    """
    Generate a prompt asking the model to explain its prediction.
    """
    clean_question = question.replace('Question: ', '').replace('Answer: ', '').strip()
    
    if include_nodes_in_context and important_nodes:
        nodes_text = "\n".join([
            f"  {i+1}. {node['name']}" + (f" - {node['description'][:200]}..." if node['description'] and len(node['description']) > 200 else (f" - {node['description']}" if node['description'] else ""))
            for i, node in enumerate(important_nodes[:10])
        ])
        prompt = f"""Question: {clean_question}

The Graph Neural Network identified these as the most important entities for answering this question:
{nodes_text}

Based on the question and these important entities, explain:
1. What is the likely answer to the question?
2. How do these important entities relate to finding the answer?
3. Which of these entities are most helpful for answering the question?

Make sure the explanation is in plain English.

Answer and Explanation: """
    else:
        prompt = f"""Question: {clean_question}

Please answer this question and briefly explain your reasoning. Make sure the explanation is in plain English.

Answer and Explanation: """
    
    return prompt


def generate_llm_explanation(model, prompt: str) -> str:
    """
    Generate an explanation using the LLM.
    """
    device = model.llm.device
    
    # Use the LLM's tokenizer and generate
    with torch.no_grad():
        # Tokenize the prompt
        inputs = model.llm.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(device)
        
        # Get word embeddings
        inputs_embeds = model.llm.word_embedding(inputs.input_ids)
        
        # Generate response
        with model.llm.autocast_context:
            outputs = model.llm.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,
                do_sample=False,
                use_cache=True,
                pad_token_id=model.llm.tokenizer.eos_token_id,
            )
        
        # Decode output
        response = model.llm.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the response if it's included
        if response.startswith(prompt[:100]):
            response = response[len(prompt):].strip()
        
    return response


def run_model_inference(model, data, desc: str = None):
    """
    Run inference on a single data point and return prediction.
    """
    device = model.llm.device
    
    # Prepare inputs
    question = data.question if isinstance(data.question, str) else data.question[0]
    desc_to_use = desc if desc is not None else (data.desc if isinstance(data.desc, str) else data.desc[0])
    
    # Ensure batch attribute exists
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
    
    data = data.to(device)
    
    with torch.no_grad():
        pred_list = model.inference(
            data,
            [question],
            data.x,
            data.edge_index,
            data.batch,
            data.edge_attr,
            [desc_to_use],
        )
    
    return pred_list[0] if isinstance(pred_list, (list, tuple)) else str(pred_list)


def build_augmented_desc_with_important_nodes(original_desc: str, node_info: list, 
                                               top_indices: list, top_scores: list,
                                               topk: int = 10) -> str:
    """
    Augment the description with important nodes from GNAN.
    Adds a section at the beginning highlighting the most important nodes.
    """
    important_nodes_section = "IMPORTANT NODES (identified by Graph Neural Network):\n"
    important_nodes_section += "-" * 50 + "\n"
    
    for rank, (node_idx, score) in enumerate(zip(top_indices[:topk], top_scores[:topk])):
        if node_idx < len(node_info):
            node = node_info[node_idx]
            name = node['name']
            desc = node['description']
            if desc and len(desc) > 150:
                desc = desc[:150] + "..."
            important_nodes_section += f"{rank + 1}. {name}"
            if desc:
                important_nodes_section += f" - {desc}"
            important_nodes_section += f" (importance: {score:.4f})\n"
        else:
            important_nodes_section += f"{rank + 1}. <node {node_idx}> (importance: {score:.4f})\n"
    
    important_nodes_section += "-" * 50 + "\n\n"
    
    return important_nodes_section + original_desc


def run_inference_with_augmented_context(model, data, node_info: list, 
                                         top_indices: list, top_scores: list,
                                         topk_nodes: int = 10) -> str:
    """
    Run inference with context augmented by important nodes.
    """
    original_desc = data.desc if isinstance(data.desc, str) else data.desc[0]
    augmented_desc = build_augmented_desc_with_important_nodes(
        original_desc, node_info, top_indices, top_scores, topk_nodes
    )
    return run_model_inference(model, data, desc=augmented_desc)


def format_example_output(idx: int, data, prediction: str, correctness: dict,
                         top_indices: list, top_scores: list, node_info: list,
                         explanation_without_nodes: str = None,
                         explanation_with_nodes: str = None) -> str:
    """
    Format the output for a single example.
    """
    question = data.question if isinstance(data.question, str) else data.question[0]
    label = data.label if isinstance(data.label, str) else data.label[0]
    
    output = []
    output.append("=" * 80)
    output.append(f"EXAMPLE {idx + 1}")
    output.append("=" * 80)
    output.append(f"\nQuestion: {question}")
    output.append(f"\nPredicted Answer: {prediction}")
    output.append(f"Correct Answer: {label}")
    output.append(f"\nCorrectness: {'✓ CORRECT' if correctness['is_correct'] else '✗ INCORRECT'}")
    output.append(f"  - Exact hit@1: {correctness['exact_hit_at_1']}")
    output.append(f"  - Exact hit@any: {correctness['exact_hit_at_any']}")
    output.append(f"  - Precision: {correctness['precision']:.4f}")
    output.append(f"  - Recall: {correctness['recall']:.4f}")
    output.append(f"  - F1: {correctness['f1']:.4f}")
    
    if correctness['matches']:
        output.append(f"  - Matches: {', '.join(correctness['matches'])}")
    
    output.append(f"\n{'─' * 40}")
    output.append("TOP-10 IMPORTANT NODES (from GNAN):")
    output.append("─" * 40)
    
    for rank, (node_idx, score) in enumerate(zip(top_indices, top_scores)):
        if node_idx < len(node_info):
            node = node_info[node_idx]
            name = node['name']
            desc = node['description']
            if desc and len(desc) > 100:
                desc = desc[:100] + "..."
            node_text = f"{name}"
            if desc:
                node_text += f" | {desc}"
        else:
            node_text = f"<node {node_idx}>"
        output.append(f"  {rank + 1:2d}. [idx={node_idx:3d}] score={score:.4f} | {node_text}")
    
    if explanation_without_nodes:
        output.append(f"\n{'─' * 40}")
        output.append("EXPLANATION (without important nodes in context):")
        output.append("─" * 40)
        output.append(explanation_without_nodes)
    
    if explanation_with_nodes:
        output.append(f"\n{'─' * 40}")
        output.append("EXPLANATION (with important nodes in context):")
        output.append("─" * 40)
        output.append(explanation_with_nodes)
    
    output.append("\n")
    return "\n".join(output)


def analyze_node_importance_quality(top_indices: list, top_scores: list, 
                                   node_info: list, label: str) -> dict:
    """
    Analyze whether the important nodes make sense as explanations.
    Check if any important node names appear in the correct answer.
    """
    label_lower = label.lower()
    label_items = [l.strip() for l in label_lower.split('|')]
    
    matching_nodes = []
    for rank, node_idx in enumerate(top_indices):
        if node_idx < len(node_info):
            node = node_info[node_idx]
            node_name_lower = node['name'].lower()
            
            # Check if node name matches any label item
            for label_item in label_items:
                if node_name_lower in label_item or label_item in node_name_lower:
                    matching_nodes.append({
                        'rank': rank + 1,
                        'node_idx': node_idx,
                        'node_name': node['name'],
                        'matched_label': label_item,
                        'score': top_scores[rank] if rank < len(top_scores) else 0,
                    })
                    break
    
    return {
        'num_matching_nodes': len(matching_nodes),
        'matching_nodes': matching_nodes,
        'answer_in_top10': len(matching_nodes) > 0,
        'answer_in_top5': any(m['rank'] <= 5 for m in matching_nodes),
        'answer_in_top1': any(m['rank'] == 1 for m in matching_nodes),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze GNAN model predictions and generate explanations.'
    )
    parser.add_argument('--algo_version', type=int, default=4,
                        help='Algorithm config version (default: 4)')
    parser.add_argument('--retrieval_version', type=int, default=0,
                        help='Retrieval config version (default: 0)')
    parser.add_argument('--g_retriever_version', type=int, default=0,
                        help='G-Retriever config version (default: 0)')
    parser.add_argument('--llama_version', type=str, default='llama3.1-8b',
                        help='LLaMA version (default: llama3.1-8b)')
    parser.add_argument('--num_correct', type=int, default=2,
                        help='Number of correct predictions to analyze (default: 2)')
    parser.add_argument('--num_incorrect', type=int, default=2,
                        help='Number of incorrect predictions to analyze (default: 2)')
    parser.add_argument('--topk_nodes', type=int, default=10,
                        help='Number of top important nodes to show (default: 10)')
    parser.add_argument('--generate_explanations', action='store_true',
                        help='Generate LLM explanations for predictions')
    parser.add_argument('--compare_augmented', action='store_true',
                        help='Compare predictions with/without important nodes in context')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path (default: auto-generated)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--max_examples_to_scan', type=int, default=100,
                        help='Maximum number of test examples to scan (default: 100)')
    
    args = parser.parse_args()
    
    load_dotenv('db.env', override=True)
    seed_everything(args.seed)
    
    # Determine paths
    root_path = f"stark_qa_v{args.retrieval_version}_{args.algo_version}"
    model_name = f"{args.retrieval_version}_{args.algo_version}_{args.g_retriever_version}_gnn-llm-{args.llama_version}.pt"
    model_path = os.path.join(root_path, 'models', model_name)
    
    # Check for best val loss checkpoint
    best_ckpt_name = f"{args.retrieval_version}_{args.algo_version}_{args.g_retriever_version}_gnn-llm-{args.llama_version}_best_val_loss_ckpt.pt"
    best_ckpt_path = os.path.join(root_path, 'models', best_ckpt_name)
    if os.path.exists(best_ckpt_path):
        model_path = best_ckpt_path
        print(f"Using best validation checkpoint: {model_path}")
    else:
        print(f"Using model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Available models:")
        models_dir = os.path.join(root_path, 'models')
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.pt'):
                    print(f"  - {f}")
        sys.exit(1)
    
    # Load test dataset
    print(f"\nLoading test dataset from {root_path}...")
    qa_dataset = load_qa("prime")
    qa_raw_test = qa_dataset.get_subset('test')
    test_dataset = STaRKQADataset(
        root_path, qa_raw_test, 
        args.retrieval_version, args.algo_version, 
        split="test", 
        transform=PreprocessDistances()
    )
    print(f"Loaded {len(test_dataset)} test examples")
    
    # Create model architecture
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
        llm = LLM(
            model_name='TinyLlama/TinyLlama-1.1B-Chat-v0.1',
            num_params=1
        )
        model = GRetriever(llm=llm, gnn=gnn, mlp_out_channels=2048)
    elif args.llama_version == 'llama2-7b':
        llm = LLM(
            model_name='meta-llama/Llama-2-7b-chat-hf',
            num_params=7
        )
        model = GRetriever(llm=llm, gnn=gnn)
    elif args.llama_version == 'llama3.1-8b':
        llm = LLM(
            model_name='meta-llama/Llama-3.1-8B-Instruct',
            num_params=8
        )
        model = GRetriever(llm=llm, gnn=gnn)
    else:
        raise ValueError(f"Unknown llama_version: {args.llama_version}")
    
    # Load model weights
    print(f"\nLoading model weights from {model_path}...")
    model = load_params_dict(model, model_path)
    model.eval()
    print("Model loaded successfully!")
    
    # Find correct and incorrect examples
    print(f"\nScanning test set for examples (up to {args.max_examples_to_scan} examples)...")
    correct_examples = []
    incorrect_examples = []
    
    num_to_scan = min(args.max_examples_to_scan, len(test_dataset))
    
    for idx in tqdm(range(num_to_scan), desc="Scanning examples"):
        data = test_dataset[idx]
        
        try:
            # Run inference
            prediction = run_model_inference(model, data)
            label = data.label if isinstance(data.label, str) else data.label[0]
            
            # Check correctness
            correctness = check_prediction_correctness(prediction, label)
            
            example_info = {
                'idx': idx,
                'data': data,
                'prediction': prediction,
                'correctness': correctness,
            }
            
            if correctness['is_correct']:
                if len(correct_examples) < args.num_correct:
                    correct_examples.append(example_info)
            else:
                if len(incorrect_examples) < args.num_incorrect:
                    incorrect_examples.append(example_info)
            
            # Stop early if we have enough examples
            if (len(correct_examples) >= args.num_correct and 
                len(incorrect_examples) >= args.num_incorrect):
                break
                
        except Exception as e:
            print(f"Warning: Error processing example {idx}: {e}")
            continue
    
    print(f"\nFound {len(correct_examples)} correct and {len(incorrect_examples)} incorrect examples")
    
    # Analyze selected examples
    all_examples = [('CORRECT', ex) for ex in correct_examples] + \
                   [('INCORRECT', ex) for ex in incorrect_examples]
    
    results = []
    output_lines = []
    
    output_lines.append("=" * 80)
    output_lines.append("GNAN MODEL EXPLANATION ANALYSIS")
    output_lines.append("=" * 80)
    output_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Model: {model_path}")
    output_lines.append(f"Algorithm version: {args.algo_version}")
    output_lines.append(f"Test set size: {len(test_dataset)}")
    output_lines.append(f"Examples scanned: {num_to_scan}")
    output_lines.append(f"Correct examples found: {len(correct_examples)}")
    output_lines.append(f"Incorrect examples found: {len(incorrect_examples)}")
    output_lines.append("")
    
    for category, ex_info in all_examples:
        idx = ex_info['idx']
        data = ex_info['data']
        prediction = ex_info['prediction']
        correctness = ex_info['correctness']
        
        output_lines.append(f"\n{'#' * 80}")
        output_lines.append(f"# {category} PREDICTION - Example Index: {idx}")
        output_lines.append("#" * 80)
        
        # Get node importance
        top_indices, top_scores = get_node_importance(model, data, args.topk_nodes)
        
        # Parse node info from description
        desc = data.desc if isinstance(data.desc, str) else data.desc[0]
        node_info = parse_node_info_from_desc(desc)
        
        # Analyze if important nodes match the answer
        importance_quality = analyze_node_importance_quality(
            top_indices, top_scores, node_info, 
            data.label if isinstance(data.label, str) else data.label[0]
        )
        
        # Compare predictions with and without augmented context
        prediction_with_augmented = None
        correctness_with_augmented = None
        
        if args.compare_augmented:
            print(f"  Running inference with augmented context for example {idx}...")
            try:
                prediction_with_augmented = run_inference_with_augmented_context(
                    model, data, node_info, top_indices, top_scores, args.topk_nodes
                )
                label = data.label if isinstance(data.label, str) else data.label[0]
                correctness_with_augmented = check_prediction_correctness(
                    prediction_with_augmented, label
                )
            except Exception as e:
                print(f"  Warning: Could not run augmented inference: {e}")
        
        # Generate explanations if requested
        explanation_without_nodes = None
        explanation_with_nodes = None
        
        if args.generate_explanations:
            question = data.question if isinstance(data.question, str) else data.question[0]
            label = data.label if isinstance(data.label, str) else data.label[0]
            
            # Get important nodes info for context
            important_nodes = []
            for node_idx in top_indices[:10]:
                if node_idx < len(node_info):
                    important_nodes.append(node_info[node_idx])
            
            print(f"  Generating LLM explanations for example {idx}...")
            
            # Generate explanation WITHOUT important nodes in context
            prompt_without = generate_explanation_prompt(
                question, prediction, label,
                important_nodes=None,
                include_nodes_in_context=False
            )
            try:
                explanation_without_nodes = generate_llm_explanation(model, prompt_without)
            except Exception as e:
                explanation_without_nodes = f"[Error generating explanation: {e}]"
            
            # Generate explanation WITH important nodes in context
            prompt_with = generate_explanation_prompt(
                question, prediction, label,
                important_nodes=important_nodes,
                include_nodes_in_context=True
            )
            try:
                explanation_with_nodes = generate_llm_explanation(model, prompt_with)
            except Exception as e:
                explanation_with_nodes = f"[Error generating explanation: {e}]"
        
        # Format output
        output = format_example_output(
            idx, data, prediction, correctness,
            top_indices, top_scores, node_info,
            explanation_without_nodes, explanation_with_nodes
        )
        output_lines.append(output)
        
        # Add augmented context comparison if available
        if prediction_with_augmented is not None:
            output_lines.append("─" * 40)
            output_lines.append("PREDICTION WITH AUGMENTED CONTEXT (important nodes added):")
            output_lines.append("─" * 40)
            output_lines.append(f"  Original Prediction: {prediction}")
            output_lines.append(f"  Augmented Prediction: {prediction_with_augmented}")
            output_lines.append(f"  Original Correct: {'✓' if correctness['is_correct'] else '✗'}")
            output_lines.append(f"  Augmented Correct: {'✓' if correctness_with_augmented['is_correct'] else '✗'}")
            
            # Check if augmentation helped/hurt
            if correctness_with_augmented['is_correct'] and not correctness['is_correct']:
                output_lines.append("  ⬆ IMPROVEMENT: Augmentation fixed the prediction!")
            elif not correctness_with_augmented['is_correct'] and correctness['is_correct']:
                output_lines.append("  ⬇ REGRESSION: Augmentation broke a correct prediction!")
            elif prediction_with_augmented != prediction:
                output_lines.append("  ↔ CHANGED: Prediction changed but correctness stayed same")
            else:
                output_lines.append("  = UNCHANGED: Same prediction")
        
        # Add importance quality analysis
        output_lines.append("─" * 40)
        output_lines.append("NODE IMPORTANCE QUALITY ANALYSIS:")
        output_lines.append("─" * 40)
        output_lines.append(f"  - Answer in top-1 nodes: {importance_quality['answer_in_top1']}")
        output_lines.append(f"  - Answer in top-5 nodes: {importance_quality['answer_in_top5']}")
        output_lines.append(f"  - Answer in top-10 nodes: {importance_quality['answer_in_top10']}")
        output_lines.append(f"  - Number of matching nodes: {importance_quality['num_matching_nodes']}")
        
        if importance_quality['matching_nodes']:
            output_lines.append("  - Matching nodes:")
            for match in importance_quality['matching_nodes']:
                output_lines.append(f"      Rank {match['rank']}: '{match['node_name']}' matches '{match['matched_label']}'")
        
        # Store result
        result = {
            'category': category,
            'idx': idx,
            'question': data.question if isinstance(data.question, str) else data.question[0],
            'prediction': prediction,
            'prediction_with_augmented': prediction_with_augmented,
            'label': data.label if isinstance(data.label, str) else data.label[0],
            'correctness': correctness,
            'correctness_with_augmented': correctness_with_augmented,
            'top_indices': top_indices,
            'top_scores': top_scores,
            'importance_quality': importance_quality,
            'top_node_names': [
                node_info[i]['name'] if i < len(node_info) else f'<node {i}>'
                for i in top_indices[:10]
            ],
        }
        results.append(result)
    
    # Summary statistics
    output_lines.append("\n" + "=" * 80)
    output_lines.append("SUMMARY: NODE IMPORTANCE QUALITY")
    output_lines.append("=" * 80)
    
    correct_quality = [r['importance_quality'] for r in results if r['category'] == 'CORRECT']
    incorrect_quality = [r['importance_quality'] for r in results if r['category'] == 'INCORRECT']
    
    if correct_quality:
        output_lines.append("\nFor CORRECT predictions:")
        output_lines.append(f"  - Answer in top-1: {sum(q['answer_in_top1'] for q in correct_quality)}/{len(correct_quality)}")
        output_lines.append(f"  - Answer in top-5: {sum(q['answer_in_top5'] for q in correct_quality)}/{len(correct_quality)}")
        output_lines.append(f"  - Answer in top-10: {sum(q['answer_in_top10'] for q in correct_quality)}/{len(correct_quality)}")
    
    if incorrect_quality:
        output_lines.append("\nFor INCORRECT predictions:")
        output_lines.append(f"  - Answer in top-1: {sum(q['answer_in_top1'] for q in incorrect_quality)}/{len(incorrect_quality)}")
        output_lines.append(f"  - Answer in top-5: {sum(q['answer_in_top5'] for q in incorrect_quality)}/{len(incorrect_quality)}")
        output_lines.append(f"  - Answer in top-10: {sum(q['answer_in_top10'] for q in incorrect_quality)}/{len(incorrect_quality)}")
    
    # Augmented context summary
    if args.compare_augmented:
        output_lines.append("\n" + "=" * 80)
        output_lines.append("SUMMARY: AUGMENTED CONTEXT EFFECT")
        output_lines.append("=" * 80)
        
        improvements = sum(1 for r in results 
                          if r.get('correctness_with_augmented') 
                          and r['correctness_with_augmented']['is_correct'] 
                          and not r['correctness']['is_correct'])
        regressions = sum(1 for r in results 
                         if r.get('correctness_with_augmented')
                         and not r['correctness_with_augmented']['is_correct'] 
                         and r['correctness']['is_correct'])
        unchanged = sum(1 for r in results 
                       if r.get('correctness_with_augmented')
                       and r['correctness_with_augmented']['is_correct'] == r['correctness']['is_correct'])
        
        output_lines.append(f"\n  Improvements (incorrect → correct): {improvements}/{len(results)}")
        output_lines.append(f"  Regressions (correct → incorrect): {regressions}/{len(results)}")
        output_lines.append(f"  Unchanged: {unchanged}/{len(results)}")
    
    # Print output
    full_output = "\n".join(output_lines)
    print(full_output)
    
    # Save output
    if args.output_file is None:
        args.output_file = os.path.join(
            root_path, 'models',
            f'gnan_explanation_analysis_algo{args.algo_version}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        )
    
    with open(args.output_file, 'w') as f:
        f.write(full_output)
    print(f"\nOutput saved to: {args.output_file}")
    
    # Also save JSON results
    json_output_file = args.output_file.replace('.txt', '.json')
    # Convert non-serializable items
    serializable_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != 'data'}
        serializable_results.append(sr)
    
    with open(json_output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    print(f"JSON results saved to: {json_output_file}")


if __name__ == '__main__':
    main()

