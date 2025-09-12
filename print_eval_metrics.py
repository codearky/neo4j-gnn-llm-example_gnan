import argparse
import os

import torch

from compute_metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Load eval outputs and print metrics")
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.join(
            "stark_qa_v0_0",
            "models",
            "0_0_0_gnn-llm-llama3.1-8b_eval_outs.pt",
        ),
        help="Path to the eval outputs .pt file (default points to v0_0 models path)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise FileNotFoundError(f"File not found: {args.path}")

    eval_output = torch.load(args.path, map_location="cpu")
    compute_metrics(eval_output)


if __name__ == "__main__":
    main() 