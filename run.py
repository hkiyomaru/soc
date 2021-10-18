import argparse
import logging
import os
import random

import numpy as np
import torch
import tqdm

from gibbs_sampling import run_gibbs_sampling
from soc import run_soc

logger = logging.getLogger(__file__)


def load_inputs(path: str) -> list[tuple[str, str]]:
    inputs = []
    with open(path) as f:
        for i, line in tqdm.tqdm(enumerate(f)):
            if line.strip() == "":
                continue
            phrase, text = line.strip().split("\t")
            assert phrase in text, f"{phrase} not in {text} (line: {i + 1})"
            inputs.append((phrase, text))
    return inputs


def save_outputs(path: str, inputs: list[tuple[str, str]], scores: list[float]) -> None:
    with open(path, "w") as f:
        for (phrase, text), score in zip(inputs, scores):
            f.write(f"{phrase}\t{text}\t{score}\n")


def save_samples(path: str, list_of_samples: list[list[str]]) -> None:
    stem = path
    while True:
        _stem, _ = os.path.splitext(stem)
        if stem == _stem:
            break
        stem = _stem

    with open(stem + ".sample.tsv", "w") as f:
        for samples in list_of_samples:
            f.write("\t".join(samples) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input file.")
    parser.add_argument("--output", required=True, help="Path to output directory.")
    parser.add_argument(
        "--cls", required=True, help="A pretrained sequence classification model."
    )
    parser.add_argument(
        "--mlm", required=True, help="A pretrained masked language model."
    )
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size.")
    parser.add_argument(
        "--max_seq_length", type=int, default=128, help="Max sequence length."
    )
    parser.add_argument("--n", type=int, default=10, help="The size of context region.")
    parser.add_argument("--k", type=int, default=20, help="The number of samples.")
    parser.add_argument("--save_samples", action="store_true", help="Save samples.")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU ID.")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"

    logger.info("Load the input file.")
    inputs = load_inputs(args.input)

    logger.info("Run Gibbs sampling.")
    list_of_samples = run_gibbs_sampling(
        inputs, args.mlm, args.n, args.k, args.batch_size, args.max_seq_length, device
    )

    logger.info("Run SOC.")
    scores = run_soc(
        inputs, list_of_samples, args.cls, args.batch_size, args.max_seq_length, device
    )

    logger.info("Save the results.")
    save_outputs(args.output, inputs, scores)
    if args.save_samples:
        logger.info("Save the sampled texts.")
        save_samples(args.output, list_of_samples)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level="INFO"
    )

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    main()
