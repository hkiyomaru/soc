# Sampling and Occlusion

This repository contains scripts to perform Sampling and Occlusion proposed in [Jin et al. (2020)](https://openreview.net/forum?id=BkxRRkSKwr).

## Requirements

- Python: 3.9+
- torch: 1.9+
- transformers: 4.11+
- Others: See [pyproject.toml](pyproject.toml).

## Installation

Use [poetry](https://python-poetry.org/).

```commandline
poetry install
```

## Run

Prepare a tsv input file like [input.tsv in the data directory](./data/input.tsv); each row has a phrase being calculated its importance score (1st column) and a sentence that contains the phrase (2nd column).

```tsv
very	We are very happy to show you the Transformers library.
happy	We are very happy to show you the Transformers library.
```

Run `run.py` to perform SOC.

```commandline
poetry run python run.py \
    --input data/input.tsv \
    --output data/output.tsv \
    --mlm roberta-base \
    --cls textattack/roberta-base-SST-2 \
    --save_samples
```

## Notes

- This implementation performs Gibbs sampling from a masked LM (see [Wang and Cho (2019)](https://aclanthology.org/W19-2304/)). 

## TODOs

- In order to specify a phrase, use its span instead of its surface form.
- Use GPUs to fasten the process.

## Reference

- [Towards Hierarchical Importance Attribution: Explaining Compositional Semantics for Neural Sequence Models [Jin et al. (2020)]](https://openreview.net/forum?id=BkxRRkSKwr)
- [BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model [Wang and Cho (2019)]](https://aclanthology.org/W19-2304/)
