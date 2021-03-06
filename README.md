# Sampling and Occlusion

This repository contains scripts to perform Sampling and Occlusion (SOC) proposed in [Jin et al. (2020)](https://openreview.net/forum?id=BkxRRkSKwr), an algorithm for calculating the importance of a phrase in prediction by a model.

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
Transformers library	We are very happy to show you the Transformers library.
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

The output file will be like this:

```tsv
Transformers library	We are very happy to show you the Transformers library.	-0.11680495142936706
happy	We are very happy to show you the Transformers library.	0.383293890953064
```

## Notes

- While the original implementation performs sampling from an LSTM LM, this implementation performs Gibbs sampling from a masked LM (c.f., [Wang and Cho (2019)](https://aclanthology.org/W19-2304/)).

## TODOs

- Handle cases in which a phrase appears multiple times.

## Reference

- [Towards Hierarchical Importance Attribution: Explaining Compositional Semantics for Neural Sequence Models [Jin et al. (2020)]](https://openreview.net/forum?id=BkxRRkSKwr)
- [BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model [Wang and Cho (2019)]](https://aclanthology.org/W19-2304/)
