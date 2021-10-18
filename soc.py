import logging

import torch
import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__file__)


class SOCDataset(Dataset):
    def __init__(
        self,
        inputs: list[tuple[str, str]],
        list_of_samples: list[list[str]],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 128,
    ):
        self.flattened_inputs = [
            (phrase, text, sample)
            for (phrase, text), samples in zip(inputs, list_of_samples)
            for sample in samples
        ]

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.flattened_inputs)

    def __getitem__(self, index: int):
        phrase, _, sample = self.flattened_inputs[index]
        features_s = self.tokenizer(
            sample,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        features_soc = self.tokenizer(
            sample.replace(phrase, self.tokenizer.pad_token, 1),
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        features = {
            "input_ids_s": features_s["input_ids"],
            "attention_mask_s": features_s["attention_mask"],
            "input_ids_soc": features_soc["input_ids"],
            "attention_mask_soc": features_soc["attention_mask"],
        }
        return {name: torch.LongTensor(feature) for name, feature in features.items()}


def run_soc(
    inputs: list[tuple[str, str]],
    list_of_samples: list[list[str]],
    model_name_or_path: str,
    batch_size: int,
    max_seq_length: int,
    device: str,
) -> list[float]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    cls = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    cls.eval()
    cls.to(device)

    dataset = SOCDataset(inputs, list_of_samples, tokenizer, max_seq_length)
    loader = DataLoader(
        dataset, batch_size // 2
    )  # because SOC uses two inputs for one instance

    _scores = []
    for batch in tqdm.tqdm(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            scores_s = cls(
                input_ids=batch["input_ids_s"], attention_mask=batch["attention_mask_s"]
            ).logits[:, 1]
            scores_soc = cls(
                input_ids=batch["input_ids_soc"],
                attention_mask=batch["attention_mask_soc"],
            ).logits[:, 1]
        _scores.extend((scores_s - scores_soc).tolist())

    scores = []
    i = 0
    for samples in list_of_samples:
        scores.append(sum(_scores[i : i + len(samples)]) / len(samples))
        i += len(samples)

    assert len(scores) == len(inputs)
    assert i == len(_scores)

    return scores
