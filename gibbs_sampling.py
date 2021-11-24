import logging
import random
import typing

import torch
import tqdm
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__file__)

special_token_ids: list[int] = []


def set_special_token_ids(tokenizer: PreTrainedTokenizer) -> None:
    global special_token_ids
    for special_token in tokenizer.special_tokens_map.values():
        special_token_id = tokenizer.get_vocab()[special_token]
        if special_token_id not in special_token_ids:
            special_token_ids.append(special_token_id)
    special_token_ids.sort()


class GibbsSamplingDataset(Dataset):
    def __init__(
        self,
        inputs: list[tuple[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 128,
        n: int = 10,
    ) -> None:
        self.texts = [text for _, text in inputs]
        self.phrases = [phrase for phrase, _ in inputs]
        self._list_of_samples: list[list[str]] = [[text] for _, text in inputs]

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.n = n

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        text = self._list_of_samples[index][-1]
        phrase = self.phrases[index]

        # Mask the phrase with the SEP special token
        text = text.replace(phrase, self.tokenizer.sep_token, 1)

        # Convert the text into token IDs
        features = self.tokenizer(text)
        features = {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"],
        }
        phrase_features = self.tokenizer(phrase, add_special_tokens=False)

        # get ID to mask
        sep_id = features["input_ids"].index(self.tokenizer.sep_token_id)
        mask_id_cands = [
            i
            for i, id_ in enumerate(features["input_ids"])
            if (
                i < self.max_seq_length
                and id_ not in special_token_ids
                and sep_id - self.n <= i <= sep_id + self.n
            )
        ]
        mask_id = random.choice(mask_id_cands) if mask_id_cands else -1

        # Replace IDs
        if mask_id >= 0:
            features["input_ids"][mask_id] = self.tokenizer.mask_token_id
        features["input_ids"] = (
            features["input_ids"][:sep_id]
            + phrase_features["input_ids"]
            + features["input_ids"][sep_id + 1 :]
        )
        features["attention_mask"] = (
            features["attention_mask"][:sep_id]
            + phrase_features["attention_mask"]
            + features["attention_mask"][sep_id + 1 :]
        )

        # Truncate
        features["input_ids"] = features["input_ids"][: self.max_seq_length]
        features["attention_mask"] = features["attention_mask"][: self.max_seq_length]
        num_pad_tokens = self.max_seq_length - len(features["input_ids"])
        features["input_ids"] += [self.tokenizer.pad_token_id] * num_pad_tokens
        features["attention_mask"] += [0] * num_pad_tokens
        features["mask_id"] = [mask_id]
        return {name: torch.LongTensor(feature) for name, feature in features.items()}

    @property
    def list_of_samples(self) -> list[list[str]]:
        return [samples[1:] for samples in self._list_of_samples]

    def add_samples(self, samples: list[str]) -> None:
        assert len(self._list_of_samples) == len(samples)
        for i, sample in enumerate(samples):
            self._list_of_samples[i].append(sample)


def run_gibbs_sampling_step(
    mlm, loader: DataLoader, tokenizer: PreTrainedTokenizer, device: str
) -> list[str]:
    samples = []
    for batch in tqdm.tqdm(loader):  # type: dict[str, torch.Tensor]
        batch = {name: tensor.to(device) for name, tensor in batch.items()}

        mask_id = batch["mask_id"].view(-1)
        batch_size = len(mask_id)

        sample_input_ids = batch["input_ids"].clone()
        if (mask_id != -1).sum().item() != 0:
            with torch.no_grad():
                outputs = mlm(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits = outputs.logits
                logits[
                    :, :, special_token_ids
                ] -= 128.0  # prevent the model from producing a special token
                indices_with_mask_0 = torch.arange(batch_size)[mask_id != -1]
                indices_with_mask_1 = mask_id[mask_id != -1]
                sample_input_ids[
                    indices_with_mask_0, indices_with_mask_1
                ] = Categorical(
                    logits=logits[indices_with_mask_0, indices_with_mask_1]
                ).sample()
        samples.extend(
            tokenizer.batch_decode(sample_input_ids, skip_special_tokens=True)
        )
    return samples


def run_gibbs_sampling(
    inputs: list[tuple[str, str]],
    model_name_or_path: str,
    n: int,
    k: int,
    batch_size: int,
    max_seq_length: int,
    device: str,
) -> list[list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    set_special_token_ids(tokenizer)

    mlm = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    mlm.eval()
    mlm.to(device)

    dataset = GibbsSamplingDataset(inputs, tokenizer, max_seq_length, n)
    loader = DataLoader(dataset, batch_size)

    for _ in tqdm.tqdm(range(k)):
        samples = run_gibbs_sampling_step(mlm, loader, tokenizer, device)
        dataset.add_samples(samples)
    return dataset.list_of_samples
