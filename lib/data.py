import torch
import torch.utils.data.distributed
from torch.utils.data import DataLoader, IterableDataset

from collections.abc import Callable, Iterator
from typing import Optional
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class TextDataset(IterableDataset):

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        block_size: int,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        buffer: list[int] = []
        indices = torch.arange(len(self.dataset))
        for i in indices:
            example = self.dataset[i.unsqueeze(0)]
            if not example or not example.get('text'):
                continue
            assert isinstance(example['text'], list)
            tokens = self.tokenizer.encode(
                example['text'][0], add_special_tokens=False
            )
            if not tokens:
                continue
            buffer.extend(tokens)
            while len(buffer) >= self.block_size + 1:
                x = torch.tensor(buffer[: self.block_size], dtype=torch.long)
                y = torch.tensor(
                    buffer[1 : self.block_size + 1], dtype=torch.long
                )
                buffer = buffer[self.block_size :]
                yield x, y

    def __len__(self) -> int:
        return len(
            self.dataset
        )  # incorrect estimation! (it should be # tokens/block size)


class OpenWebTextDataset(TextDataset):

    def __init__(
        self,
        root: str,
        tokenizer: PreTrainedTokenizerBase,
        train: bool = True,
        block_size: int = 256,
    ) -> None:
        split = 'train' if train else 'test'
        dataset = load_dataset('Skylion007/openwebtext')
        dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)[
            split
        ]
        super(OpenWebTextDataset, self).__init__(dataset, tokenizer, block_size)


# Datasets tokenized with the GPT-2 tokenizer, by --data.name.
TEXT_DATASETS = {
    'openwebtext': OpenWebTextDataset,
}


def load_datasets(
    name: str, data_dir: str, block_size: int
) -> tuple[IterableDataset, IterableDataset, int]:
    """Loads the train and validation splits of a dataset.

    Args:
        name: A key of TEXT_DATASETS.
        data_dir: The dataset root directory.
        block_size: The sequence length of each example.

    Returns:
        A (train dataset, validation dataset, vocabulary size) tuple.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    if name not in TEXT_DATASETS:
        raise ValueError(
            f'Invalid dataset name {name!r}; known: '
            f"{', '.join(TEXT_DATASETS)}"
        )
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Pad with the EOS token.
    cls = TEXT_DATASETS[name]
    train = cls(
        data_dir, tokenizer=tokenizer, train=True, block_size=block_size
    )
    val = cls(data_dir, tokenizer=tokenizer, train=False, block_size=block_size)
    return train, val, tokenizer.vocab_size


def make_loaders(
    train_dataset: IterableDataset,
    val_dataset: IterableDataset,
    batch_size: int,
    eval_batch_size: int,
    workers: int,
    rank: Optional[int],
    first_stage_ranks: int,
    worker_init_fn: Callable[[int], None],
) -> tuple[
    DataLoader,
    DataLoader,
    Optional[torch.utils.data.distributed.DistributedSampler],
]:
    """Builds the train and validation loaders.

    When the first stage is replicated (first_stage_ranks > 1), each replica
    reads its own shard through a DistributedSampler.

    Args:
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        batch_size: The training mini-batch size.
        eval_batch_size: The eval mini-batch size.
        workers: Data-loading worker processes.
        rank: This process's rank.
        first_stage_ranks: The number of ranks running the first stage.
        worker_init_fn: Seeds each loader worker.

    Returns:
        A (train loader, validation loader, train sampler) tuple; the sampler
        is None unless the first stage is replicated (call its set_epoch()
        every epoch).
    """
    train_sampler = val_sampler = None
    if first_stage_ranks > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=first_stage_ranks, rank=rank
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=first_stage_ranks, rank=rank
        )

    def loader(
        dataset: IterableDataset,
        batch: int,
        sampler: Optional[torch.utils.data.distributed.DistributedSampler],
    ) -> DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch,
            num_workers=workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )

    return (
        loader(train_dataset, batch_size, train_sampler),
        loader(val_dataset, eval_batch_size, val_sampler),
        train_sampler,
    )


class DataUtil:

    def __init__(
        self,
        train_loader: Optional[DataLoader],
        eval_loader: Optional[DataLoader],
    ) -> None:
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.train_loader_iter: Optional[
            Iterator[tuple[torch.Tensor, torch.Tensor]]
        ] = (iter(train_loader) if train_loader is not None else None)
        self.eval_loader_iter: Optional[
            Iterator[tuple[torch.Tensor, torch.Tensor]]
        ] = (iter(eval_loader) if eval_loader is not None else None)

    def get_batch(
        self, eval: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not eval or self.eval_loader_iter is None:
            try:
                x, y = next(self.train_loader_iter)
            except StopIteration:
                self.train_loader_iter = iter(self.train_loader)
                x, y = next(self.train_loader_iter)
        else:
            try:
                x, y = next(self.eval_loader_iter)
            except StopIteration:
                self.eval_loader_iter = iter(self.eval_loader)
                x, y = next(self.eval_loader_iter)

        return x, y
