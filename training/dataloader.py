import os
import random
from typing import Optional

import pandas as pd
import torch

from args import TrainingArgs
from tokenizer import Tokenizer


class DataLoader:
    def __init__(
        self,
        args: TrainingArgs,
        dataset_dir: str = "",
        shuffle: bool = True,
        seq_len: Optional[int] = None,
        epochs: Optional[int] = None,
    ) -> None:
        self.seq_len = seq_len if seq_len else args.seq_len
        self.batch_size = args.batch_size
        self.batch_tokens = self.seq_len * self.batch_size
        self.device = args.device
        self.shuffle = shuffle
        self.dataset_dir = dataset_dir
        self.chunk_names = sorted(os.listdir(dataset_dir))
        self.seed = args.seed
        self.epochs = epochs if epochs else args.epochs
        self.approx_chunk_tokens = args.approx_chunk_tokens
        self.tokenizer = Tokenizer()

    def __iter__(self) -> ((torch.tensor, torch.tensor), dict):
        step = 0
        for epoch in range(self.epochs):
            tokens = torch.empty(size=[0], dtype=torch.uint8)
            for chunk_num, chunk_name in enumerate(self.chunk_names):
                series = pd.read_parquet(f"{self.dataset_dir}/{chunk_name}")["content"]
                series = series.apply(lambda s: self.tokenizer.encode(s))
                samples = [torch.tensor(sample, dtype=torch.uint8) for sample in series]
                del series

                if self.shuffle:
                    random.seed(self.seed)
                    random.shuffle(samples)

                tokens = torch.cat([tokens, torch.cat(samples)])
                del samples

                n_chunk_full_batches = (len(tokens) - 1) // self.batch_tokens
                for chunk_batch_num in range(n_chunk_full_batches):
                    batch_start = chunk_batch_num * self.batch_tokens
                    batch_end = (chunk_batch_num + 1) * self.batch_tokens + 1
                    batch = tokens[batch_start:batch_end]

                    x = batch[:-1].view(self.batch_size, self.seq_len)
                    y = batch[1:].view(self.batch_size, self.seq_len)
                    x = x.to(dtype=torch.int64)
                    y = y.to(dtype=torch.int64)

                    if self.device:
                        x = x.to(self.device)
                        y = y.to(self.device)

                    batch = x, y
                    stats = {
                        "epoch": epoch,
                        "chunk_num": chunk_num,
                        "step": step,
                        "tokens": step * self.batch_tokens,
                    }
                    yield batch, stats
                    step += 1

                n_chunk_used_tokens = n_chunk_full_batches * self.batch_tokens
                tokens = tokens[n_chunk_used_tokens:]

    def __len__(self):
        total_tokens = self.approx_chunk_tokens * len(self.chunk_names) * self.epochs
        return total_tokens // self.batch_tokens
