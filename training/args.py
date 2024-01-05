from dataclasses import dataclass


@dataclass
class ModelArgs:
    dim: int = 288
    hidden_dim: int = 768
    multiple_of: int = 32
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 6
    vocab_size: int = 128  # dont change
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0

    total_model_params: int = -1


@dataclass
class TrainingArgs:
    wandb_project: str = "ttcc"
    epochs: int = 1
    warmup_ratio: float = 0.01
    max_lr: float = 5e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    fused: bool = True
    cycle_momentum: bool = True
    grad_clip: float = 1.0
    device: str = "cuda"  # dont change
    dtype: str = "bfloat16"  # dont change
    batch_size: int = 96
    seq_len: int = 2048
    eval_interval: int = 2000
    seed: int = 0
    approx_chunk_tokens: int = 109_000_000  # for scheduler

    steps: int = -1


@dataclass
class PreprocessingArgs:
    langs: tuple[str] = ("javascript",)
    source_path: str = "../data/raw/the-stack-dedup/"
    output_path: str = "../data/processed/"
    filter_ext: bool = False
    valid_ext: tuple[str] = ("",)
    filter_size: bool = True
    max_size: float = 0.95
    min_size: float = 0.01
    filter_max_line_length: bool = True
    max_max_line_length: float = 0.95
    min_max_line_length: float = 0.01
    filter_avg_line_length: bool = False
    max_avg_line_length: float = -1
    min_avg_line_length: float = -1
    filter_valid_chars: bool = True
    filter_ast_parsable: bool = False
