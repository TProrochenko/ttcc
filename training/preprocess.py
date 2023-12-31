import ast
import io
import multiprocessing
import os
import string
import tokenize
import warnings
from functools import partial

import pandas as pd
from tqdm import tqdm

from args import PreprocessingArgs


def allowed_characters(text):
    return set(text).issubset(set(string.printable))


def ast_parsable(text):
    with warnings.catch_warnings():
        warnings.simplefilter(action="error", category=SyntaxWarning)
        try:
            ast.parse(text)
            return True
        except (SyntaxError, ValueError, SyntaxWarning):
            return False


def python_tokenize(text):
    try:
        bytes_io = io.BytesIO(text.encode("utf-8"))
        tokens = tokenize.tokenize(bytes_io.readline)
        return len(list(tokens))
    except (SyntaxError, LookupError, UnicodeDecodeError, tokenize.TokenError):
        return -1


def preprocess_chunk(args: PreprocessingArgs, lang, filename) -> None:
    df = pd.read_parquet(f"{args.source_path}/{lang}/{filename}")
    if args.filter_ext:
        df = df.loc[df["ext"].isin(args.valid_ext)]

    if args.filter_size:
        df = df.loc[
            df["size"].between(
                df["size"].quantile(args.min_size), df["size"].quantile(args.max_size)
            )
        ]

    if args.filter_max_line_length:
        df = df.loc[
            df["max_line_length"].between(
                df["max_line_length"].quantile(args.min_max_line_length),
                df["max_line_length"].quantile(args.max_max_line_length),
            )
        ]

    if args.filter_avg_line_length:
        df = df.loc[
            df["avg_line_length"].between(
                df["avg_line_length"].quantile(args.min_avg_line_length),
                df["avg_line_length"].quantile(args.max_avg_line_length),
            )
        ]

    if args.filter_valid_chars:
        df = df.loc[df["content"].apply(lambda x: allowed_characters(x))]

    if args.filter_ast_parsable:
        df = df.loc[df["content"].apply(lambda x: ast_parsable(x))]

    df = df.set_index("hexsha").loc[:, ["content"]]

    df.to_parquet(f"{args.output_path}/{lang}/{filename}")


def preprocess(args: PreprocessingArgs) -> None:
    for lang in args.langs:
        filenames = sorted(os.listdir(f"{args.source_path}/{lang}"))

        with multiprocessing.Pool(processes=3) as pool:
            partial_preprocess_chunk = partial(preprocess_chunk, args, lang)
            for _ in tqdm(
                pool.imap_unordered(partial_preprocess_chunk, filenames),
                total=len(filenames),
            ):
                pass


if __name__ == "__main__":
    preprocess(PreprocessingArgs())
