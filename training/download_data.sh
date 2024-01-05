#!/bin/bash

TO_DOWNLOAD=100
TOTAL=$(printf "%05d" 153)

mkdir -p "data/raw/the-stack-dedup/php"

for i in $(seq -f "%05g" 0 $((TO_DOWNLOAD - 1))); do
    wget --header="Authorization: Bearer $HUGGINGFACE_TOKEN" \
    -O "../data/raw/the-stack-dedup/php/data-$i-of-$TOTAL.parquet" \
    "https://huggingface.co/datasets/bigcode/the-stack-dedup/resolve/main/data/php/data-$i-of-$TOTAL.parquet"
done
