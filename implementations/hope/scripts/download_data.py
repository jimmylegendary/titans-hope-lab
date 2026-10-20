"""Download the FineWeb-Edu sample (Penedo et al., 2024) used for byte-level
pretraining, matching the data regime of the Titans/ATLAS/HOPE paper family
(they pretrain on FineWeb-Edu; we use the public 100BT sample shards).

Usage:
    python scripts/download_data.py --out data/fineweb-edu --shards 3

Each parquet shard is ~2.2 GB and contains ~0.7B tokens worth of text.
Three shards are more than enough for the 0.3B-token byte recipe in
scripts/train_byte.py (it reads the FIRST shard only and holds out its last 2M
bytes for validation). The last-shard-as-validation split applies to the
BPE recipe (scripts/train_bpe.py), which streams all shards.
"""

import argparse
import os

from huggingface_hub import hf_hub_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fineweb-edu")
    ap.add_argument("--shards", type=int, default=3,
                    help="number of parquet shards to download (>= 2)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Filenames in HuggingFaceFW/fineweb-edu, sample/100BT configuration.
    names = [f"sample/100BT/{i:03d}_00000.parquet" for i in range(args.shards)]
    for name in names:
        print(f"downloading {name} ...", flush=True)
        path = hf_hub_download(
            repo_id="HuggingFaceFW/fineweb-edu",
            repo_type="dataset",
            filename=name,
            local_dir=args.out,
        )
        print(f"  -> {path}", flush=True)
    print("done. byte recipe: first shard (last 2M bytes held out); "
          "BPE recipe: train = all but last shard, val = last shard.")


if __name__ == "__main__":
    main()
