"""Build the training corpus: stream FineWeb-Edu, train a 16,384 BPE tokenizer on it,
then tokenize the stream into flat uint16 shards.

Why streaming: FineWeb-Edu's sample-10BT is ~27 GB on disk for 10B tokens. A
compute-optimal run for 49.3M parameters is ~1B tokens, and even a heavily
over-trained run is ~10B, so downloading the whole sample first would waste hours
of wall clock before the GPU sees a single batch. Streaming takes only what we ask for.

Why uint16: the vocabulary is 16,384, which fits in 16 bits with room to spare. At
2 bytes per token a 2B-token corpus is 4 GB on disk instead of 8.

Run:
    python tools/prepare_data.py --tokens 2000000000   # 19 train shards + 1 val shard
    python tools/prepare_data.py --tokenizer-only      # just fit the vocabulary
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import GPTConfig  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TOKENIZER_PATH = os.path.join(DATA, "tokenizer.json")

DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
SHARD_TOKENS = 100_000_000        # 100M tokens per shard = 200 MB on disk


def stream_docs(limit=None):
    """Yield raw text from FineWeb-Edu without downloading the whole sample."""
    from datasets import load_dataset
    ds = load_dataset(DATASET, name=CONFIG, split="train", streaming=True)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            return
        text = row.get("text")
        if text:
            yield text


def train_tokenizer(vocab_size, sample_docs):
    """Fit a byte-level BPE on the same corpus the model trains on.

    Byte-level so there is no out-of-vocabulary case and no unknown token to waste an
    id on: every input is representable, which matters when the whole vocabulary is
    16,384 rather than 50,257.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    print(f"fitting a {vocab_size}-token BPE on {sample_docs:,} documents")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    t0 = time.time()
    tok.train_from_iterator(stream_docs(limit=sample_docs), trainer=trainer)
    os.makedirs(DATA, exist_ok=True)
    tok.save(TOKENIZER_PATH)
    print(f"tokenizer -> {TOKENIZER_PATH}  ({time.time()-t0:.0f}s, "
          f"{tok.get_vocab_size():,} tokens)")
    return tok


def load_tokenizer():
    from tokenizers import Tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        return None
    return Tokenizer.from_file(TOKENIZER_PATH)


def tokenize(tok, target_tokens, val_shards=1, overwrite=False):
    """Write flat uint16 shards, holding the LAST val_shards back for validation.

    The split is made here rather than afterwards for one reason: the stream always
    restarts FineWeb-Edu at row 0, so splitting by hand after the fact means a later
    re-run happily writes the validation documents back out as training data. That turns
    val_loss into a restatement of train_loss without anything visibly breaking.

    Held out by position, not at random: the validation shard is the tail of the stream,
    so it is whole documents the model has never seen rather than sentences sampled from
    documents it has. Only whole shards are written -- the partial tail is discarded --
    so a given --tokens always produces a byte-identical corpus.
    """
    eot = tok.token_to_id("<|endoftext|>")
    assert eot is not None, "the end-of-text token is missing from the vocabulary"

    if val_shards < 1:
        raise SystemExit(f"--val-shards must be at least 1 (got {val_shards}): with nothing "
                         f"held out, val_loss is just train_loss under another name")
    total_shards = target_tokens // SHARD_TOKENS
    if total_shards <= val_shards:
        raise SystemExit(f"--tokens {target_tokens:,} is only {total_shards} shard(s); "
                         f"need more than the {val_shards} held out for validation")
    first_val = total_shards - val_shards

    # Refuse to write into a directory that already holds shards. The stream restarts at
    # row 0, so a smaller rebuild over a larger one leaves old train shards behind -- and
    # one of those is byte-identical to the new validation shard, which train.py's
    # train_*.bin glob would then train on. --overwrite clears them first.
    existing = sorted(glob.glob(os.path.join(DATA, "train_*.bin")) +
                      glob.glob(os.path.join(DATA, "val_*.bin")))
    if existing and not overwrite:
        raise SystemExit(
            f"{len(existing)} shard file(s) already in {DATA}; refusing to write over them. "
            f"A smaller rebuild would leave old train shards behind, one of them identical to "
            f"the new validation shard. Pass --overwrite to delete them first.")
    if overwrite:
        for p in existing:
            os.remove(p)
        meta_path = os.path.join(DATA, "meta.json")
        if os.path.exists(meta_path):
            os.remove(meta_path)            # it describes the corpus being deleted
        if existing:
            print(f"--overwrite: removed {len(existing)} old shard file(s)")

    os.makedirs(DATA, exist_ok=True)
    buf = np.empty(SHARD_TOKENS, dtype=np.uint16)
    fill, shard, written = 0, 0, 0
    written_paths = []
    t0 = time.time()

    def flush():
        nonlocal shard
        kind = "val" if shard >= first_val else "train"
        idx = shard - first_val if kind == "val" else shard
        path = os.path.join(DATA, f"{kind}_{idx:04d}.bin")
        buf.tofile(path)
        written_paths.append(path)
        print(f"  shard {shard}/{total_shards}: {SHARD_TOKENS:,} tokens -> {os.path.basename(path)}")
        shard += 1

    for text in stream_docs():
        ids = tok.encode(text).ids
        ids.append(eot)                      # documents are separated, never merged
        n = len(ids)
        if fill + n > SHARD_TOKENS:
            take = SHARD_TOKENS - fill
            buf[fill:fill + take] = ids[:take]
            flush()
            written += SHARD_TOKENS
            rest = ids[take:]
            fill = len(rest)
            buf[:fill] = rest
            if shard >= total_shards:
                break
            rate = written / max(1e-9, time.time() - t0)
            print(f"  {written/1e9:.2f}B tokens, {rate/1e6:.1f}M tok/s")
        else:
            buf[fill:fill + n] = ids
            fill += n

    # The loop has two exits and only one is success. If the stream runs dry first, the
    # tail shards -- which are exactly the validation shards -- were never written, and a
    # meta.json written anyway would claim a held-out set that does not exist.
    if shard < total_shards:
        for p in written_paths:
            os.remove(p)
        raise SystemExit(
            f"the stream ran out after {shard} of {total_shards} shards ({written:,} tokens); "
            f"removed the {len(written_paths)} partial shard(s) this run wrote and wrote no "
            f"meta.json, because it would claim {val_shards} validation shard(s) that do not "
            f"exist. Lower --tokens.")

    train_tokens = (total_shards - val_shards) * SHARD_TOKENS
    val_tokens = val_shards * SHARD_TOKENS
    meta = {
        "dataset": f"{DATASET}:{CONFIG}",
        "tokens": int(written),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_shards": total_shards - val_shards,
        "val_shards": val_shards,
        "shard_tokens": SHARD_TOKENS,
        "vocab_size": tok.get_vocab_size(),
        "dtype": "uint16",
        "seconds": round(time.time() - t0, 1),
        "split_note": "val is the last full shard of the stream: documents the model never sees.",
    }
    with open(os.path.join(DATA, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n{written:,} tokens in {time.time()-t0:.0f}s -> {DATA}")
    print(f"  train {train_tokens:,} across {total_shards - val_shards} shards")
    print(f"  val   {val_tokens:,} held out")
    return written


def main():
    cfg = GPTConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2_000_000_000,
                    help="how many tokens to write (default 2B, ~2x Chinchilla-optimal)")
    ap.add_argument("--val-shards", type=int, default=1,
                    help="shards held back for validation, taken from the end of the stream")
    ap.add_argument("--overwrite", action="store_true",
                    help="delete existing train/val shards and meta.json before writing")
    ap.add_argument("--tokenizer-docs", type=int, default=200_000,
                    help="documents to fit the BPE on")
    ap.add_argument("--tokenizer-only", action="store_true")
    args = ap.parse_args()

    tok = load_tokenizer()
    if tok is None:
        tok = train_tokenizer(cfg.vocab_size, args.tokenizer_docs)
    else:
        print(f"reusing {TOKENIZER_PATH} ({tok.get_vocab_size():,} tokens)")

    if tok.get_vocab_size() != cfg.vocab_size:
        raise SystemExit(f"tokenizer is {tok.get_vocab_size()} tokens but the model "
                         f"config expects {cfg.vocab_size}; delete {TOKENIZER_PATH} to refit")

    if args.tokenizer_only:
        return 0
    tokenize(tok, args.tokens, args.val_shards, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
