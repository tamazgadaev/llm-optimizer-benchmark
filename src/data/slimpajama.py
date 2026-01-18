import os
import time

import numpy as np
import tiktoken
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm

tknzr = tiktoken.get_encoding("gpt2")


def _is_main_process():
    """Check if this is the main process (rank 0) in distributed setting."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _wait_for_file(filepath, timeout=3600, poll_interval=1.0):
    """Wait for a file to exist (file-based barrier for data preparation)."""
    start_time = time.time()
    while not os.path.exists(filepath):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for {filepath}")
        time.sleep(poll_interval)


def get_slimpajama_data(datasets_dir, num_proc=40):
    SPJ_DATA_PATH = os.path.join(datasets_dir, "slimpajama6B/")
    train_path = os.path.join(SPJ_DATA_PATH, "train.bin")
    val_path = os.path.join(SPJ_DATA_PATH, "val.bin")
    
    # Only rank 0 prepares the data, others wait
    needs_prep = not os.path.exists(train_path) or not os.path.exists(val_path)
    
    if needs_prep and _is_main_process():
        os.makedirs(SPJ_DATA_PATH, exist_ok=True)
        dataset = load_dataset("DKYoon/SlimPajama-6B")

        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")

        def process(example):
            ids = tknzr.encode_ordinary(
                example["text"]
            )  # encode_ordinary ignores any special tokens
            ids.append(
                tknzr.eot_token
            )  # add the end of text token, e.g. 50256 for gpt2 bpe
            out = {"ids": ids, "len": len(ids)}
            return out

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(SPJ_DATA_PATH, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
    
    # Non-rank-0 processes wait for files to be ready (file-based sync, no NCCL timeout)
    if dist.is_initialized() and not _is_main_process():
        print(f"[Rank {dist.get_rank()}] Waiting for data preparation to complete...")
        _wait_for_file(train_path)
        _wait_for_file(val_path)
        print(f"[Rank {dist.get_rank()}] Data ready, continuing...")

    return {
        "train": os.path.join(SPJ_DATA_PATH, "train.bin"),
        "val": os.path.join(SPJ_DATA_PATH, "val.bin"),
    }


def get_slimpajama_chunk1(datasets_dir, num_proc=40):
    SPJ_DATA_PATH = os.path.join(datasets_dir, "slimpajama6B/")
    SPJ_CHUNK_1_DATA_PATH = os.path.join(SPJ_DATA_PATH, "chunk1")
    train_path = os.path.join(SPJ_CHUNK_1_DATA_PATH, "train.bin")
    val_path = os.path.join(SPJ_CHUNK_1_DATA_PATH, "val.bin")
    
    # Only rank 0 prepares the data, others wait
    needs_prep = not os.path.exists(train_path) or not os.path.exists(val_path)
    
    if needs_prep and _is_main_process():
        os.makedirs(SPJ_CHUNK_1_DATA_PATH, exist_ok=True)
        dataset = load_dataset("cerebras/SlimPajama-627B", split="train/chunk1")

        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")

        def process(example):
            ids = tknzr.encode_ordinary(
                example["text"]
            )  # encode_ordinary ignores any special tokens
            ids.append(
                tknzr.eot_token
            )  # add the end of text token, e.g. 50256 for gpt2 bpe
            out = {"ids": ids, "len": len(ids)}
            return out

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(SPJ_CHUNK_1_DATA_PATH, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
    
    # Non-rank-0 processes wait for files to be ready (file-based sync, no NCCL timeout)
    if dist.is_initialized() and not _is_main_process():
        print(f"[Rank {dist.get_rank()}] Waiting for data preparation to complete...")
        _wait_for_file(train_path)
        _wait_for_file(val_path)
        print(f"[Rank {dist.get_rank()}] Data ready, continuing...")

    return {
        "train": os.path.join(SPJ_CHUNK_1_DATA_PATH, "train.bin"),
        "val": os.path.join(SPJ_CHUNK_1_DATA_PATH, "val.bin"),
    }
