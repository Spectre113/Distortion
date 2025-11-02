# model_pipeline.py

import os
import json
from pathlib import Path
import random
import math
import time
import logging
from typing import Tuple, Dict, List, Optional

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Optional YAML support
try:
    import yaml
except Exception:
    yaml = None

# Optional MLflow
try:
    import mlflow
    _mlflow_available = True
except Exception:
    mlflow = None
    _mlflow_available = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_pipeline")


# ---------------------------
# 1) Config loading utility
# ---------------------------
def load_configs(train_config_path: str, metrics_config_path: str) -> Tuple[dict, dict]:
    """
    Load training configuration and metrics/plot configuration from YAML or JSON.
    Returns (train_cfg, metrics_cfg)
    """
    def _load(path: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        text = p.read_text(encoding="utf-8")
        if yaml and (p.suffix.lower() in (".yml", ".yaml")):
            return yaml.safe_load(text)
        # try json
        try:
            return json.loads(text)
        except Exception:
            if yaml:
                return yaml.safe_load(text)
            raise

    train_cfg = _load(train_config_path)
    metrics_cfg = _load(metrics_config_path)
    return train_cfg, metrics_cfg


# ---------------------------
# 2) Dataset for target/residual/mix triplets
# ---------------------------
class WaveUNetDataset(Dataset):
    """
    Loads triplets from processed dataset directory:
      processed/{dataset_name}/clean/*_target.wav
      processed/{dataset_name}/clean/*_residual.wav
      processed/{dataset_name}/noisy/*_mix.wav

    It matches by basename prefix (e.g. example_000001_target.wav / example_000001_residual.wav / example_000001_mix.wav)
    """

    def __init__(self, processed_root: str, dataset_name: str, split: str = "train"):
        """
        processed_root: base path (e.g., datasets/processed)
        dataset_name: the dataset folder name
        split: 'train' or 'test' - supports:
            - processed/{dataset_name}/clean + /noisy
            - processed/{dataset_name}/{split}/clean + /noisy
        """
        base = Path(processed_root) / dataset_name
        if not base.exists():
            raise RuntimeError(f"Processed dataset not found: {base}")

        # possible directory layouts
        possible_clean = base / "clean"
        possible_noisy = base / "noisy"
        alt_clean = base / split / "clean"
        alt_noisy = base / split / "noisy"

        if alt_clean.exists() and alt_noisy.exists():
            self.clean_dir = alt_clean
            self.noisy_dir = alt_noisy
        elif possible_clean.exists() and possible_noisy.exists():
            self.clean_dir = possible_clean
            self.noisy_dir = possible_noisy
        else:
            raise RuntimeError(f"Could not find clean/noisy directories in {base} or {base}/{split}")

        # list all relevant files
        target_files = sorted(self.clean_dir.glob("*_target.wav"))
        residual_files = sorted(self.clean_dir.glob("*_residual.wav"))
        mix_files = sorted(self.noisy_dir.glob("*_mix.wav"))

        # build basename maps
        def key_from_path(p: Path):
            stem = p.stem
            for s in ("_target", "_residual", "_mix"):
                if stem.endswith(s):
                    return stem[: -len(s)]
            return stem

        target_map = {key_from_path(p): p for p in target_files}
        residual_map = {key_from_path(p): p for p in residual_files}
        mix_map = {key_from_path(p): p for p in mix_files}

        # intersection of all three
        keys = sorted(list(set(target_map.keys()) & set(residual_map.keys()) & set(mix_map.keys())))
        if not keys:
            raise RuntimeError(f"No matching target/residual/mix triplets found in {self.clean_dir} and {self.noisy_dir}")

        self.triplets = [(target_map[k], residual_map[k], mix_map[k]) for k in keys]

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        target_p, residual_p, mix_p = self.triplets[idx]

        # load
        target, sr1 = sf.read(str(target_p))
        residual, sr2 = sf.read(str(residual_p))
        mix, sr3 = sf.read(str(mix_p))

        # ensure mono + same sr
        if target.ndim > 1:
            target = np.mean(target, axis=1)
        if residual.ndim > 1:
            residual = np.mean(residual, axis=1)
        if mix.ndim > 1:
            mix = np.mean(mix, axis=1)
        if not (sr1 == sr2 == sr3):
            raise RuntimeError(f"Sample rates mismatch for example {target_p.stem}")

        # make sure lengths align
        min_len = min(len(target), len(residual), len(mix))
        target = target[:min_len]
        residual = residual[:min_len]
        mix = mix[:min_len]

        # convert to (1, L) tensors
        target = torch.from_numpy(target.astype(np.float32)).unsqueeze(0)
        residual = torch.from_numpy(residual.astype(np.float32)).unsqueeze(0)
        mix = torch.from_numpy(mix.astype(np.float32)).unsqueeze(0)

        return mix, target, residual  # input, clean target, noise residual
    
# ---------------------------
# 3) Splitting helper (for target/residual/mix dataset)
# ---------------------------
def create_train_test_splits(processed_root: str, dataset_name: str, train_frac: float = 0.9, seed: int = 42):
    """
    Creates train/test subfolders for WaveUNetDataset-style processed datasets.
    Works with:
      processed/{dataset_name}/clean/*_target.wav
      processed/{dataset_name}/clean/*_residual.wav
      processed/{dataset_name}/noisy/*_mix.wav

    Output:
      processed/{dataset_name}/train/{clean,noisy}
      processed/{dataset_name}/test/{clean,noisy}

    Returns:
      (train_dataset_dir, test_dataset_dir)
    """
    base = Path(processed_root) / dataset_name
    train_clean = base / "train" / "clean"
    train_noisy = base / "train" / "noisy"
    test_clean = base / "test" / "clean"
    test_noisy = base / "test" / "noisy"

    # if already split, skip
    if train_clean.exists() and test_clean.exists():
        print("Train/test subfolders already exist; skipping split creation.")
        return str(base), str(base)

    # locate files
    all_target = sorted((base / "clean").glob("*_target.wav"))
    all_residual = sorted((base / "clean").glob("*_residual.wav"))
    all_mix = sorted((base / "noisy").glob("*_mix.wav"))

    if not all_target or not all_residual or not all_mix:
        raise RuntimeError(f"Missing one or more sets of files in {base} (need target, residual, mix)")

    # match keys like WaveUNetDataset
    def key(p: Path):
        s = p.stem
        for suf in ("_target", "_residual", "_mix"):
            if s.endswith(suf):
                return s[:-len(suf)]
        return s

    target_map = {key(p): p for p in all_target}
    residual_map = {key(p): p for p in all_residual}
    mix_map = {key(p): p for p in all_mix}

    keys = sorted(list(set(target_map.keys()) & set(residual_map.keys()) & set(mix_map.keys())))
    if not keys:
        raise RuntimeError(f"No matching triplets found under {base}")

    # random split
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_train = int(math.floor(len(keys) * train_frac))
    train_keys = keys[:n_train]
    test_keys = keys[n_train:]

    # make dirs
    for d in [train_clean, train_noisy, test_clean, test_noisy]:
        d.mkdir(parents=True, exist_ok=True)

    from shutil import copy2
    def copy_triplet(k: str, dst_clean: Path, dst_noisy: Path):
        copy2(target_map[k], dst_clean / target_map[k].name)
        copy2(residual_map[k], dst_clean / residual_map[k].name)
        copy2(mix_map[k], dst_noisy / mix_map[k].name)

    for k in train_keys:
        copy_triplet(k, train_clean, train_noisy)
    for k in test_keys:
        copy_triplet(k, test_clean, test_noisy)

    print(f"Created train/test split: {len(train_keys)} train, {len(test_keys)} test examples.")
    return str(base), str(base)