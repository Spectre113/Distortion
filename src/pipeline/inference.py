# inference_pipeline.py
import os
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import soundfile as sf
import librosa
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from .model_pipeline import Waveunet

# ------------------------
# Helpers: chunking & IO
# ------------------------
def split_audio_into_chunks(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration: float
) -> Tuple[List[np.ndarray], int]:
    """
    Split 1D numpy audio into non-overlapping chunks of chunk_duration (seconds).
    Pads the last chunk with zeros if needed.

    Returns:
      chunks: list of numpy arrays (each length == chunk_len)
      orig_len: original audio length in samples (for trimming after reconstruction)
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # to mono

    orig_len = len(audio)
    chunk_len = int(round(chunk_duration * sample_rate))
    if chunk_len <= 0:
        raise ValueError("chunk_duration too small for given sample_rate")

    n_chunks = (orig_len + chunk_len - 1) // chunk_len
    chunks = []
    for i in range(n_chunks):
        s = i * chunk_len
        e = s + chunk_len
        if e <= orig_len:
            chunks.append(audio[s:e].astype(np.float32))
        else:
            # pad right
            pad = e - orig_len
            chunk = np.pad(audio[s:orig_len], (0, pad), mode='constant').astype(np.float32)
            chunks.append(chunk)
    return chunks, orig_len


def reconstruct_from_chunks(chunks_preds: List[np.ndarray], orig_len: int) -> np.ndarray:
    """
    Concatenate list of 1D arrays (all same length) and trim to orig_len samples.
    """
    if not chunks_preds:
        return np.zeros(orig_len, dtype=np.float32)
    out = np.concatenate(chunks_preds, axis=0)
    if len(out) >= orig_len:
        return out[:orig_len].astype(np.float32)
    # if concatenated length is shorter (shouldn't happen), pad
    pad = orig_len - len(out)
    return np.pad(out, (0, pad), mode='constant').astype(np.float32)

# ------------------------
# Model loading
# ------------------------
def load_best_model(checkpoint_path: str, device: Optional[torch.device] = None):
    """
    Loads a checkpoint saved by train_and_evaluate(...) and returns a model in eval mode.
    The checkpoint is expected to contain 'model_state_dict' and optionally 'train_cfg' dict.
    If train_cfg exists, it is used to re-create the WaveUNet1D with the same architecture.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    ckpt = torch.load(checkpoint_path, map_location=device)

    # try to infer model config
    train_cfg = ckpt.get("train_cfg") or ckpt.get("args") or {}
    kernel_size = int(train_cfg.get("kernel_size", 15))

    levels = int(train_cfg.get("levels", 6))
    features = int(train_cfg.get("features", 32))
    feature_growth = train_cfg.get("feature_growth", "add")

    depth = int(train_cfg.get("depth", 1))
    strides = int(train_cfg.get("stride", 4))


    num_features = [features*i for i in range(1, levels+1)] if feature_growth == "add" else \
        [features*2**i for i in range(0, levels)]

    model = Waveunet(
        num_inputs=1,
        num_channels=num_features,
        num_outputs=1,
        instruments=['target', 'residual'],
        kernel_size=kernel_size,
        target_output_size=66150,
        conv_type="gn",
        res="fixed",
        separate=False,
        depth=depth,
        strides=strides
    )
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, train_cfg

# ------------------------
# Core inference for a single recording
# ------------------------
def infer_single_recording(
    input_path: str,
    model: torch.nn.Module,
    sample_rate: int,
    chunk_duration: float,
    batch_size: int,
    device: torch.device,
    output_dir: str,
    checkpoint_path: str,
    dtype=np.float32
) -> Dict:
    """
    Process one recording: split into chunks, run model in batches, reconstruct, save output and metadata.

    Returns metadata dict with paths and basic stats.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load and ensure mono
    audio, sr = librosa.load(str(input_path), sr=sample_rate, mono=True)
    chunks, orig_len = split_audio_into_chunks(audio, sample_rate, chunk_duration)

    preds_target = []
    preds_residual = []
    model_device = device

    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            # build tensor shape (B,1,L)
            batch_arr = np.stack(batch_chunks, axis=0)  # (B, L)
            batch_tensor = torch.from_numpy(batch_arr).float().unsqueeze(1).to(model_device)  # (B,1,L)

            input_size = model.shapes["input_frames"]

            # Padding from left and right for mixed signal
            sample_diff = input_size - batch_tensor.shape[-1]
            if sample_diff > 0:
                # pad left side
                batch_tensor = nn.functional.pad(batch_tensor, (sample_diff // 2, 0))
                # pad right side
                batch_tensor = nn.functional.pad(batch_tensor, (0, sample_diff - sample_diff // 2))
            else:
                raise ValueError(f"Expected a input_size > mix.shape[-1], but got {input_size} < {mix.shape[-1]}")

            out_dict = model(batch_tensor)  # expected (B,1,L) thanks to model alignment
            target_est, residual_est = out_dict["target"], out_dict["residual"]
            target_tensor = target_est.detach().cpu().numpy()  # (B,1,L)
            residual_tensor = residual_est.detach().cpu().numpy()
            
            out_arrs_target = [o[0].astype(dtype) for o in target_tensor]  # list of 1D arrays
            out_arrs_residual = [o[0].astype(dtype) for o in residual_tensor]
            preds_target.extend(out_arrs_target)
            preds_residual.extend(out_arrs_residual)

    # reconstruct
    reconstructed_target = reconstruct_from_chunks(preds_target, orig_len)
    reconstructed_residual = reconstruct_from_chunks(preds_residual, orig_len)

    # save
    out_basename = input_path.stem + "_target_recon.wav"
    out_path = output_dir / out_basename
    sf.write(str(out_path), reconstructed_target, sample_rate)

    out_basename = input_path.stem + "_resid_recon.wav"
    out_path = output_dir / out_basename
    sf.write(str(out_path), reconstructed_residual, sample_rate)

    # save metadata
    meta = {
        "input_path": str(input_path),
        "output_path": str(out_path),
        "checkpoint": str(checkpoint_path),
        "sample_rate": int(sample_rate),
        "chunk_duration": float(chunk_duration),
        "n_chunks": int(len(chunks)),
        "orig_len_samples": int(orig_len),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    meta_path = output_dir / (input_path.stem + "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta

# ------------------------
# Top-level Airflow-friendly entrypoint
# ------------------------
def run_inference_pipeline(
    input_root: str,
    model_checkpoint: str,
    output_root: str,
    chunk_duration: float = 3.0,
    sample_rate: int = 22050,
    batch_size: int = 8,
    device_str: Optional[str] = None,
    glob_patterns: Optional[List[str]] = None,
    max_files: Optional[int] = None
) -> Dict:
    """
    Main function to be called by Airflow PythonOperator.

    Parameters:
      - input_root: folder with recordings (will search recursively)
      - model_checkpoint: path to checkpoint (best model)
      - output_root: where to save reconstructed recordings and metadata
      - chunk_duration: seconds per chunk to pass to model
      - sample_rate: sampling rate for loading/saving
      - batch_size: inference batch size
      - device_str: 'cuda' or 'cpu' (if None, automatically picked)
      - glob_patterns: list of glob patterns to find files (default ['**/*.wav'])
      - max_files: optional cap on number of recordings to process

    Returns:
      summary dict with list of processed files and metadata paths.
    """
    device = torch.device(device_str if device_str is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, train_cfg = load_best_model(model_checkpoint, device=device)

    input_root = Path(input_root)
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    glob_patterns = glob_patterns or ["**/*.wav"]

    # collect files
    files = []
    for pat in glob_patterns:
        files.extend(sorted(input_root.glob(pat)))
    if not files:
        raise RuntimeError(f"No audio files found in {input_root} with patterns {glob_patterns}")

    if max_files is not None:
        files = files[:max_files]

    processed = []
    pbar = tqdm(files, desc="Inference files", dynamic_ncols=True)
    for f in pbar:
        try:
            meta = infer_single_recording(
                input_path=str(f),
                model=model,
                sample_rate=sample_rate,
                chunk_duration=chunk_duration,
                batch_size=batch_size,
                device=device,
                output_dir=str(out_root),
                checkpoint_path=model_checkpoint
            )
            processed.append(meta)
        except Exception as e:
            # don't crash whole DAG on single file; instead record error
            processed.append({"input_path": str(f), "error": str(e)})
            # you may also choose to re-raise if you want failure semantics in Airflow
            # raise

    summary = {
        "model_checkpoint": str(model_checkpoint),
        "n_files_requested": len(files),
        "n_processed": len(processed),
        "output_root": str(out_root),
        "processed": processed
    }
    # save summary
    with open(out_root / "inference_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":
    run_inference_pipeline(
        input_root="guitar_dataset/processed/quality_test/orig",
        model_checkpoint="model_output/checkpoints/best_snr_db_5.pt",
        output_root="guitar_dataset/processed/quality_test/recon"
    )