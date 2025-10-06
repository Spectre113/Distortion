import os
from pathlib import Path
import json
import random
import math
from typing import List, Tuple, Optional, Dict

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from tqdm import tqdm

# -------------------------
# Noise generator (your function, slightly hardened)
# -------------------------
def create_custom_noise_profile(duration, sample_rate, overall_gain_db=-25):
    """Return a noise array length = duration*sample_rate (dtype=float32)."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    harmonic_noise = np.zeros_like(t)

    # harmonic hum
    fundamental_freqs = [440, 516, 645]
    for fundamental in fundamental_freqs:
        for harmonic in range(2, 7):
            freq = fundamental * harmonic
            detuned_freq = freq * (1 + np.random.uniform(-0.01, 0.01))
            amplitude = 0.15 / harmonic
            am_depth = 0.1
            am_rate = 0.5
            am_mod = 1 + am_depth * np.sin(2 * np.pi * am_rate * t)
            harmonic_noise += amplitude * am_mod * np.sin(2 * np.pi * detuned_freq * t)

    # resonant peaks (narrow band)
    resonant_freqs = [3158, 3856, 5109]
    resonant_amplitudes = [0.08, 0.08, 0.06]
    resonant_noise = np.zeros_like(t)
    nyquist = sample_rate / 2.0
    for freq, amp in zip(resonant_freqs, resonant_amplitudes):
        white = np.random.normal(0, 1, len(t))
        low_cut = max(0.0001, (freq * 0.9) / nyquist)
        high_cut = min(0.9999, (freq * 1.1) / nyquist)
        try:
            b, a = signal.butter(4, [low_cut, high_cut], btype='band')
            narrow = signal.filtfilt(b, a, white)
        except Exception:
            narrow = white
        resonant_noise += amp * narrow

    # broadband hiss shaped by FIR
    white_noise = np.random.normal(0, 1, len(t))
    try:
        from scipy.signal import firwin2
        freq_points = [0, 1000, 1290, 1548, 1858, 2229, 2675, 4000, 6000, sample_rate/2]
        gain_response = [10, 15, 15, 15, 15, 15, 15, 8, 5, 5]
        norm = np.array(freq_points) / (sample_rate/2)
        norm = np.clip(norm, 0.0, 1.0)
        fir_coeffs = firwin2(1025, norm, gain_response)
        shaped_hiss = signal.filtfilt(fir_coeffs, [1.0], white_noise)
    except Exception:
        shaped_hiss = white_noise

    # combine -> apply notch -> normalize -> gain
    combined = harmonic_noise + resonant_noise + shaped_hiss
    try:
        b_notch, a_notch = signal.iirnotch(3179.3, 4, sample_rate)
        combined = signal.filtfilt(b_notch, a_notch, combined)
    except Exception:
        pass

    maxabs = np.max(np.abs(combined)) + 1e-12
    combined = combined / maxabs
    gain_lin = 10 ** (overall_gain_db / 20.0)
    return (combined * gain_lin).astype(np.float32)


# -------------------------
# SAD: find non-silent intervals (librosa)
# -------------------------
def detect_activity_intervals(audio: np.ndarray, sr: int, top_db: float = 30.0, frame_length: int = 2048, hop_length: int = 512) -> List[Tuple[int, int]]:
    """
    Return list of (start_sample, end_sample) intervals containing activity.
    Uses librosa.effects.split which is a simple SAD (energy thresholding).
    top_db: threshold in dB below reference to consider silence (lower -> more aggressive keep)
    """
    intervals = librosa.effects.split(y=audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    return [(int(s), int(e)) for s, e in intervals]


# -------------------------
# Helper: sample one fixed-length clip from non-silent intervals
# -------------------------
def sample_clip_from_intervals(audio: np.ndarray, sr: int, intervals: List[Tuple[int,int]], clip_duration: float, rng: Optional[random.Random] = None) -> np.ndarray:
    """
    Choose a random interval that can contain a clip of clip_duration.
    If no single interval is long enough, try to stitch or center-pad shorter audio to clip length.
    Returns a numpy array of length = clip_duration * sr
    """
    rng = rng or random
    clip_len = int(round(clip_duration * sr))
    # Filter intervals long enough
    long_intervals = [iv for iv in intervals if (iv[1] - iv[0]) >= clip_len]
    if long_intervals:
        s, e = rng.choice(long_intervals)
        start = rng.randint(s, e - clip_len)
        clip = audio[start:start + clip_len]
        return clip.astype(np.float32)
    # otherwise try to sample from any interval, possibly concatenating up to clip_len by wrapping/padding:
    if intervals:
        # pick a random interval, extract it, then either pad or loop to reach clip_len
        s, e = rng.choice(intervals)
        seg = audio[s:e].astype(np.float32)
        if len(seg) >= clip_len:
            # deterministic crop
            start = rng.randint(0, len(seg) - clip_len)
            return seg[start:start+clip_len]
        else:
            # repeat or pad center
            needed = clip_len - len(seg)
            left = needed // 2
            right = needed - left
            return np.pad(seg, (left, right), mode='constant', constant_values=0.0)
    # if no intervals (silent file) -> zero pad or use entire audio center
    if len(audio) >= clip_len:
        center = len(audio) // 2
        start = max(0, center - clip_len // 2)
        return audio[start:start + clip_len].astype(np.float32)
    else:
        return np.pad(audio.astype(np.float32), (0, clip_len - len(audio)), mode='constant')


# -------------------------
# RMS utilities
# -------------------------
def rms(x: np.ndarray, eps=1e-12) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64)**2) + eps))


# -------------------------
# Build one simulated mixture example (core of augmentation)
# -------------------------
def build_simulated_mixture(
    stem_paths: List[Path],
    sr: int,
    clip_duration: float = 3.0,
    min_stems: int = 1,
    max_stems: int = 8,
    energy_db_range: Tuple[float, float] = (-10.0, 10.0),
    rng: Optional[random.Random] = None,
    top_db: float = 30.0
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Given a list of available stems (Paths), randomly select k stems (k in [min_stems, max_stems])
    and produce a clean mixture (1D numpy array of length clip_duration*sr) and metadata list.
    Metadata describes which stems, start samples, applied dB gains, original RMS.
    """
    rng = rng or random
    n_available = len(stem_paths)
    if n_available == 0:
        raise ValueError("No stems provided to build_simulated_mixture()")

    k = rng.randint(min_stems, min(max_stems, n_available))
    selected = rng.sample(stem_paths, k)
    clip_len = int(round(clip_duration * sr))

    mixture = np.zeros(clip_len, dtype=np.float32)
    metadata = []

    for p in selected:
        audio, file_sr = librosa.load(str(p), sr=None, mono=True)
        # resample if needed
        if file_sr != sr:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        intervals = detect_activity_intervals(audio, sr, top_db=top_db)
        clip = sample_clip_from_intervals(audio, sr, intervals, clip_duration, rng=rng)
        orig_rms = rms(clip)
        db_change = rng.uniform(energy_db_range[0], energy_db_range[1])
        gain_lin = 10 ** (db_change / 20.0)
        scaled = (clip * gain_lin).astype(np.float32)
        # sum to mixture
        mixture = mixture + scaled
        metadata.append({
            "stem_path": str(p),
            "db_change": float(db_change),
            "gain_lin": float(gain_lin),
            "orig_rms": float(orig_rms)
        })

    # After summing, avoid clipping: scale mixture by peak if needed, but preserve RMS relationships.
    peak = float(np.max(np.abs(mixture)) + 1e-12)
    if peak > 0.99:
        mixture = (mixture / peak * 0.99).astype(np.float32)

    return mixture, metadata


# -------------------------
# Add synthetic noise to get input features + return metadata
# -------------------------
def add_noise_to_mixture(
    clean_mixture: np.ndarray,
    sr: int,
    snr_db: float,
    noise_func=create_custom_noise_profile,
    overall_noise_gain_db: float = -25.0
) -> Tuple[np.ndarray, Dict]:
    """
    Create noise using noise_func, scale to target SNR with respect to clean_mixture RMS,
    return noisy_mixture and noise metadata.
    """
    duration = len(clean_mixture) / sr
    noise = noise_func(duration, sr, overall_gain_db=overall_noise_gain_db)
    if len(noise) > len(clean_mixture):
        noise = noise[:len(clean_mixture)]
    elif len(noise) < len(clean_mixture):
        noise = np.pad(noise, (0, len(clean_mixture)-len(noise)))

    rms_clean = rms(clean_mixture)
    rms_noise = rms(noise)
    target_lin = 10 ** (snr_db / 20.0)
    required_noise_rms = (rms_clean / target_lin) if target_lin > 0 else rms_clean
    noise_gain = (required_noise_rms / (rms_noise + 1e-12))
    adjusted_noise = (noise * noise_gain).astype(np.float32)
    noisy = clean_mixture + adjusted_noise

    # prevent clipping
    peak = float(np.max(np.abs(noisy)) + 1e-12)
    if peak > 1.0:
        noisy = (noisy / peak * 0.99).astype(np.float32)

    meta = {
        "snr_db_target": float(snr_db),
        "rms_clean": float(rms_clean),
        "rms_noise_before_gain": float(rms_noise),
        "noise_gain": float(noise_gain),
        "overall_noise_profile_db": float(overall_noise_gain_db)
    }
    return noisy, meta


# -------------------------
# Pipeline orchestration (Airflow friendly)
# -------------------------
def run_augmentation_pipeline(
    stems_root: str,
    output_base: str = "dataset",
    dataset_name: str = "aug_mixtures_v1",
    sample_rate: int = 22050,
    clip_duration: float = 3.0,
    min_stems: int = 1,
    max_stems: int = 8,
    energy_db_range: Tuple[float,float] = (-10.0, 10.0),
    snr_db_range: Tuple[float,float] = (5.0, 20.0),
    max_files: Optional[int] = None,         # restrict number of stems considered (N); None -> all
    n_examples: int = 1000,                  # number of augmented examples to synthesize
    top_db: float = 30.0,
    seed: Optional[int] = None
):
    """
    High-level pipeline:
      - discover stems (N)
      - limit to max_files if provided
      - for i in [0..n_examples): create one mixture example:
          - randomly choose between min_stems..max_stems stems
          - sample clip_duration from each stem (using SAD)
          - apply dB scaling per stem
          - sum -> clean_mixture (target)
          - sample snr in snr_db_range -> create noisy (feature)
          - save noisy and clean wavs + metadata json
    """
    rng = random.Random(seed)
    stems_root = Path(stems_root)
    stem_paths = sorted([p for p in stems_root.rglob("*") if p.suffix.lower() in (".wav", ".flac", ".mp3")])
    if not stem_paths:
        raise RuntimeError(f"No audio stems found in {stems_root}")

    if max_files is not None:
        stem_paths = stem_paths[:max_files]

    out_root = Path(output_base) / "processed" / dataset_name
    clean_dir = out_root / "clean"
    noisy_dir = out_root / "noisy"
    meta_dir = out_root / "meta"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(stem_paths)} stems. Will create {n_examples} examples using up to {max_stems} stems each.")

    for idx in tqdm(range(n_examples), desc="Synth examples"):
        # Build clean mixture
        mixture, stems_meta = build_simulated_mixture(
            stem_paths=stem_paths,
            sr=sample_rate,
            clip_duration=clip_duration,
            min_stems=min_stems,
            max_stems=max_stems,
            energy_db_range=energy_db_range,
            rng=rng,
            top_db=top_db
        )

        # Choose an SNR for noise
        snr = float(rng.uniform(snr_db_range[0], snr_db_range[1]))
        noisy, noise_meta = add_noise_to_mixture(mixture, sr=sample_rate, snr_db=snr)

        # Save files
        basename = f"example_{idx:06d}"
        clean_path = clean_dir / f"{basename}_clean.wav"
        noisy_path = noisy_dir / f"{basename}_noisy.wav"
        meta_path = meta_dir / f"{basename}.json"

        sf.write(str(clean_path), mixture, sample_rate)
        sf.write(str(noisy_path), noisy, sample_rate)

        full_meta = {
            "example_id": basename,
            "sample_rate": int(sample_rate),
            "clip_duration": float(clip_duration),
            "min_stems": int(min_stems),
            "max_stems": int(max_stems),
            "energy_db_range": [float(energy_db_range[0]), float(energy_db_range[1])],
            "snr_db_range": [float(snr_db_range[0]), float(snr_db_range[1])],
            "chosen_snr_db": float(snr),
            "stems_meta": stems_meta,
            "noise_meta": noise_meta
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(full_meta, f, indent=2)

    print(f"Saved {n_examples} examples to {out_root}")
    return str(out_root)
input_dir = "IDMT-SMT-GUITAR_V2/dataset2/audio/"

run_augmentation_pipeline(
    stems_root=input_dir,
    output_base="guitar_dataset",
    dataset_name="dataset2-test",
    n_examples=1000,
    seed=1
)