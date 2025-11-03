# wave_unet.py

import torch
import numpy as np
import torch.nn as nn
import math
import json
import matplotlib as plt
import time
import random
import logging

from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional
from model.resample import Resample1d
from model.conv import ConvLayer
from model.crop import centre_crop
from model.model_pipeline import WaveUNetDataset, load_configs, create_train_test_splits
# from utils import save_model, load_model, compute_loss
from pathlib import Path
from typing import List

try:
    import mlflow
    _mlflow_available = True
except Exception:
    mlflow = None
    _mlflow_available = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_pipeline")

class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(UpsamplingBlock, self).__init__()
        assert(stride > 1)

        # CONV 1 for UPSAMPLING
        if res == "fixed":
            self.upconv = Resample1d(n_inputs, 15, stride, transpose=True)
        else:
            self.upconv = ConvLayer(n_inputs, n_inputs, kernel_size, stride, conv_type, transpose=True)

        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_outputs, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        # CONVS to combine high- with low-level information (from shortcut)
        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

    def forward(self, x, shortcut):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        for conv in self.pre_shortcut_convs:
            upsampled = conv(upsampled)

        # Prepare shortcut connection
        combined = centre_crop(shortcut, upsampled)

        # Combine high- and low-level features
        for conv in self.post_shortcut_convs:
            combined = conv(torch.cat([combined, centre_crop(upsampled, combined)], dim=1))
        return combined

    def get_output_size(self, input_size):
        curr_size = self.upconv.get_output_size(input_size)

        # Upsampling convs
        for conv in self.pre_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        # Combine convolutions
        for conv in self.post_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        return curr_size

class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(DownsamplingBlock, self).__init__()
        assert(stride > 1)

        self.kernel_size = kernel_size
        self.stride = stride

        # CONV 1
        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in
                                                  range(depth - 1)])

        # CONV 2 with decimation
        if res == "fixed":
            self.downconv = Resample1d(n_outputs, 15, stride) # Resampling with fixed-size sinc lowpass filter
        else:
            self.downconv = ConvLayer(n_outputs, n_outputs, kernel_size, stride, conv_type)

    def forward(self, x):
        # PREPARING SHORTCUT FEATURES
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)

        # PREPARING FOR DOWNSAMPLING
        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)

        # DOWNSAMPLING
        out = self.downconv(out)

        return out, shortcut

    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)

        for conv in reversed(self.post_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)

        for conv in reversed(self.pre_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)
        return curr_size

class Waveunet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_outputs, instruments, kernel_size, target_output_size, conv_type, res, separate=False, depth=1, strides=2):
        super(Waveunet, self).__init__()

        self.num_levels = len(num_channels)
        self.strides = strides
        self.kernel_size = kernel_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.depth = depth
        self.instruments = instruments
        self.separate = separate

        # Only odd filter kernels allowed
        assert(kernel_size % 2 == 1)

        self.waveunets = nn.ModuleDict()

        model_list = instruments if separate else ["ALL"]
        # Create a model for each source if we separate sources separately, otherwise only one (model_list=["ALL"])
        for instrument in model_list:
            module = nn.Module()

            module.downsampling_blocks = nn.ModuleList()
            module.upsampling_blocks = nn.ModuleList()

            for i in range(self.num_levels - 1):
                in_ch = num_inputs if i == 0 else num_channels[i]

                module.downsampling_blocks.append(
                    DownsamplingBlock(in_ch, num_channels[i], num_channels[i+1], kernel_size, strides, depth, conv_type, res))

            for i in range(0, self.num_levels - 1):
                module.upsampling_blocks.append(
                    UpsamplingBlock(num_channels[-1-i], num_channels[-2-i], num_channels[-2-i], kernel_size, strides, depth, conv_type, res))

            module.bottlenecks = nn.ModuleList(
                [ConvLayer(num_channels[-1], num_channels[-1], kernel_size, 1, conv_type) for _ in range(depth)])

            # Output conv
            outputs = num_outputs if separate else num_outputs * len(instruments)
            module.output_conv = nn.Conv1d(num_channels[0], outputs, 1)

            self.waveunets[instrument] = module

        self.set_output_size(target_output_size)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size

        self.input_size, self.output_size = self.check_padding(target_output_size)
        print("Using valid convolutions with " + str(self.input_size) + " inputs and " + str(self.output_size) + " outputs")

        assert((self.input_size - self.output_size) % 2 == 0)
        self.shapes = {"output_start_frame" : (self.input_size - self.output_size) // 2,
                       "output_end_frame" : (self.input_size - self.output_size) // 2 + self.output_size,
                       "output_frames" : self.output_size,
                       "input_frames" : self.input_size}

    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles so each output in the cycle is weighted equally during training
        bottleneck = 1

        while True:
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)
            if out is not False:
                return out
            bottleneck += 1

    def check_padding_for_bottleneck(self, bottleneck, target_output_size):
        module = self.waveunets[[k for k in self.waveunets.keys()][0]]
        try:
            curr_size = bottleneck
            for idx, block in enumerate(module.upsampling_blocks):
                curr_size = block.get_output_size(curr_size)
            output_size = curr_size

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(module.bottlenecks):
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(module.downsampling_blocks)):
                curr_size = block.get_input_size(curr_size)

            assert(output_size >= target_output_size)
            return curr_size, output_size
        except AssertionError as e:
            return False

    def forward_module(self, x, module):
        '''
        A forward pass through a single Wave-U-Net (multiple Wave-U-Nets might be used, one for each source)
        :param x: Input mix
        :param module: Network module to be used for prediction
        :return: Source estimates
        '''
        shortcuts = []
        out = x

        # DOWNSAMPLING BLOCKS
        for block in module.downsampling_blocks:
            out, short = block(out)
            shortcuts.append(short)

        # BOTTLENECK CONVOLUTION
        for conv in module.bottlenecks:
            out = conv(out)

        # UPSAMPLING BLOCKS
        for idx, block in enumerate(module.upsampling_blocks):
            out = block(out, shortcuts[-1 - idx])

        # OUTPUT CONV
        out = module.output_conv(out)
        if not self.training:  # At test time clip predictions to valid amplitude range
            out = out.clamp(min=-1.0, max=1.0)
        return out

    def forward(self, x, inst=None):
        curr_input_size = x.shape[-1]
        assert(curr_input_size == self.input_size) # User promises to feed the proper input himself, to get the pre-calculated (NOT the originally desired) output size

        if self.separate:
            return {inst : self.forward_module(x, self.waveunets[inst])}
        else:
            assert(len(self.waveunets) == 1)
            out = self.forward_module(x, self.waveunets["ALL"])

            out_dict = {}
            for idx, inst in enumerate(self.instruments):
                out_dict[inst] = out[:, idx * self.num_outputs:(idx + 1) * self.num_outputs]
            return out_dict
        
# ---------------------------
# 5) Losses
# ---------------------------

def negative_snr(estimated, target):
    """Negative SNR to maximize SNR during minimization"""
    noise = estimated - target
    signal_power = torch.mean(target ** 2)
    noise_power = torch.mean(noise ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    return -snr  # Minimize negative SNR = maximize SNR

def negative_sisnr(estimated, target):
    """Scale-Invariant SNR"""
    # Center the signals
    estimated = estimated - torch.mean(estimated, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)
    
    # Project estimated onto target
    dot = torch.sum(estimated * target, dim=-1, keepdim=True)
    target_power = torch.sum(target ** 2, dim=-1, keepdim=True) + 1e-8
    scale = dot / target_power
    
    # Get projection and error
    target_component = scale * target
    noise_component = estimated - target_component
    
    # Compute SI-SNR
    target_power = torch.sum(target_component ** 2, dim=-1)
    noise_power = torch.sum(noise_component ** 2, dim=-1)
    sisnr = 10 * torch.log10(target_power / (noise_power + 1e-8))
    return -torch.mean(sisnr)

def multi_resolution_stft_loss(estimated, target, fft_sizes=[512, 1024, 2048], hop_ratio=0.25):
    """Multi-resolution spectral convergence + log-magnitude loss"""
    loss = 0.0

    estimated2d = estimated.squeeze(1)
    target2d = target.squeeze(1)
    
    for n_fft in fft_sizes:
        hop_length = int(n_fft * hop_ratio)
        
        # Compute STFTs
        target_stft = torch.stft(target2d, n_fft, hop_length, window=torch.hann_window(n_fft).to(target2d.device), return_complex=True)
        estimated_stft = torch.stft(estimated2d, n_fft, hop_length, window=torch.hann_window(n_fft).to(target2d.device), return_complex=True)
        
        target_mag = torch.abs(target_stft)
        estimated_mag = torch.abs(estimated_stft)
        
        # Spectral convergence
        sc_loss = torch.norm(target_mag - estimated_mag, p='fro') / torch.norm(target_mag, p='fro')
        
        # Log magnitude loss (perceptually weighted)
        log_target = torch.log(target_mag + 1e-5)
        log_estimated = torch.log(estimated_mag + 1e-5)
        mag_loss = nn.functional.l1_loss(log_estimated, log_target)
        
        loss += sc_loss + mag_loss
    
    return loss / len(fft_sizes)

def combined_loss(estimated_target, estimated_noise, true_target, true_noise):
    """
    Loss for noise separation:
        1. Reconstruction part;
        2. SNR part;
        3. Spectral loss part.
    """
    # 1. L1 loss for both components
    l1_loss = nn.functional.l1_loss(estimated_target, true_target) + nn.functional.l1_loss(estimated_noise, true_noise)
    
    # 2. SNR maximization for target
    snr_loss = negative_snr(estimated_target, true_target)
    
    # 3. Spectral loss for better quality
    spectral_loss = multi_resolution_stft_loss(estimated_target, true_target)
    
    return l1_loss + snr_loss + 0.5 * spectral_loss

# ---------------------------
# 6) Metrics (MSE, MAE, SNR, SI-SDR)
# ---------------------------
def mse_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred)**2))


def mae_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def snr_db_metric(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-8) -> float:
    # SNR = 20*log10(rms_true / rms_error)
    rms_true = math.sqrt(np.mean(y_true**2) + eps)
    rms_err = math.sqrt(np.mean((y_true - y_pred)**2) + eps)
    return 20.0 * math.log10(rms_true / (rms_err + 1e-12))


def si_sdr_metric(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-8) -> float:
    """
    Scale-Invariant SDR for single-channel signals
    y_true, y_pred: 1D numpy arrays (same length)
    """
    # remove mean
    s = y_true.astype(np.float64) - np.mean(y_true)
    s_hat = y_pred.astype(np.float64) - np.mean(y_pred)
    # projection
    s_target = (np.dot(s_hat, s) / (np.dot(s, s) + eps)) * s
    e_noise = s_hat - s_target
    num = np.sum(s_target**2)
    den = np.sum(e_noise**2) + eps
    return 10.0 * math.log10((num + eps) / den)


def sdr_metric(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Signal to Distortion Ratio (SDR)
    
    SDR = 10 * log10(||s_target||^2 / (||e_interf + e_noise + e_artif||^2))
    
    Args:
        y_true: Reference signal (target)
        y_pred: Estimated signal
        eps: Small value to avoid division by zero
    
    Returns:
        SDR in dB
    """
    # Remove mean (optional but often done)
    s = y_true.astype(np.float64) - np.mean(y_true)
    s_hat = y_pred.astype(np.float64) - np.mean(y_pred)
    
    # Projection for target component
    alpha = np.dot(s_hat, s) / (np.dot(s, s) + eps)
    s_target = alpha * s
    
    # Error (distortion)
    e_total = s_hat - s_target
    
    # Calculate energies
    target_energy = np.sum(s_target**2)
    distortion_energy = np.sum(e_total**2)
    
    return 10.0 * math.log10((target_energy + eps) / (distortion_energy + eps))


def sir_metric(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Signal to Interference Ratio (SIR)
    
    SIR = 10 * log10(||s_target||^2 / ||e_interf||^2)
    
    Args:
        y_true: Reference signal (target)
        y_pred: Estimated signal
        eps: Small value to avoid division by zero
    
    Returns:
        SIR in dB
    """
    # Remove mean
    s = y_true.astype(np.float64) - np.mean(y_true)
    s_hat = y_pred.astype(np.float64) - np.mean(y_pred)
    
    # Projection for target component
    alpha = np.dot(s_hat, s) / (np.dot(s, s) + eps)
    s_target = alpha * s
    
    # For SIR, we need the interference component
    # In single-channel case, interference is the part that correlates with other sources
    # For simplicity in single-channel, we use the residual after removing target
    e_interf = s_hat - s_target
    
    # Calculate energies
    target_energy = np.sum(s_target**2)
    interf_energy = np.sum(e_interf**2)
    
    return 10.0 * math.log10((target_energy + eps) / (interf_energy + eps))


def sar_metric(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Signal to Artifacts Ratio (SAR)
    
    SAR = 10 * log10(||s_target + e_interf||^2 / ||e_artif||^2)
    
    Args:
        y_true: Reference signal (target)
        y_pred: Estimated signal
        eps: Small value to avoid division by zero
    
    Returns:
        SAR in dB
    """
    # Remove mean
    s = y_true.astype(np.float64) - np.mean(y_true)
    s_hat = y_pred.astype(np.float64) - np.mean(y_pred)
    
    # Projection for target component
    alpha = np.dot(s_hat, s) / (np.dot(s, s) + eps)
    s_target = alpha * s
    
    # In single-channel case, artifacts are typically the residual
    # For SAR, we consider s_target + e_interf vs artifacts
    # In single-channel context, artifacts ≈ e_noise
    e_interf = s_hat - s_target
    
    # Signal + interference
    signal_plus_interf = s_target + e_interf
    
    # For single-channel, artifacts are typically modeled as the non-linear distortions
    # We approximate artifacts as the residual that cannot be explained by linear projection
    e_artif = s_hat - signal_plus_interf  # This would be zero in linear model
    
    # More practical approach for single-channel SAR
    signal_plus_interf_energy = np.sum(signal_plus_interf**2)
    artifacts_energy = np.sum(e_artif**2) if np.sum(e_artif**2) > eps else eps
    
    return 10.0 * math.log10((signal_plus_interf_energy + eps) / (artifacts_energy + eps))


def isr_metric(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Image to Spatial distortion Ratio (ISR) - Also known as Source Image to Spatial distortion Ratio
    
    ISR = 10 * log10(||s_target||^2 / ||e_spat||^2)
    
    Note: In single-channel audio, ISR is less commonly used and may be approximated.
    This implementation provides a reasonable approximation for single-channel case.
    
    Args:
        y_true: Reference signal (target)
        y_pred: Estimated signal
        eps: Small value to avoid division by zero
    
    Returns:
        ISR in dB
    """
    # Remove mean
    s = y_true.astype(np.float64) - np.mean(y_true)
    s_hat = y_pred.astype(np.float64) - np.mean(y_pred)
    
    # Projection for target component
    alpha = np.dot(s_hat, s) / (np.dot(s, s) + eps)
    s_target = alpha * s
    
    # For ISR in single-channel, we approximate spatial distortion
    # as the part that doesn't align with the target signal
    e_spat = s_hat - s_target
    
    # Calculate energies
    target_energy = np.sum(s_target**2)
    spat_energy = np.sum(e_spat**2)
    
    return 10.0 * math.log10((target_energy + eps) / (spat_energy + eps))


_METRIC_FUNCS = {
    "mse": mse_metric,
    "mae": mae_metric,
    "snr_db": snr_db_metric,
    "si_sdr": si_sdr_metric,
    "sdr": sdr_metric,
    "sir": sir_metric,
    "sar": sar_metric,
    "isr": isr_metric
}

# ---------------------------
# 6) train_epoch & validate_epoch
# ---------------------------
def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device, clip_grad: Optional[float] = 5.0):
    model.train()
    running_loss = 0.0
    n_samples = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for mix, target, residual in pbar:

        input_size = model.shapes["input_frames"]
        output_size = model.target_output_size

        # Original mix signal with sample size of output_size
        orig_mix = mix.clone()

        # Padding from left and right for mixed signal
        sample_diff = input_size - mix.shape[-1]
        if sample_diff > 0:
            # pad left side
            mix = nn.functional.pad(mix, (sample_diff // 2, 0))
            # pad right side
            mix = nn.functional.pad(mix, (0, sample_diff - sample_diff // 2))
        else:
            raise ValueError(f"Expected a input_size > mix.shape[-1], but got {input_size} < {mix.shape[-1]}")
        
        orig_mix = orig_mix.to(device)
        mix = mix.to(device)
        target = target.to(device)
        residual = residual.to(device)

        optimizer.zero_grad()

        out_dict = model(mix)
        target_est, residual_est = out_dict["target"], out_dict["residual"]

        if target_est.shape[-1] != output_size:
            target_est = nn.functional.interpolate(
                target_est, 
                size=output_size, 
                mode='linear', 
                align_corners=False
            )
        
        # For residual signal  
        if residual_est.shape[-1] != output_size:
            residual_est = nn.functional.interpolate(
                residual_est,
                size=output_size,
                mode='linear',
                align_corners=False
            )

        loss = criterion(target_est, residual_est, target, orig_mix)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        batch_size = mix.shape[0]
        running_loss += loss.item() * batch_size
        n_samples += batch_size
        pbar.set_postfix(loss=running_loss / n_samples)
    return running_loss / max(1, n_samples)


def validate_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                   device: torch.device, metric_names: List[str]):
    model.eval()
    running_loss = 0.0
    n_samples = 0
    metric_sums = {m: 0.0 for m in metric_names}
    with torch.no_grad():
        pbar = tqdm(loader, desc="val", leave=False)
        for mix, target, residual in pbar:
            input_size = model.shapes["input_frames"]
            output_size = model.target_output_size

            # Original mix signal with sample size of output_size
            orig_mix = mix.clone()

            # Padding from left and right for mixed signal
            sample_diff = input_size - mix.shape[-1]
            if sample_diff > 0:
                # pad left side
                mix = nn.functional.pad(mix, (sample_diff // 2, 0))
                # pad right side
                mix = nn.functional.pad(mix, (0, sample_diff - sample_diff // 2))
            else:
                raise ValueError(f"Expected a input_size > mix.shape[-1], but got {input_size} < {mix.shape[-1]}")
            
            orig_mix = orig_mix.to(device)
            mix = mix.to(device)
            target = target.to(device)
            residual = residual.to(device)

            out_dict = model(mix)
            target_est, residual_est = out_dict["target"], out_dict["residual"]

            if target_est.shape[-1] != output_size:
                target_est = nn.functional.interpolate(
                    target_est, 
                    size=output_size, 
                    mode='linear', 
                    align_corners=False
                )
            
            # For residual signal  
            if residual_est.shape[-1] != output_size:
                residual_est = nn.functional.interpolate(
                    residual_est,
                    size=output_size,
                    mode='linear',
                    align_corners=False
                )

            loss = criterion(target_est, residual_est, target, orig_mix)
            batch_size = mix.shape[0]
            running_loss += loss.item() * batch_size
            n_samples += batch_size
            # compute metrics sample-wise in numpy
            est_np = target_est.detach().cpu().numpy()
            clean_np = target.detach().cpu().numpy()
            for b in range(batch_size):
                y_true = clean_np[b, 0, :]
                y_pred = est_np[b, 0, :]
                for m in metric_names:
                    metric_value = _METRIC_FUNCS[m](y_true, y_pred)
                    metric_sums[m] += metric_value
            pbar.set_postfix(loss=running_loss / n_samples)
    avg_loss = running_loss / max(1, n_samples)
    avg_metrics = {m: (metric_sums[m] / n_samples) for m in metric_names}
    return avg_loss, avg_metrics

# ---------------------------
# 7) plotting helpers
# ---------------------------
def plot_history(history: dict, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # loss curve
    plt.figure()
    plt.plot(history.get("train_loss", []), label="train_loss")
    plt.plot(history.get("val_loss", []), label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(True)
    f1 = out_dir / "loss_curve.png"
    plt.savefig(f1); plt.close()
    # metrics
    for m in history.get("metrics", {}).keys():
        plt.figure()
        plt.plot(history["metrics"][m], label=m)
        plt.xlabel("epoch"); plt.ylabel(m); plt.legend(); plt.grid(True)
        plt.savefig(out_dir / f"metric_{m}.png")
        plt.close()
    return out_dir

# ---------------------------
# 8) High-level train & evaluate pipeline
# ---------------------------
def train_and_evaluate(
    processed_root: str,
    dataset_name: str,
    train_config_path: str,
    metrics_config_path: str,
    output_dir: str = "model_output",
    mlflow_enabled: bool = False,
    seed: int = 42
) -> dict:
    """
    High-level entrypoint for training and evaluation. This is Airflow callable.
    Returns a summary dict with final metrics and paths.
    """

    # load configs
    train_cfg, metrics_cfg = load_configs(train_config_path, metrics_config_path)
    # set seeds
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # create train/test splits if needed
    create_train_test_splits(processed_root, dataset_name, train_frac=float(train_cfg.get("train_frac", 0.9)), seed=seed)

    # dataset & dataloaders
    train_dataset = WaveUNetDataset(processed_root, dataset_name, split="train")
    val_dataset = WaveUNetDataset(processed_root, dataset_name, split="test")
    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=int(train_cfg.get("num_workers", 4)), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=int(train_cfg.get("num_workers", 4)), pin_memory=True)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model.to(device)

    # criterion, optimizer, scheduler
    criterion = combined_loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 5e-4))
    )
    scheduler = None
    if train_cfg.get("lr_step"):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg["lr_step"], gamma=float(train_cfg.get("lr_gamma", 0.5)))

    # metrics and monitoring
    metric_names = metrics_cfg.get("metrics", ["mse"])
    monitor = train_cfg.get("monitor_metric", "val_loss")  # e.g. "val_loss" or "si_sdr"
    monitor_mode = train_cfg.get("monitor_mode", "min")   # "min" or "max"
    best_score = math.inf if monitor_mode == "min" else -math.inf
    best_ckpt_path = None

    # output dirs
    out_root = Path(output_dir)
    ckpt_dir = out_root / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_root / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "metrics": {m: [] for m in metric_names}}

    # MLflow start run
    mlflow_run = None
    if mlflow_enabled:
        if not _mlflow_available:
            logger.warning("MLflow requested but not available; continuing without MLflow.")
            mlflow_enabled = False
        else:
            mlflow.start_run()
            mlflow_run = mlflow.active_run()
            mlflow.log_params(train_cfg)
            mlflow.log_params({"metrics_cfg": metrics_cfg})

    num_epochs = int(train_cfg.get("epochs", 50))
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, clip_grad=float(train_cfg.get("clip_grad", 5.0)))
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, metric_names)
        logger.info(f"Epoch {epoch+1} train_loss={train_loss:.6f} val_loss={val_loss:.6f} metrics={val_metrics} time={(time.time()-t0):.1f}s")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for m in metric_names:
            history["metrics"][m].append(val_metrics.get(m, None))

        # scheduler step
        if scheduler:
            scheduler.step()

        # monitoring and checkpointing
        # If monitor_metric is "val_loss" use that; if it's in val_metrics use that.
        if monitor == "val_loss":
            current = val_loss
        else:
            current = val_metrics.get(monitor)
            if current is None:
                logger.warning(f"Monitor metric {monitor} not found in val metrics; defaulting to val_loss.")
                current = val_loss

        is_better = (current < best_score) if monitor_mode == "min" else (current > best_score)
        if is_better:
            best_score = current
            best_ckpt_path = ckpt_dir / f"best_{monitor}_{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_cfg": train_cfg,
                "metrics_cfg": metrics_cfg,
                "best_score": best_score
            }, str(best_ckpt_path))
            logger.info(f"Saved new best checkpoint: {best_ckpt_path}")
            if mlflow_enabled:
                mlflow.log_metric(f"best_{monitor}", float(best_score), step=epoch)

        # log epoch metrics to mlflow
        if mlflow_enabled:
            mlflow.log_metric("train_loss", float(train_loss), step=epoch)
            mlflow.log_metric("val_loss", float(val_loss), step=epoch)
            for m, v in val_metrics.items():
                mlflow.log_metric(m, float(v), step=epoch)

    # after training: load best model and evaluate on test set (here val set is test)
    if best_ckpt_path is not None:
        ckpt = torch.load(str(best_ckpt_path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best model from {best_ckpt_path} for final evaluation.")
    else:
        logger.warning("No checkpoint saved during training; using last model for evaluation.")

    # Final evaluation (on val/test loader)
    final_loss, final_metrics = validate_epoch(model, val_loader, criterion, device, metric_names)
    logger.info(f"Final evaluation: loss={final_loss:.6f}, metrics={final_metrics}")

    # Save history & plots
    hist_json = out_root / "history.json"
    hist_json.write_text(json.dumps(history, indent=2))
    plot_history(history, plot_dir)

    # MLflow final logging & artifacts
    if mlflow_enabled:
        mlflow.log_metric("final_val_loss", float(final_loss))
        for m, v in final_metrics.items():
            mlflow.log_metric(f"final_{m}", float(v))
        # log artifacts
        mlflow.log_artifacts(str(plot_dir), artifact_path="plots")
        mlflow.log_artifact(str(hist_json), artifact_path="history")
        if best_ckpt_path:
            mlflow.log_artifact(str(best_ckpt_path), artifact_path="checkpoints")
        mlflow.end_run()

    summary = {
        "best_checkpoint": str(best_ckpt_path) if best_ckpt_path else None,
        "final_val_loss": float(final_loss),
        "final_metrics": final_metrics,
        "history_path": str(hist_json),
        "plot_dir": str(plot_dir)
    }
    return summary
