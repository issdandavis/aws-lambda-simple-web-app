"""
Audio Axis Module - Layer 14 FFT Telemetry

FFT-based telemetry channel for SCBE-AETHERMOORE without altering the invariant metric.

Feature Vector:
    f_audio(t) = [Ea, Ca, Fa, rHF,a]

Where:
    - Ea = log(ε + Σₙ a[n]²) — Frame energy
    - Ca = (Σₖ fₖ·Pₐ[k]) / (Σₖ Pₐ[k]) — Spectral centroid
    - Fa = Σₖ (√Pₐ[k] - √Pₐ_prev[k])² — Spectral flux
    - rHF,a = Σₖ∈Khigh Pₐ[k] / Σₖ Pₐ[k] — High-frequency ratio
    - Saudio = 1 - rHF,a — Audio stability score

Risk Integration:
    Risk' = Risk_base + w_a·(1 - S_audio)

Properties Proven:
1. Stability bounded: S_audio ∈ [0,1]
2. HF detection: high-freq signals → high rHF,a
3. Flux sensitivity: different frames → flux > 0

Reference: SCBE Patent Specification, Layer 14 (Audio Axis)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, NamedTuple
import numpy as np
from enum import Enum, auto


# Small epsilon to avoid log(0)
EPSILON = 1e-10

# Default sample rate (Hz)
DEFAULT_SAMPLE_RATE = 44100

# High-frequency cutoff ratio (above this fraction of Nyquist = HF)
HF_CUTOFF_RATIO = 0.5


class AudioStabilityLevel(Enum):
    """Audio stability classification."""
    STABLE = auto()       # S_audio >= 0.8
    MODERATE = auto()     # 0.5 <= S_audio < 0.8
    UNSTABLE = auto()     # 0.2 <= S_audio < 0.5
    CRITICAL = auto()     # S_audio < 0.2


class AudioFeatures(NamedTuple):
    """
    Audio feature vector f_audio(t).

    Attributes:
        energy: E_a = log(ε + Σ a[n]²)
        centroid: C_a = spectral centroid frequency
        flux: F_a = spectral flux from previous frame
        hf_ratio: r_HF,a = high-frequency energy ratio
        stability: S_audio = 1 - r_HF,a
    """
    energy: float
    centroid: float
    flux: float
    hf_ratio: float
    stability: float

    def get_stability_level(self) -> AudioStabilityLevel:
        """Classify stability level."""
        if self.stability >= 0.8:
            return AudioStabilityLevel.STABLE
        elif self.stability >= 0.5:
            return AudioStabilityLevel.MODERATE
        elif self.stability >= 0.2:
            return AudioStabilityLevel.UNSTABLE
        else:
            return AudioStabilityLevel.CRITICAL


@dataclass
class SpectralState:
    """Maintains spectral state for flux computation."""
    prev_magnitude_spectrum: Optional[np.ndarray] = None
    prev_energy: float = 0.0
    frame_count: int = 0


@dataclass
class AudioAxisProcessor:
    """
    Layer 14 Audio Axis FFT processor.

    Extracts telemetry features from audio signals without modifying
    the invariant hyperbolic metric.

    Features:
    - Frame energy (log scale)
    - Spectral centroid (brightness indicator)
    - Spectral flux (change detection)
    - High-frequency ratio (stability metric)
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame_size: int = 2048
    hop_size: int = 512
    hf_cutoff_ratio: float = HF_CUTOFF_RATIO
    risk_weight: float = 0.1  # w_a in risk integration

    _state: SpectralState = field(default_factory=SpectralState)

    def __post_init__(self):
        """Initialize frequency bins."""
        self._freq_bins = np.fft.rfftfreq(self.frame_size, 1.0 / self.sample_rate)
        self._nyquist = self.sample_rate / 2
        self._hf_cutoff_freq = self._nyquist * self.hf_cutoff_ratio
        self._hf_bin_start = int(len(self._freq_bins) * self.hf_cutoff_ratio)

    def reset_state(self):
        """Reset spectral state for new stream."""
        self._state = SpectralState()

    def compute_energy(self, frame: np.ndarray) -> float:
        """
        Compute frame energy: E_a = log(ε + Σ a[n]²)

        Logarithmic scale provides better dynamic range.
        """
        energy_sum = np.sum(frame ** 2)
        return math.log(EPSILON + energy_sum)

    def compute_spectrum(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitude and power spectrum via FFT.

        Returns:
            (magnitude_spectrum, power_spectrum)
        """
        # Apply Hann window to reduce spectral leakage
        windowed = frame * np.hanning(len(frame))

        # Zero-pad if necessary
        if len(windowed) < self.frame_size:
            windowed = np.pad(windowed, (0, self.frame_size - len(windowed)))

        # FFT
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum)
        power = magnitude ** 2

        return magnitude, power

    def compute_centroid(self, power_spectrum: np.ndarray) -> float:
        """
        Compute spectral centroid: C_a = (Σ fₖ·Pₐ[k]) / (Σ Pₐ[k])

        The centroid indicates the "brightness" of the sound.
        """
        total_power = np.sum(power_spectrum)
        if total_power < EPSILON:
            return 0.0

        weighted_sum = np.sum(self._freq_bins[:len(power_spectrum)] * power_spectrum)
        return weighted_sum / total_power

    def compute_flux(self, magnitude_spectrum: np.ndarray) -> float:
        """
        Compute spectral flux: F_a = Σ (√Pₐ[k] - √Pₐ_prev[k])²

        Measures spectral change between consecutive frames.
        """
        if self._state.prev_magnitude_spectrum is None:
            return 0.0

        # Align lengths
        min_len = min(len(magnitude_spectrum), len(self._state.prev_magnitude_spectrum))
        curr = magnitude_spectrum[:min_len]
        prev = self._state.prev_magnitude_spectrum[:min_len]

        # Spectral flux (using magnitude directly, equivalent to sqrt of power)
        diff = curr - prev
        flux = np.sum(diff ** 2)

        return float(flux)

    def compute_hf_ratio(self, power_spectrum: np.ndarray) -> float:
        """
        Compute high-frequency ratio: r_HF,a = Σₖ∈Khigh Pₐ[k] / Σ Pₐ[k]

        Measures proportion of energy in high-frequency bands.
        High r_HF indicates potentially unstable/anomalous signal.
        """
        total_power = np.sum(power_spectrum)
        if total_power < EPSILON:
            return 0.0

        hf_power = np.sum(power_spectrum[self._hf_bin_start:])
        return float(hf_power / total_power)

    def process_frame(self, frame: np.ndarray) -> AudioFeatures:
        """
        Process a single audio frame and extract features.

        Args:
            frame: Audio samples (mono, float normalized to [-1, 1])

        Returns:
            AudioFeatures with all computed metrics
        """
        # Ensure float array
        frame = np.asarray(frame, dtype=np.float64)

        # Compute energy
        energy = self.compute_energy(frame)

        # Compute spectrum
        magnitude, power = self.compute_spectrum(frame)

        # Compute features
        centroid = self.compute_centroid(power)
        flux = self.compute_flux(magnitude)
        hf_ratio = self.compute_hf_ratio(power)

        # Stability score
        stability = 1.0 - hf_ratio

        # Update state
        self._state.prev_magnitude_spectrum = magnitude.copy()
        self._state.prev_energy = energy
        self._state.frame_count += 1

        return AudioFeatures(
            energy=energy,
            centroid=centroid,
            flux=flux,
            hf_ratio=hf_ratio,
            stability=stability
        )

    def process_signal(self, signal: np.ndarray) -> List[AudioFeatures]:
        """
        Process entire audio signal with hop-based framing.

        Args:
            signal: Complete audio signal

        Returns:
            List of AudioFeatures for each frame
        """
        self.reset_state()
        features = []

        num_frames = 1 + (len(signal) - self.frame_size) // self.hop_size
        num_frames = max(0, num_frames)

        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            frame = signal[start:end]
            features.append(self.process_frame(frame))

        return features

    def integrate_risk(self, base_risk: float, features: AudioFeatures) -> float:
        """
        Integrate audio features into risk assessment.

        Risk' = Risk_base + w_a·(1 - S_audio)

        This adds audio instability to the base risk without
        modifying the hyperbolic metric.
        """
        audio_risk_contribution = self.risk_weight * (1.0 - features.stability)
        return min(1.0, base_risk + audio_risk_contribution)

    def aggregate_features(self, feature_list: List[AudioFeatures]) -> AudioFeatures:
        """
        Aggregate multiple frames into summary features.

        Uses mean for stable metrics, max for flux (peak change).
        """
        if not feature_list:
            return AudioFeatures(0.0, 0.0, 0.0, 0.0, 1.0)

        energies = [f.energy for f in feature_list]
        centroids = [f.centroid for f in feature_list]
        fluxes = [f.flux for f in feature_list]
        hf_ratios = [f.hf_ratio for f in feature_list]

        mean_energy = sum(energies) / len(energies)
        mean_centroid = sum(centroids) / len(centroids)
        max_flux = max(fluxes)  # Peak change
        mean_hf_ratio = sum(hf_ratios) / len(hf_ratios)
        mean_stability = 1.0 - mean_hf_ratio

        return AudioFeatures(
            energy=mean_energy,
            centroid=mean_centroid,
            flux=max_flux,
            hf_ratio=mean_hf_ratio,
            stability=mean_stability
        )


def verify_stability_bounded(features: AudioFeatures) -> bool:
    """
    Proof 1: Verify S_audio ∈ [0, 1].

    Since r_HF,a = HF_power / total_power and both are non-negative
    with HF_power ≤ total_power, we have r_HF,a ∈ [0, 1].
    Therefore S_audio = 1 - r_HF,a ∈ [0, 1].
    """
    return 0.0 <= features.stability <= 1.0


def verify_hf_detection(processor: AudioAxisProcessor) -> bool:
    """
    Proof 2: High-frequency signals produce high r_HF,a.

    Generate pure high-frequency tone and verify r_HF,a > 0.5.
    """
    # Generate high-frequency tone (3/4 of Nyquist)
    hf_freq = processor.sample_rate * 0.375  # 75% of Nyquist
    t = np.arange(processor.frame_size) / processor.sample_rate
    hf_signal = np.sin(2 * np.pi * hf_freq * t)

    features = processor.process_frame(hf_signal)
    return features.hf_ratio > 0.5


def verify_flux_sensitivity(processor: AudioAxisProcessor) -> bool:
    """
    Proof 3: Different consecutive frames produce flux > 0.

    Process two different frames and verify non-zero flux.
    """
    processor.reset_state()

    # First frame: low frequency
    t = np.arange(processor.frame_size) / processor.sample_rate
    frame1 = np.sin(2 * np.pi * 100 * t)
    processor.process_frame(frame1)

    # Second frame: high frequency
    frame2 = np.sin(2 * np.pi * 5000 * t)
    features2 = processor.process_frame(frame2)

    return features2.flux > 0


def generate_test_signal(
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frequency: float = 440.0,
    noise_level: float = 0.0
) -> np.ndarray:
    """
    Generate test signal for audio axis testing.

    Args:
        duration: Signal length in seconds
        sample_rate: Samples per second
        frequency: Base frequency (Hz)
        noise_level: Amount of white noise [0, 1]

    Returns:
        Audio signal array
    """
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate

    # Pure tone
    signal = np.sin(2 * np.pi * frequency * t)

    # Add noise if requested
    if noise_level > 0:
        noise = np.random.randn(num_samples) * noise_level
        signal = signal + noise
        # Normalize
        signal = signal / (1 + noise_level)

    return signal


@dataclass
class AudioAxisTelemetry:
    """
    High-level telemetry interface for Layer 14.

    Provides streaming telemetry with anomaly detection.
    """

    processor: AudioAxisProcessor = field(default_factory=AudioAxisProcessor)
    anomaly_threshold: float = 0.3  # S_audio below this triggers anomaly
    flux_threshold: float = 100.0   # Flux above this indicates sudden change

    _history: List[AudioFeatures] = field(default_factory=list)
    _anomaly_count: int = 0

    def ingest_frame(self, frame: np.ndarray) -> Tuple[AudioFeatures, bool]:
        """
        Process frame and check for anomalies.

        Returns:
            (features, is_anomaly)
        """
        features = self.processor.process_frame(frame)
        self._history.append(features)

        # Anomaly detection
        is_anomaly = (
            features.stability < self.anomaly_threshold or
            features.flux > self.flux_threshold
        )

        if is_anomaly:
            self._anomaly_count += 1

        return features, is_anomaly

    def get_summary(self) -> dict:
        """Get telemetry summary."""
        if not self._history:
            return {
                'frame_count': 0,
                'anomaly_count': 0,
                'mean_stability': 1.0,
                'max_flux': 0.0
            }

        return {
            'frame_count': len(self._history),
            'anomaly_count': self._anomaly_count,
            'mean_stability': sum(f.stability for f in self._history) / len(self._history),
            'max_flux': max(f.flux for f in self._history),
            'mean_energy': sum(f.energy for f in self._history) / len(self._history),
            'mean_centroid': sum(f.centroid for f in self._history) / len(self._history),
        }

    def reset(self):
        """Reset telemetry state."""
        self.processor.reset_state()
        self._history.clear()
        self._anomaly_count = 0


# Convenience exports
__all__ = [
    'EPSILON',
    'DEFAULT_SAMPLE_RATE',
    'HF_CUTOFF_RATIO',
    'AudioStabilityLevel',
    'AudioFeatures',
    'SpectralState',
    'AudioAxisProcessor',
    'AudioAxisTelemetry',
    'verify_stability_bounded',
    'verify_hf_detection',
    'verify_flux_sensitivity',
    'generate_test_signal',
]
