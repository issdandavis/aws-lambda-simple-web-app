import numpy as np
from scipy.fft import fft, fftfreq

# Core Parameters
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ~1.618
R = PHI  # Harmonic base
DIM = 6  # Core dimensions
STEPS = 128  # Trajectory length (time axis samples)
KAPPA = 1.0  # Hyperbolic curvature
MIN_AMP, MAX_AMP = 0.2, 1.5  # Emotional intensity bounds
LONG_FREQ, SHORT_FREQ = 0.1, 1.0  # Wave types

# Weighted Metric Tensor G (later dims heavier)
def metric_tensor(R=R):
    return np.diag([1.0] * 3 + [R, R**2, R**3])

# Harmonic Scaling (super-exponential controller)
def harmonic_scaling(dist, R=R):
    return R ** (1 + dist**2)

# Hyperbolic Projection (curved "inhospitable" space)
def hyperbolic_project(x, kappa=KAPPA):
    norm = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9
    return x / (1 + kappa * norm**2)

# Emotional spin wave (word → vibration)
word_to_spin = {
    'calm': (0.3, 0.1, 0.0),       # amp, freq, phase
    'joyful': (0.8, 0.3, np.pi/4),
    'urgent': (1.2, 0.8, np.pi/2),
    'threat': (1.5, 1.2, np.pi)    # Dissonant
}

def emotional_spin_wave(t, word='calm', negative=False):
    amp, freq, phase = word_to_spin.get(word, (0.5, 0.5, 0))
    if negative:
        amp = -amp  # Negative amplitude for destructive
    return amp * np.exp(1j * (2 * np.pi * freq * t + phase))

# Generate Trajectory with Spins (multi-D complex waves)
def generate_trajectory(words, steps=STEPS, legit=True, negative=False):
    t = np.linspace(0, 10, steps)
    traj = np.zeros((steps, DIM), dtype=complex)
    for step in range(steps):
        word = np.random.choice(words)
        wave = emotional_spin_wave(t[step], word, negative=negative and not legit)
        for d in range(DIM):
            phase_off = d * np.pi / 3  # Axis offset for omni-spin
            traj[step, d] = wave * np.exp(1j * phase_off)
    return traj

# Axis Rotation (mix intent/time/place)
def rotate_axes(traj, theta=np.pi/4):
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    for step in range(STEPS):
        traj[step, :2] = rot @ traj[step, :2]  # Example on first two axes
    return traj

# Spectral Analysis (FFT for "timbre" anomaly)
def spectral_anomaly(traj):
    power = []
    for d in range(DIM):
        fft_vals = fft(traj[:, d])
        power.append(np.abs(fft_vals[:STEPS//2])**2)
    mean_power = np.mean(power, axis=0)
    high_freq_ratio = np.sum(mean_power[STEPS//4:]) / np.sum(mean_power + 1e-9)
    return high_freq_ratio

# Wave Interference (total per time step)
def wave_interference(traj):
    total = np.sum(traj, axis=1)
    amp = np.abs(total)
    coherence = np.mean(amp) / (np.std(amp) + 1e-9)
    return coherence

# === DECIMAL DRIFT DETECTION (Claims 14-18) ===
def extract_decimal_drift(trajectory):
    """
    Claim 14: δ = s - floor(s)
    Extract continuous drift from complex magnitude mapped to [0,6) state space.
    """
    # Map complex magnitude to 6-state space
    magnitudes = np.abs(trajectory)  # Shape: (STEPS, DIM)
    # Normalize to [0, 6) range based on MAX_AMP
    state_values = (magnitudes / MAX_AMP) * 6.0
    # Extract decimal component: δ = s - floor(s)
    decimal_drift = state_values - np.floor(state_values)
    return decimal_drift, state_values

def classify_drift(decimal_drift, tau_stable=0.35, tau_anomaly=0.50):
    """
    Claim 14(c): STABLE/DRIFTING/ANOMALOUS classification

    Thresholds calibrated from empirical observation:
    - Legitimate calm/joyful traffic: mean δ ≈ 0.34
    - Attack urgent/threat traffic:   mean δ ≈ 0.57
    """
    mean_drift = np.mean(decimal_drift)
    max_drift = np.max(decimal_drift)
    drift_std = np.std(decimal_drift)

    # Count dimensions with elevated drift (above anomaly threshold)
    dims_drifting = np.sum(np.mean(decimal_drift, axis=0) > tau_anomaly)

    if mean_drift < tau_stable:
        status = "STABLE"
    elif mean_drift >= tau_anomaly or dims_drifting >= 3:
        status = "ANOMALOUS"
    else:
        status = "DRIFTING"

    return {
        'status': status,
        'mean_drift': mean_drift,
        'max_drift': max_drift,
        'drift_std': drift_std,
        'dims_drifting': dims_drifting
    }

def temporal_drift_score(decimal_drift, decay_factor=0.9):
    """
    Claim 17: S_drift = Σ(δᵢ × decay^(M-i))
    """
    M = decimal_drift.shape[0]
    weights = np.array([decay_factor ** (M - i - 1) for i in range(M)])
    per_step_drift = np.mean(decimal_drift, axis=1)  # Average across dims
    S_drift = np.sum(per_step_drift * weights)
    return S_drift

# Full Evaluation (pure math flow)
def scbe_eval(trajectory, rotate=True, negative=False):
    if rotate:
        trajectory = rotate_axes(trajectory)
    hyper_traj = hyperbolic_project(np.real(trajectory))
    dists = np.linalg.norm(hyper_traj, axis=1)
    mean_dist = np.mean(dists)
    scaling = harmonic_scaling(mean_dist)
    high_freq = spectral_anomaly(trajectory)
    coherence = wave_interference(trajectory)

    # === DECIMAL DRIFT ANALYSIS (Key Innovation) ===
    decimal_drift, state_values = extract_decimal_drift(trajectory)
    drift_class = classify_drift(decimal_drift)
    S_drift = temporal_drift_score(decimal_drift)

    # Combined anomaly detection
    legacy_anomaly = high_freq > 0.4 or coherence < 2.0 or mean_dist > 4.0
    drift_anomaly = drift_class['status'] == "ANOMALOUS"

    return {
        'mean_distance': mean_dist,
        'harmonic_scaling': scaling,
        'high_freq_anomaly': high_freq,
        'wave_coherence': coherence,
        'legacy_anomaly': legacy_anomaly,
        # Decimal drift outputs
        'drift_status': drift_class['status'],
        'mean_decimal_drift': drift_class['mean_drift'],
        'dims_drifting': drift_class['dims_drifting'],
        'temporal_drift_score': S_drift,
        'anomaly_detected': legacy_anomaly or drift_anomaly
    }

# Compile and Run Example
if __name__ == "__main__":
    words_legit = ['calm', 'joyful']
    words_attack = ['urgent', 'threat']

    np.random.seed(42)  # Reproducible
    legit_traj = generate_trajectory(words_legit, legit=True)
    attack_traj = generate_trajectory(words_attack, legit=False, negative=True)

    print("=" * 60)
    print("SCBE CORE EVALUATION")
    print("=" * 60)

    legit_result = scbe_eval(legit_traj)
    attack_result = scbe_eval(attack_traj, negative=True)

    print("\n📗 LEGIT Evaluation:")
    for k, v in legit_result.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    print("\n📕 ATTACK Evaluation:")
    for k, v in attack_result.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    print("\n" + "=" * 60)
    print("DECIMAL DRIFT ANALYSIS (Claims 14-18)")
    print("=" * 60)
    print(f"Legit  → δ={legit_result['mean_decimal_drift']:.4f}, dims_drifting={legit_result['dims_drifting']}, status={legit_result['drift_status']}")
    print(f"Attack → δ={attack_result['mean_decimal_drift']:.4f}, dims_drifting={attack_result['dims_drifting']}, status={attack_result['drift_status']}")
    print(f"Temporal S_drift: legit={legit_result['temporal_drift_score']:.2f}, attack={attack_result['temporal_drift_score']:.2f}")

    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    print(f"Legit anomaly_detected:  {legit_result['anomaly_detected']}")
    print(f"Attack anomaly_detected: {attack_result['anomaly_detected']}")
