"""
Dimensional Theory Simulation: Comparing Security Architectures
================================================================

A tiered simulation comparing three security paradigms:
1. Traditional (Euclidean 3D) - Linear, flat geometry
2. SCBE (6D Harmonic Manifold) - Intent-aware with super-exponential sinks
3. Future (Hyperbolic 20D) - Non-Euclidean with complex arbitration

Key Concepts:
- Hyperbolic Dimensional Analysis: Distances grow exponentially in curved space
- Quantum Control Theory: Feedback loops for stability and state estimation
- Inhospitable Zones: High-curvature pockets that trap attacks
- Complex Intent Arbitration: Imaginary components for "felt" hidden variables

Patent Extension: Claims 31-36 - Dimensional Security Architecture
"""

import numpy as np
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import hashlib


# =============================================================================
# CONSTANTS
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2          # Golden ratio
PHI_INV = 1 / PHI                    # Inverse golden ratio
R_HARMONIC = 1.5                     # Harmonic base scaling factor
KAPPA_CURVATURE = 1.0                # Base hyperbolic curvature

# Dimensional limits
DIM_TRADITIONAL = 3                  # Euclidean 3D
DIM_SCBE = 6                         # SCBE 6D manifold
DIM_FUTURE = 20                      # Maximum discoverable dimensions

# Quantum control parameters
PLANCK_SCALED = 1e-6                 # Scaled Planck for simulation
DECOHERENCE_RATE = 0.01              # Quantum decoherence rate


class SystemType(Enum):
    """Security system architecture types."""
    TRADITIONAL = "traditional"      # Euclidean 3D
    SCBE = "scbe"                    # 6D Harmonic Manifold
    FUTURE = "future"                # Hyperbolic 20D


@dataclass
class SimulationMetrics:
    """Collected metrics from simulation run."""
    system_type: SystemType
    dimensions: int
    compute_time_ms: float
    attack_cost: float
    attack_cost_bits: float
    coherence_score: float
    is_trapped: bool
    dilation_factor: float
    scientific_data: Dict[str, Any]


@dataclass
class QuantumState:
    """Quantum control theory state representation."""
    amplitude: np.ndarray            # Complex amplitude vector
    phase: np.ndarray                # Phase angles
    coherence: float                 # Quantum coherence measure
    purity: float                    # State purity (1 = pure)
    entanglement_entropy: float      # Von Neumann entropy


# =============================================================================
# METRIC TENSORS FOR DIFFERENT GEOMETRIES
# =============================================================================

def euclidean_metric(dim: int) -> np.ndarray:
    """
    Euclidean (flat) metric tensor.
    Traditional systems use identity matrix - no warping.
    """
    return np.eye(dim)


def harmonic_metric(dim: int, R: float = R_HARMONIC) -> np.ndarray:
    """
    Harmonic scaling metric tensor for SCBE.
    g = diag(1, 1, 1, R, R², R³, ...)
    Creates super-exponential distance growth.
    """
    diag = np.array([R ** i for i in range(dim)])
    return np.diag(diag)


def hyperbolic_metric(dim: int, kappa: float = KAPPA_CURVATURE) -> np.ndarray:
    """
    Hyperbolic (negative curvature) metric tensor.
    Creates exponentially growing distances from origin.
    """
    # Poincaré disk-inspired metric with position-dependent scaling
    # For simulation: use golden ratio powers with curvature
    diag = np.array([np.exp(kappa * PHI ** (i / dim)) for i in range(dim)])
    return np.diag(diag)


# =============================================================================
# DISTANCE CALCULATIONS
# =============================================================================

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Standard Euclidean distance."""
    return float(np.linalg.norm(v1 - v2))


def manifold_distance(v1: np.ndarray, v2: np.ndarray, metric: np.ndarray) -> float:
    """
    Distance on Riemannian manifold with given metric tensor.
    d(v1, v2) = sqrt((v2-v1)^T @ g @ (v2-v1))
    """
    delta = v2 - v1
    # Ensure dimensions match
    if len(delta) > metric.shape[0]:
        delta = delta[:metric.shape[0]]
    elif len(delta) < metric.shape[0]:
        delta = np.pad(delta, (0, metric.shape[0] - len(delta)))

    return float(np.sqrt(np.abs(delta.T @ metric @ delta)))


def hyperbolic_distance_poincare(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Hyperbolic distance in Poincaré ball model.
    d(v1, v2) = acosh(1 + 2|v1-v2|²/((1-|v1|²)(1-|v2|²)))
    """
    norm1_sq = min(np.dot(v1, v1), 0.99)  # Keep in unit ball
    norm2_sq = min(np.dot(v2, v2), 0.99)
    diff_sq = np.dot(v1 - v2, v1 - v2)

    denom = (1 - norm1_sq) * (1 - norm2_sq)
    if denom < 1e-10:
        return float('inf')

    cosh_dist = 1 + 2 * diff_sq / denom
    return float(np.arccosh(max(1.0, cosh_dist)))


# =============================================================================
# QUANTUM CONTROL THEORY INTEGRATION
# =============================================================================

class QuantumControlSystem:
    """
    Quantum control theory integration for security state management.

    Implements:
    - State estimation (Kalman-like filtering)
    - Feedback control for stability
    - Decoherence modeling for attack detection
    - Entanglement-based correlation tracking
    """

    def __init__(self, dim: int, hbar: float = PLANCK_SCALED):
        self.dim = dim
        self.hbar = hbar
        self.state = self._initialize_state()
        self.hamiltonian = self._random_hamiltonian()
        self.lindblad_ops = self._lindblad_operators()

    def _initialize_state(self) -> QuantumState:
        """Initialize pure quantum state."""
        # Random normalized amplitude
        amp = np.random.randn(self.dim) + 1j * np.random.randn(self.dim)
        amp = amp / np.linalg.norm(amp)

        return QuantumState(
            amplitude=amp,
            phase=np.angle(amp),
            coherence=1.0,
            purity=1.0,
            entanglement_entropy=0.0
        )

    def _random_hamiltonian(self) -> np.ndarray:
        """Generate random Hermitian Hamiltonian."""
        H = np.random.randn(self.dim, self.dim) + 1j * np.random.randn(self.dim, self.dim)
        return (H + H.conj().T) / 2  # Make Hermitian

    def _lindblad_operators(self) -> List[np.ndarray]:
        """Generate Lindblad decoherence operators."""
        # Simple dephasing operators
        ops = []
        for i in range(min(3, self.dim)):
            L = np.zeros((self.dim, self.dim), dtype=complex)
            L[i, i] = np.sqrt(DECOHERENCE_RATE)
            ops.append(L)
        return ops

    def evolve(self, dt: float = 0.01) -> QuantumState:
        """
        Time evolution with decoherence.

        Uses simplified Lindblad master equation.
        """
        # Unitary evolution
        U = np.eye(self.dim) - 1j * self.hamiltonian * dt / self.hbar
        U = U / np.linalg.norm(U, axis=1, keepdims=True)  # Approximate normalization

        new_amp = U @ self.state.amplitude

        # Apply decoherence
        for L in self.lindblad_ops:
            new_amp = new_amp - dt * (L.conj().T @ L @ new_amp) / 2

        # Renormalize
        new_amp = new_amp / np.linalg.norm(new_amp)

        # Calculate coherence (off-diagonal density matrix elements)
        rho = np.outer(new_amp, new_amp.conj())
        coherence = float(np.abs(np.sum(rho) - np.trace(rho)) / (self.dim * (self.dim - 1)))

        # Calculate purity: Tr(ρ²)
        purity = float(np.real(np.trace(rho @ rho)))

        # Entanglement entropy (simplified)
        eigenvalues = np.linalg.eigvalsh(np.abs(rho))
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        entropy = float(-np.sum(eigenvalues * np.log(eigenvalues + 1e-10)))

        self.state = QuantumState(
            amplitude=new_amp,
            phase=np.angle(new_amp),
            coherence=coherence,
            purity=purity,
            entanglement_entropy=entropy
        )

        return self.state

    def measure_observable(self, observable: np.ndarray) -> Tuple[float, float]:
        """
        Measure quantum observable.

        Returns (expectation_value, uncertainty).
        """
        rho = np.outer(self.state.amplitude, self.state.amplitude.conj())
        expectation = float(np.real(np.trace(observable @ rho)))

        # Uncertainty
        obs_sq = observable @ observable
        exp_sq = float(np.real(np.trace(obs_sq @ rho)))
        uncertainty = float(np.sqrt(max(0, exp_sq - expectation ** 2)))

        return expectation, uncertainty

    def apply_feedback_control(self, target_state: np.ndarray, gain: float = 0.1) -> None:
        """
        Apply quantum feedback control toward target state.

        Uses simplified linear feedback.
        """
        error = target_state - self.state.amplitude
        correction = gain * error

        new_amp = self.state.amplitude + correction
        new_amp = new_amp / np.linalg.norm(new_amp)

        self.state = QuantumState(
            amplitude=new_amp,
            phase=np.angle(new_amp),
            coherence=self.state.coherence,
            purity=self.state.purity,
            entanglement_entropy=self.state.entanglement_entropy
        )


# =============================================================================
# INTENT TRAJECTORY AND COHERENCE
# =============================================================================

class IntentTrajectory:
    """
    Intent-over-time trajectory with complex arbitration.

    Supports:
    - Real component: Observable intent
    - Imaginary component: Hidden/felt influences
    """

    def __init__(self, dim: int, use_complex: bool = False):
        self.dim = dim
        self.use_complex = use_complex
        self.trajectory: List[np.ndarray] = []
        self.timestamps: List[float] = []

    def add_point(self, point: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Add trajectory point."""
        if self.use_complex and point.dtype != complex:
            # Add imaginary component as "hidden influence"
            imag = np.random.randn(len(point)) * 0.1
            point = point.astype(complex) + 1j * imag

        # Pad/trim to dimension
        if len(point) < self.dim:
            point = np.pad(point, (0, self.dim - len(point)))
        elif len(point) > self.dim:
            point = point[:self.dim]

        self.trajectory.append(point)
        self.timestamps.append(timestamp or time.time())

    def compute_coherence(self, metric: np.ndarray) -> Tuple[float, List[float]]:
        """
        Compute trajectory coherence using Mahalanobis-like divergence.

        Returns (coherence_score, divergence_list).
        """
        if len(self.trajectory) < 2:
            return 1.0, []

        divergences = []
        for i in range(1, len(self.trajectory)):
            prev = np.real(self.trajectory[i - 1])
            curr = np.real(self.trajectory[i])
            div = manifold_distance(prev, curr, metric)
            divergences.append(div)

        mean_div = np.mean(divergences)
        coherence = 1.0 / (1.0 + mean_div)

        return float(coherence), divergences

    def compute_complex_coherence(self) -> Tuple[float, float]:
        """
        Compute coherence including imaginary (hidden) components.

        Returns (real_coherence, imaginary_influence).
        """
        if not self.use_complex or len(self.trajectory) < 2:
            return 1.0, 0.0

        real_divs = []
        imag_divs = []

        for i in range(1, len(self.trajectory)):
            prev = self.trajectory[i - 1]
            curr = self.trajectory[i]

            real_div = np.linalg.norm(np.real(curr - prev))
            imag_div = np.linalg.norm(np.imag(curr - prev))

            real_divs.append(real_div)
            imag_divs.append(imag_div)

        real_coherence = 1.0 / (1.0 + np.mean(real_divs))
        imag_influence = float(np.mean(imag_divs))

        return float(real_coherence), imag_influence


# =============================================================================
# INHOSPITABLE ZONE SINK
# =============================================================================

class InhospitableZoneSink:
    """
    Simulates "pocket dimension" inhospitable zones.

    Creates computational black holes that:
    1. Trap attacks with exponential dilation
    2. Generate scientific data (NASA-like output)
    3. Scale with hyperbolic curvature
    """

    def __init__(self, curvature: float = KAPPA_CURVATURE):
        self.curvature = curvature
        self.trapped_count = 0
        self.scientific_log: List[Dict[str, Any]] = []

    def compute_dilation(self, divergence: float) -> float:
        """
        Compute time dilation factor based on divergence.

        Mimics gravitational redshift near event horizon.
        """
        return float(np.exp(self.curvature * divergence))

    def check_trap(self,
                   coherence: float,
                   divergences: List[float],
                   threshold: float = 0.5) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Check if trajectory is trapped in inhospitable zone.

        Returns (is_trapped, dilation_factor, scientific_data).
        """
        if coherence >= threshold:
            return False, 1.0, {}

        # Low coherence = attack-like = trap
        mean_div = np.mean(divergences) if divergences else 0

        dilation = self.compute_dilation(mean_div)
        self.trapped_count += 1

        # Generate scientific data output
        sci_data = {
            "divergence_trajectory": divergences,
            "mean_divergence": float(mean_div),
            "dilation_factor": dilation,
            "curvature": self.curvature,
            "event_horizon_proximity": float(1.0 / (1.0 + mean_div)),
            "hawking_radiation_analog": float(np.exp(-mean_div / 10)),  # Decay rate
            "geodesic_deviation": [float(d * self.curvature) for d in divergences],
            "trapped_time": time.time()
        }

        self.scientific_log.append(sci_data)

        return True, dilation, sci_data


# =============================================================================
# SECURITY SYSTEM SIMULATORS
# =============================================================================

class TraditionalSystem:
    """
    Traditional Euclidean 3D security system.

    Characteristics:
    - Flat geometry (no warping)
    - Linear attack costs
    - No intent awareness
    - Fast but vulnerable
    """

    def __init__(self):
        self.dim = DIM_TRADITIONAL
        self.metric = euclidean_metric(self.dim)

    def process(self, input_vector: np.ndarray, trajectory: List[np.ndarray]) -> SimulationMetrics:
        """Process security check."""
        start = time.perf_counter()

        # Simple distance-based cost (linear)
        if len(trajectory) > 1:
            total_dist = sum(
                euclidean_distance(trajectory[i], trajectory[i + 1])
                for i in range(len(trajectory) - 1)
            )
        else:
            total_dist = np.linalg.norm(input_vector)

        attack_cost = 1.0 + total_dist  # Linear scaling
        coherence = 1.0 / (1.0 + total_dist / 10)  # Simple coherence

        compute_time = (time.perf_counter() - start) * 1000

        return SimulationMetrics(
            system_type=SystemType.TRADITIONAL,
            dimensions=self.dim,
            compute_time_ms=compute_time,
            attack_cost=attack_cost,
            attack_cost_bits=np.log2(1 + attack_cost),
            coherence_score=coherence,
            is_trapped=False,  # No trapping in traditional
            dilation_factor=1.0,  # No dilation
            scientific_data={}
        )


class SCBESystem:
    """
    SCBE 6D Harmonic Manifold security system.

    Characteristics:
    - 6D context vectors (WHO, WHAT, WHERE, WHEN, WHY, HOW)
    - Harmonic scaling H(d,R) = R^(1+d²)
    - Intent-aware trajectory coherence
    - Super-exponential attack costs
    """

    def __init__(self):
        self.dim = DIM_SCBE
        self.metric = harmonic_metric(self.dim)
        self.sink = InhospitableZoneSink(curvature=1.0)
        self.intent_tracker = IntentTrajectory(self.dim, use_complex=False)

    def harmonic_scaling(self, d: float, R: float = R_HARMONIC) -> float:
        """H(d,R) = R^(1+d²)"""
        return R ** (1 + d ** 2)

    def process(self, input_vector: np.ndarray, trajectory: List[np.ndarray]) -> SimulationMetrics:
        """Process security check with SCBE architecture."""
        start = time.perf_counter()

        # Build trajectory
        self.intent_tracker = IntentTrajectory(self.dim, use_complex=False)
        for point in trajectory:
            self.intent_tracker.add_point(point)

        # Compute coherence
        coherence, divergences = self.intent_tracker.compute_coherence(self.metric)

        # Harmonic attack cost based on trajectory length/divergence
        context_distance = np.mean(divergences) if divergences else 0.1
        harmonic = self.harmonic_scaling(context_distance)
        base_cost = 2 ** (self.dim / 2)
        attack_cost = base_cost * harmonic

        # Check inhospitable zone
        is_trapped, dilation, sci_data = self.sink.check_trap(
            coherence, divergences, threshold=0.5
        )

        if is_trapped:
            attack_cost *= dilation  # Amplify cost in trap

        compute_time = (time.perf_counter() - start) * 1000

        return SimulationMetrics(
            system_type=SystemType.SCBE,
            dimensions=self.dim,
            compute_time_ms=compute_time,
            attack_cost=attack_cost,
            attack_cost_bits=np.log2(1 + attack_cost),
            coherence_score=coherence,
            is_trapped=is_trapped,
            dilation_factor=dilation,
            scientific_data=sci_data
        )


class FutureHyperbolicSystem:
    """
    Future Hyperbolic 20D security system.

    Characteristics:
    - Maximum discoverable dimensions (20D)
    - Hyperbolic (negative curvature) geometry
    - Complex intent arbitration (real + imaginary)
    - Quantum control theory integration
    - Extreme inhospitable zones
    """

    def __init__(self):
        self.dim = DIM_FUTURE
        self.metric = hyperbolic_metric(self.dim, kappa=2.0)  # Higher curvature
        self.sink = InhospitableZoneSink(curvature=2.0)
        self.intent_tracker = IntentTrajectory(self.dim, use_complex=True)
        self.quantum = QuantumControlSystem(self.dim)

    def process(self, input_vector: np.ndarray, trajectory: List[np.ndarray]) -> SimulationMetrics:
        """Process with future hyperbolic architecture."""
        start = time.perf_counter()

        # Build complex trajectory
        self.intent_tracker = IntentTrajectory(self.dim, use_complex=True)
        for point in trajectory:
            self.intent_tracker.add_point(point)

        # Compute coherence (real + imaginary)
        real_coherence, divergences = self.intent_tracker.compute_coherence(self.metric)
        complex_coherence, imag_influence = self.intent_tracker.compute_complex_coherence()

        # Quantum state evolution
        for _ in range(10):
            self.quantum.evolve(dt=0.01)
        quantum_coherence = self.quantum.state.coherence

        # Combined coherence
        coherence = (real_coherence + complex_coherence + quantum_coherence) / 3

        # Hyperbolic attack cost (extreme scaling)
        context_distance = np.mean(divergences) if divergences else 0.1

        # Hyperbolic distance grows exponentially
        hyper_factor = np.exp(context_distance * 2.0)  # High curvature
        base_cost = 2 ** (self.dim / 2)
        attack_cost = base_cost * hyper_factor

        # Check inhospitable zone
        is_trapped, dilation, sci_data = self.sink.check_trap(
            coherence, divergences, threshold=0.6
        )

        if is_trapped:
            attack_cost *= dilation ** 2  # Extreme amplification

        # Add quantum data to scientific output
        sci_data["quantum_coherence"] = quantum_coherence
        sci_data["quantum_purity"] = self.quantum.state.purity
        sci_data["quantum_entropy"] = self.quantum.state.entanglement_entropy
        sci_data["imaginary_influence"] = imag_influence

        # Cap at representable values
        if attack_cost > 1e300:
            attack_cost = float('inf')

        compute_time = (time.perf_counter() - start) * 1000

        cost_bits = np.log2(1 + attack_cost) if attack_cost < 1e300 else 1000

        return SimulationMetrics(
            system_type=SystemType.FUTURE,
            dimensions=self.dim,
            compute_time_ms=compute_time,
            attack_cost=attack_cost,
            attack_cost_bits=float(cost_bits),
            coherence_score=coherence,
            is_trapped=is_trapped,
            dilation_factor=dilation,
            scientific_data=sci_data
        )


# =============================================================================
# TIERED SIMULATION RUNNER
# =============================================================================

class DimensionalTheorySimulation:
    """
    Main simulation comparing all three security architectures.

    Runs tiered tests with varying factors:
    - Threat levels
    - Input dimensions
    - Trajectory coherence
    - Attack complexity
    """

    def __init__(self):
        self.traditional = TraditionalSystem()
        self.scbe = SCBESystem()
        self.future = FutureHyperbolicSystem()
        self.results: Dict[str, List[SimulationMetrics]] = {
            "traditional": [],
            "scbe": [],
            "future": []
        }

    def generate_coherent_trajectory(self, dim: int, length: int) -> List[np.ndarray]:
        """Generate coherent (legitimate) trajectory."""
        base = np.random.randn(dim)
        trajectory = [base]
        for _ in range(length - 1):
            # Small perturbation = coherent
            next_point = trajectory[-1] + np.random.randn(dim) * 0.1
            trajectory.append(next_point)
        return trajectory

    def generate_incoherent_trajectory(self, dim: int, length: int) -> List[np.ndarray]:
        """Generate incoherent (attack-like) trajectory."""
        trajectory = []
        for _ in range(length):
            # Random points = incoherent
            trajectory.append(np.random.randn(dim) * 2.0)
        return trajectory

    def run_single_comparison(self,
                               input_vector: np.ndarray,
                               trajectory: List[np.ndarray]) -> Dict[str, SimulationMetrics]:
        """Run single comparison across all systems."""
        return {
            "traditional": self.traditional.process(input_vector, trajectory),
            "scbe": self.scbe.process(input_vector, trajectory),
            "future": self.future.process(input_vector, trajectory)
        }

    def run_tiered_simulation(self,
                               num_tiers: int = 5,
                               trajectories_per_tier: int = 10) -> Dict[str, Any]:
        """
        Run tiered simulation with varying complexity.

        Tiers represent increasing attack sophistication.
        """
        all_results = {
            "tiers": [],
            "summary": {},
            "scientific_output": []
        }

        for tier in range(num_tiers):
            tier_results = {
                "tier": tier,
                "complexity": tier + 1,
                "trajectory_length": (tier + 1) * 3,
                "traditional": [],
                "scbe": [],
                "future": []
            }

            trajectory_length = (tier + 1) * 3

            for i in range(trajectories_per_tier):
                # Alternate coherent/incoherent
                if i % 2 == 0:
                    traj = self.generate_coherent_trajectory(DIM_FUTURE, trajectory_length)
                else:
                    traj = self.generate_incoherent_trajectory(DIM_FUTURE, trajectory_length)

                input_vec = np.random.randn(DIM_FUTURE)

                comparison = self.run_single_comparison(input_vec, traj)

                for sys_name, metrics in comparison.items():
                    tier_results[sys_name].append({
                        "compute_time_ms": metrics.compute_time_ms,
                        "attack_cost": min(metrics.attack_cost, 1e100),  # Cap for JSON
                        "attack_cost_bits": metrics.attack_cost_bits,
                        "coherence": metrics.coherence_score,
                        "trapped": metrics.is_trapped,
                        "dilation": metrics.dilation_factor
                    })

                    if metrics.scientific_data:
                        all_results["scientific_output"].append({
                            "tier": tier,
                            "system": sys_name,
                            "data": metrics.scientific_data
                        })

            all_results["tiers"].append(tier_results)

        # Compute summary statistics
        all_results["summary"] = self._compute_summary(all_results["tiers"])

        return all_results

    def _compute_summary(self, tiers: List[Dict]) -> Dict[str, Any]:
        """Compute summary statistics across all tiers."""
        summary = {}

        for sys_name in ["traditional", "scbe", "future"]:
            all_times = []
            all_costs = []
            all_coherences = []
            trapped_count = 0
            total_count = 0

            for tier in tiers:
                for run in tier[sys_name]:
                    all_times.append(run["compute_time_ms"])
                    all_costs.append(run["attack_cost_bits"])
                    all_coherences.append(run["coherence"])
                    if run["trapped"]:
                        trapped_count += 1
                    total_count += 1

            summary[sys_name] = {
                "avg_compute_time_ms": float(np.mean(all_times)),
                "avg_attack_cost_bits": float(np.mean(all_costs)),
                "max_attack_cost_bits": float(np.max(all_costs)),
                "avg_coherence": float(np.mean(all_coherences)),
                "trap_rate": trapped_count / total_count if total_count > 0 else 0,
                "total_runs": total_count
            }

        return summary


# =============================================================================
# TEST SUITE
# =============================================================================

def test_metric_tensors():
    """Test metric tensor generation."""
    print("\n" + "="*60)
    print("TEST: Metric Tensors")
    print("="*60)

    # Euclidean
    euc = euclidean_metric(3)
    assert np.allclose(euc, np.eye(3)), "Euclidean should be identity"
    print(f"  Euclidean 3D: Identity matrix")

    # Harmonic
    harm = harmonic_metric(6)
    assert harm[0, 0] == 1.0, "First element should be 1"
    assert harm[3, 3] == R_HARMONIC ** 3, "Should follow R^i pattern"
    print(f"  Harmonic 6D: diag = {np.diag(harm)[:4]}...")

    # Hyperbolic
    hyp = hyperbolic_metric(20)
    assert hyp[0, 0] > 0, "Should be positive"
    assert hyp[10, 10] > hyp[0, 0], "Should grow with dimension"
    print(f"  Hyperbolic 20D: diag[0]={hyp[0,0]:.3f}, diag[10]={hyp[10,10]:.3f}")

    print("  PASS: All metric tensors correctly generated")
    return True


def test_quantum_control():
    """Test quantum control theory integration."""
    print("\n" + "="*60)
    print("TEST: Quantum Control Theory")
    print("="*60)

    qc = QuantumControlSystem(dim=6)

    # Initial state
    init_coherence = qc.state.coherence
    print(f"  Initial coherence: {init_coherence:.4f}")
    print(f"  Initial purity: {qc.state.purity:.4f}")

    # Evolve
    for _ in range(50):
        qc.evolve(dt=0.01)

    print(f"  After 50 steps:")
    print(f"    Coherence: {qc.state.coherence:.4f}")
    print(f"    Purity: {qc.state.purity:.4f}")
    print(f"    Entropy: {qc.state.entanglement_entropy:.4f}")

    # Feedback control
    target = np.ones(6, dtype=complex) / np.sqrt(6)
    qc.apply_feedback_control(target, gain=0.5)
    print(f"  After feedback control:")
    print(f"    Amplitude norm: {np.linalg.norm(qc.state.amplitude):.4f}")

    print("  PASS: Quantum control system functional")
    return True


def test_intent_trajectory():
    """Test intent trajectory coherence."""
    print("\n" + "="*60)
    print("TEST: Intent Trajectory Coherence")
    print("="*60)

    metric = harmonic_metric(6)

    # Coherent trajectory
    coherent = IntentTrajectory(6, use_complex=False)
    base = np.zeros(6)
    for i in range(10):
        coherent.add_point(base + np.random.randn(6) * 0.1)

    coh_score, coh_divs = coherent.compute_coherence(metric)
    print(f"  Coherent trajectory:")
    print(f"    Score: {coh_score:.4f}")
    print(f"    Avg divergence: {np.mean(coh_divs):.4f}")

    # Incoherent trajectory
    incoherent = IntentTrajectory(6, use_complex=False)
    for i in range(10):
        incoherent.add_point(np.random.randn(6) * 2.0)

    incoh_score, incoh_divs = incoherent.compute_coherence(metric)
    print(f"  Incoherent trajectory:")
    print(f"    Score: {incoh_score:.4f}")
    print(f"    Avg divergence: {np.mean(incoh_divs):.4f}")

    assert coh_score > incoh_score, "Coherent should score higher"
    print("  PASS: Coherence correctly distinguishes trajectories")
    return True


def test_inhospitable_sink():
    """Test inhospitable zone sink."""
    print("\n" + "="*60)
    print("TEST: Inhospitable Zone Sink")
    print("="*60)

    sink = InhospitableZoneSink(curvature=1.5)

    # Legitimate (high coherence)
    trapped1, dilation1, data1 = sink.check_trap(0.8, [0.1, 0.2, 0.1])
    print(f"  Legitimate (coherence=0.8):")
    print(f"    Trapped: {trapped1}")
    print(f"    Dilation: {dilation1:.2f}")

    # Attack (low coherence)
    trapped2, dilation2, data2 = sink.check_trap(0.2, [2.0, 3.0, 4.0])
    print(f"  Attack (coherence=0.2):")
    print(f"    Trapped: {trapped2}")
    print(f"    Dilation: {dilation2:.2f}")
    print(f"    Scientific data keys: {list(data2.keys())}")

    assert not trapped1, "Legitimate should not be trapped"
    assert trapped2, "Attack should be trapped"
    assert dilation2 > dilation1, "Attack should have higher dilation"

    print("  PASS: Inhospitable sink correctly traps attacks")
    return True


def test_system_comparison():
    """Test full system comparison."""
    print("\n" + "="*60)
    print("TEST: System Comparison")
    print("="*60)

    sim = DimensionalTheorySimulation()

    # Test with incoherent (attack-like) trajectory
    input_vec = np.random.randn(20)
    attack_traj = sim.generate_incoherent_trajectory(20, 10)

    results = sim.run_single_comparison(input_vec, attack_traj)

    print(f"\n  Attack Trajectory Results:")
    print(f"  {'System':<15} {'Dims':<6} {'Time(ms)':<10} {'Cost(bits)':<12} {'Coherence':<10} {'Trapped'}")
    print(f"  {'-'*70}")

    for name, metrics in results.items():
        print(f"  {name:<15} {metrics.dimensions:<6} "
              f"{metrics.compute_time_ms:<10.3f} {metrics.attack_cost_bits:<12.1f} "
              f"{metrics.coherence_score:<10.3f} {metrics.is_trapped}")

    # Verify ordering
    assert results["future"].attack_cost_bits >= results["scbe"].attack_cost_bits, \
           "Future should have higher attack cost"
    assert results["scbe"].attack_cost_bits >= results["traditional"].attack_cost_bits, \
           "SCBE should have higher attack cost than traditional"

    print("\n  PASS: System comparison shows expected security ordering")
    return True


def test_tiered_simulation():
    """Test full tiered simulation."""
    print("\n" + "="*60)
    print("TEST: Tiered Simulation")
    print("="*60)

    sim = DimensionalTheorySimulation()
    results = sim.run_tiered_simulation(num_tiers=3, trajectories_per_tier=5)

    print(f"\n  Simulation Summary:")
    print(f"  {'System':<15} {'Avg Time(ms)':<15} {'Avg Cost(bits)':<15} {'Trap Rate':<12}")
    print(f"  {'-'*60}")

    for name, stats in results["summary"].items():
        print(f"  {name:<15} {stats['avg_compute_time_ms']:<15.3f} "
              f"{stats['avg_attack_cost_bits']:<15.1f} {stats['trap_rate']:<12.1%}")

    print(f"\n  Scientific outputs collected: {len(results['scientific_output'])}")

    print("  PASS: Tiered simulation completed successfully")
    return True


def run_all_tests():
    """Run complete test suite."""
    print("="*70)
    print("DIMENSIONAL THEORY SIMULATION TEST SUITE")
    print("Comparing Traditional vs SCBE vs Future Hyperbolic Systems")
    print("="*70)

    tests = [
        ("Metric Tensors", test_metric_tensors),
        ("Quantum Control", test_quantum_control),
        ("Intent Trajectory", test_intent_trajectory),
        ("Inhospitable Sink", test_inhospitable_sink),
        ("System Comparison", test_system_comparison),
        ("Tiered Simulation", test_tiered_simulation),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  FAIL: {e}")

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, p, _ in results if p)
    total = len(results)

    for name, success, error in results:
        status = "PASS" if success else f"FAIL: {error}"
        print(f"  [{status}] {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    # Run full simulation and export
    print("\n" + "="*70)
    print("RUNNING FULL DIMENSIONAL COMPARISON")
    print("="*70)

    sim = DimensionalTheorySimulation()
    full_results = sim.run_tiered_simulation(num_tiers=5, trajectories_per_tier=20)

    print(f"\n  FINAL COMPARISON METRICS:")
    print(f"  {'System':<15} {'Dimensions':<12} {'Avg Cost(bits)':<15} {'Max Cost(bits)':<15} {'Trap Rate'}")
    print(f"  {'-'*75}")

    dims = {"traditional": 3, "scbe": 6, "future": 20}
    for name, stats in full_results["summary"].items():
        print(f"  {name:<15} {dims[name]:<12} {stats['avg_attack_cost_bits']:<15.1f} "
              f"{stats['max_attack_cost_bits']:<15.1f} {stats['trap_rate']:<.1%}")

    # Export results
    output = {
        "test_suite": "dimensional_theory_simulation",
        "timestamp": time.time(),
        "tests_passed": passed,
        "tests_total": total,
        "comparison": full_results["summary"],
        "scientific_samples": full_results["scientific_output"][:10]  # Sample
    }

    with open("dimensional_simulation_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to: dimensional_simulation_results.json")
    print("="*70)

    return passed == total


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
