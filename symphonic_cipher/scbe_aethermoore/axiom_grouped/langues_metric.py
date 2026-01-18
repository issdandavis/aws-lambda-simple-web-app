"""
Langues Metric Module - Six Sacred Tongues Governance

Implements the 6D phase-shifted exponential cost function for SCBE-AETHERMOORE.

The Langues Metric:
    L(x,t) = Σ wₗ exp(βₗ · (dₗ + sin(ωₗt + φₗ)))

Where:
    - Six Sacred Tongues: KO, AV, RU, CA, UM, DR
    - Weights: wₗ = φˡ (golden ratio progression)
    - Phases: φₗ = 2πk/6 (60° intervals)

Fluxing Dimensions (Polly/Quasi/Demi):
    L_f(x,t) = Σ νᵢ(t) wᵢ exp[βᵢ(dᵢ + sin(ωᵢt + φᵢ))]
    ν̇ᵢ = κᵢ(ν̄ᵢ - νᵢ) + σᵢ sin(Ωᵢt)

Reference: SCBE Patent Specification, Axiom A7 (Six-Fold Symmetry)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict, NamedTuple
import numpy as np

# Golden ratio - fundamental constant
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618033988749895


class SacredTongue(Enum):
    """The Six Sacred Tongues of governance."""
    KO = 0  # Knowledge/Origin
    AV = 1  # Avatar/Voice
    RU = 2  # Rule/Reason
    CA = 3  # Causation/Action
    UM = 4  # Unity/Manifest
    DR = 5  # Dream/Reality


class FluxState(Enum):
    """Dimension flux states."""
    POLLY = auto()     # ν ≈ 1.0 - Full dimension active
    QUASI = auto()     # 0.5 < ν < 1.0 - Partial participation
    DEMI = auto()      # 0.0 < ν < 0.5 - Minimal participation
    COLLAPSED = auto() # ν ≈ 0.0 - Dimension off


class GovernanceDecision(Enum):
    """Risk-based governance decisions."""
    ALLOW = auto()
    QUARANTINE = auto()
    DENY = auto()


@dataclass
class TongueParameters:
    """Parameters for a single Sacred Tongue dimension."""
    tongue: SacredTongue
    weight: float      # wₗ = φˡ
    beta: float        # βₗ - exponential scaling
    omega: float       # ωₗ - phase frequency
    phase: float       # φₗ - phase offset (radians)

    @classmethod
    def create_default(cls, tongue: SacredTongue, beta_base: float = 1.0) -> TongueParameters:
        """Create default parameters for a tongue using golden ratio progression."""
        k = tongue.value
        return cls(
            tongue=tongue,
            weight=PHI ** k,
            beta=beta_base * (1 + 0.1 * k),  # Slight scaling per dimension
            omega=1.0 + 0.1 * k,              # Frequency varies slightly
            phase=2 * math.pi * k / 6         # 60° intervals
        )


@dataclass
class DimensionFlux:
    """
    Fluxing dimension state with dynamics.

    ν̇ᵢ = κᵢ(ν̄ᵢ - νᵢ) + σᵢ sin(Ωᵢt)
    """
    nu: float = 1.0           # Current flux value ∈ [0, 1]
    nu_bar: float = 1.0       # Target/mean flux
    kappa: float = 0.1        # Relaxation rate
    sigma: float = 0.05       # Oscillation amplitude
    omega_flux: float = 0.5   # Oscillation frequency

    def get_state(self) -> FluxState:
        """Determine flux state from nu value."""
        if self.nu >= 0.9:
            return FluxState.POLLY
        elif self.nu >= 0.5:
            return FluxState.QUASI
        elif self.nu > 0.1:
            return FluxState.DEMI
        else:
            return FluxState.COLLAPSED

    def update(self, dt: float, t: float) -> float:
        """
        Update flux value using dynamics equation.

        ν̇ᵢ = κᵢ(ν̄ᵢ - νᵢ) + σᵢ sin(Ωᵢt)
        """
        nu_dot = self.kappa * (self.nu_bar - self.nu) + self.sigma * math.sin(self.omega_flux * t)
        self.nu = max(0.0, min(1.0, self.nu + nu_dot * dt))
        return self.nu


class LanguesMetricResult(NamedTuple):
    """Result from Langues metric computation."""
    total: float                          # L(x,t) total cost
    per_tongue: Dict[SacredTongue, float] # Individual tongue contributions
    time: float                           # Time parameter used
    point_norm: float                     # ‖x‖ of input point


@dataclass
class LanguesMetric:
    """
    Six Sacred Tongues metric for governance cost computation.

    L(x,t) = Σ wₗ exp(βₗ · (dₗ + sin(ωₗt + φₗ)))

    Properties proven:
    1. Monotonicity: ∂L/∂dₗ > 0
    2. Phase bounded: sin ∈ [-1,1]
    3. Golden weights: wₗ = φˡ
    4. Six-fold symmetry: 60° phases
    """

    beta_base: float = 1.0
    tongues: Dict[SacredTongue, TongueParameters] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize tongue parameters if not provided."""
        if not self.tongues:
            for tongue in SacredTongue:
                self.tongues[tongue] = TongueParameters.create_default(tongue, self.beta_base)

    def compute(self, point: np.ndarray, t: float = 0.0) -> LanguesMetricResult:
        """
        Compute Langues metric at point and time.

        Args:
            point: 6D point (or will be projected/padded to 6D)
            t: Time parameter for phase modulation

        Returns:
            LanguesMetricResult with total cost and per-tongue breakdown
        """
        # Ensure 6D
        if len(point) < 6:
            point = np.pad(point, (0, 6 - len(point)), constant_values=0)
        elif len(point) > 6:
            point = point[:6]

        point_norm = np.linalg.norm(point)
        per_tongue: Dict[SacredTongue, float] = {}
        total = 0.0

        for tongue in SacredTongue:
            params = self.tongues[tongue]
            k = tongue.value

            # Distance in this dimension (scaled by position)
            d_l = abs(point[k]) if k < len(point) else 0.0

            # Phase-modulated exponential
            phase_term = math.sin(params.omega * t + params.phase)
            exponent = params.beta * (d_l + phase_term)

            # Clamp exponent to avoid overflow
            exponent = min(exponent, 50.0)

            contribution = params.weight * math.exp(exponent)
            per_tongue[tongue] = contribution
            total += contribution

        return LanguesMetricResult(
            total=total,
            per_tongue=per_tongue,
            time=t,
            point_norm=point_norm
        )

    def risk_level(self, metric_value: float) -> Tuple[float, GovernanceDecision]:
        """
        Convert metric value to risk level and decision.

        Risk thresholds based on golden ratio scaling:
        - ALLOW: L < φ³ ≈ 4.236
        - QUARANTINE: φ³ ≤ L < φ⁵ ≈ 11.09
        - DENY: L ≥ φ⁵
        """
        threshold_allow = PHI ** 3      # ≈ 4.236
        threshold_deny = PHI ** 5       # ≈ 11.09

        # Normalize to [0, 1] risk scale
        if metric_value < threshold_allow:
            risk = metric_value / threshold_allow * 0.3
            decision = GovernanceDecision.ALLOW
        elif metric_value < threshold_deny:
            risk = 0.3 + (metric_value - threshold_allow) / (threshold_deny - threshold_allow) * 0.4
            decision = GovernanceDecision.QUARANTINE
        else:
            risk = min(1.0, 0.7 + (metric_value - threshold_deny) / threshold_deny * 0.3)
            decision = GovernanceDecision.DENY

        return risk, decision

    def gradient(self, point: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Compute gradient ∇L at point.

        ∂L/∂xₖ = wₖ · βₖ · sign(xₖ) · exp(βₖ · (|xₖ| + sin(ωₖt + φₖ)))

        Proof of monotonicity: ∂L/∂dₗ = wₗ · βₗ · exp(...) > 0 always.
        """
        if len(point) < 6:
            point = np.pad(point, (0, 6 - len(point)), constant_values=0)
        elif len(point) > 6:
            point = point[:6]

        grad = np.zeros(6)

        for tongue in SacredTongue:
            params = self.tongues[tongue]
            k = tongue.value

            d_l = abs(point[k])
            phase_term = math.sin(params.omega * t + params.phase)
            exponent = min(params.beta * (d_l + phase_term), 50.0)

            # Gradient component
            sign_x = np.sign(point[k]) if point[k] != 0 else 0
            grad[k] = params.weight * params.beta * sign_x * math.exp(exponent)

        return grad

    def verify_monotonicity(self, point: np.ndarray, t: float = 0.0) -> bool:
        """
        Verify monotonicity property: ∂L/∂dₗ > 0 for all dimensions.

        This is always true since wₗ > 0, βₗ > 0, exp(...) > 0.
        """
        grad = self.gradient(point, t)
        # Check that gradient has same sign as point coordinates
        for k in range(6):
            if point[k] != 0:
                expected_sign = np.sign(point[k])
                actual_sign = np.sign(grad[k])
                if expected_sign != actual_sign:
                    return False
        return True


@dataclass
class FluxingLanguesMetric:
    """
    Langues metric with fluxing dimensions (Polly/Quasi/Demi).

    L_f(x,t) = Σ νᵢ(t) wᵢ exp[βᵢ(dᵢ + sin(ωᵢt + φᵢ))]

    Properties proven:
    5. Flux bounded: ν ∈ [0,1]
    6. Dimension conservation: mean D_f ≈ Σν̄ᵢ
    """

    base_metric: LanguesMetric = field(default_factory=LanguesMetric)
    fluxes: Dict[SacredTongue, DimensionFlux] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize fluxes if not provided."""
        if not self.fluxes:
            for tongue in SacredTongue:
                self.fluxes[tongue] = DimensionFlux()

    def compute(self, point: np.ndarray, t: float = 0.0) -> LanguesMetricResult:
        """
        Compute fluxing Langues metric.

        L_f(x,t) = Σ νᵢ(t) wᵢ exp[βᵢ(dᵢ + sin(ωᵢt + φᵢ))]
        """
        if len(point) < 6:
            point = np.pad(point, (0, 6 - len(point)), constant_values=0)
        elif len(point) > 6:
            point = point[:6]

        point_norm = np.linalg.norm(point)
        per_tongue: Dict[SacredTongue, float] = {}
        total = 0.0

        for tongue in SacredTongue:
            params = self.base_metric.tongues[tongue]
            flux = self.fluxes[tongue]
            k = tongue.value

            d_l = abs(point[k]) if k < len(point) else 0.0
            phase_term = math.sin(params.omega * t + params.phase)
            exponent = min(params.beta * (d_l + phase_term), 50.0)

            # Apply flux weighting
            contribution = flux.nu * params.weight * math.exp(exponent)
            per_tongue[tongue] = contribution
            total += contribution

        return LanguesMetricResult(
            total=total,
            per_tongue=per_tongue,
            time=t,
            point_norm=point_norm
        )

    def update_fluxes(self, dt: float, t: float) -> Dict[SacredTongue, FluxState]:
        """Update all flux values and return their states."""
        states = {}
        for tongue, flux in self.fluxes.items():
            flux.update(dt, t)
            states[tongue] = flux.get_state()
        return states

    def get_effective_dimension(self) -> float:
        """
        Get effective dimensionality (sum of flux values).

        D_f = Σνᵢ ∈ [0, 6]

        Conservation: E[D_f] ≈ Σν̄ᵢ over time.
        """
        return sum(flux.nu for flux in self.fluxes.values())

    def set_flux_targets(self, targets: Dict[SacredTongue, float]):
        """Set target flux values (ν̄ᵢ) for specified tongues."""
        for tongue, target in targets.items():
            if tongue in self.fluxes:
                self.fluxes[tongue].nu_bar = max(0.0, min(1.0, target))

    def collapse_dimension(self, tongue: SacredTongue):
        """Collapse a dimension to near-zero flux."""
        if tongue in self.fluxes:
            self.fluxes[tongue].nu_bar = 0.0
            self.fluxes[tongue].nu = 0.0

    def activate_dimension(self, tongue: SacredTongue):
        """Fully activate a dimension."""
        if tongue in self.fluxes:
            self.fluxes[tongue].nu_bar = 1.0
            self.fluxes[tongue].nu = 1.0


def project_to_1d(point: np.ndarray, direction: Optional[np.ndarray] = None) -> float:
    """
    Project 6D point to 1D for visualization.

    Default direction uses golden-weighted sum.

    Proof: 1D projection correctness - preserves relative ordering
    along the projection direction.
    """
    if len(point) < 6:
        point = np.pad(point, (0, 6 - len(point)), constant_values=0)
    elif len(point) > 6:
        point = point[:6]

    if direction is None:
        # Golden-weighted direction
        direction = np.array([PHI ** k for k in range(6)])
        direction = direction / np.linalg.norm(direction)

    return float(np.dot(point, direction))


def compute_six_fold_symmetry_error(metric: LanguesMetric) -> float:
    """
    Verify six-fold symmetry of the metric configuration.

    Checks that phases are distributed at 60° intervals.
    Returns max deviation from ideal 60° spacing.
    """
    phases = [metric.tongues[t].phase for t in SacredTongue]
    ideal_spacing = 2 * math.pi / 6

    max_error = 0.0
    for i in range(6):
        ideal_phase = ideal_spacing * i
        error = abs(phases[i] - ideal_phase)
        # Account for wraparound
        error = min(error, 2 * math.pi - error)
        max_error = max(max_error, error)

    return max_error


def verify_golden_weights(metric: LanguesMetric, tolerance: float = 1e-10) -> bool:
    """
    Verify golden ratio weight progression: wₗ = φˡ.
    """
    for tongue in SacredTongue:
        expected = PHI ** tongue.value
        actual = metric.tongues[tongue].weight
        if abs(expected - actual) > tolerance:
            return False
    return True


# Convenience exports
__all__ = [
    'PHI',
    'SacredTongue',
    'FluxState',
    'GovernanceDecision',
    'TongueParameters',
    'DimensionFlux',
    'LanguesMetricResult',
    'LanguesMetric',
    'FluxingLanguesMetric',
    'project_to_1d',
    'compute_six_fold_symmetry_error',
    'verify_golden_weights',
]
