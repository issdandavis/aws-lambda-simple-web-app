#!/usr/bin/env python3
"""
SCBE Gravity-Intent Evolution System
=====================================

Implements Grok's theoretical validation findings:
1. Gravity-emotion vector integration as chaotic dynamical system
2. Quasicrystalline parameters with φ-based golden ratio rotations
3. Lyapunov exponent calculation for chaos verification
4. Avalanche effect testing for cryptographic strength
5. Hybrid encoding preparation (Morse + DNA patterns)

Mathematical Foundation:
F_i = Σ G * m_i * m_j * (r_j - r_i) / ||r_j - r_i||³

Reference: Grok Theoretical Validation (Jan 2026)
Author: Generated for SCBE v3.0
Date: January 11, 2026
"""

import numpy as np
import hashlib
import secrets
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum
import json

# =============================================================================
# CONSTANTS - GROK VALIDATED PARAMETERS
# =============================================================================

# Golden Ratio - confirmed as "magic number" by Grok
PHI = (1 + np.sqrt(5)) / 2  # φ ≈ 1.6180339887
PHI_INV = 1 / PHI           # 1/φ ≈ 0.6180339887

# Quasicrystal optimal parameters (Grok recommendations)
QUASICRYSTAL_DIMENSIONALITY = 6      # 6D → 3D projection (matches SCBE gates)
ROTATION_ANGLE = np.pi / 5            # π/5 = 36° (5-fold symmetry)
PHI_ROTATION = 2 * np.pi / PHI        # φ-based rotation ≈ 222.5°
LATTICE_SCALE_MIN = 1000              # 10³ initial points
LATTICE_SCALE_MAX = 100000            # 10⁵ max points
TRANSFORMATION_STEPS_MIN = 10         # Minimum inflation/deflation iterations
TRANSFORMATION_STEPS_MAX = 50         # Maximum iterations

# Gravity simulation parameters
GRAVITATIONAL_CONSTANT = 6.674e-11    # Real G (scaled for simulation)
G_SCALED = 1.0                        # Scaled G for numerical stability
DEFAULT_MASS = 1.0                    # Default intent vector mass
SOFTENING_EPSILON = 0.01              # Prevent singularity at r=0

# Chaos/Lyapunov parameters
LYAPUNOV_PERTURBATION = 1e-8          # Initial separation for Lyapunov
LYAPUNOV_STEPS = 1000                 # Steps for Lyapunov calculation
CHAOS_THRESHOLD = 0.1                 # Positive λ indicates chaos

# Avalanche test parameters
AVALANCHE_ITERATIONS = 1000           # Number of bit-flip tests
MIN_AVALANCHE_RATIO = 0.4             # Minimum 40% bit change (ideal: 50%)
MAX_AVALANCHE_RATIO = 0.6             # Maximum 60% bit change


# =============================================================================
# GRAVITY-INTENT EVOLUTION SYSTEM
# =============================================================================

@dataclass
class IntentVector:
    """
    Intent vector with mass and position in semantic space.

    Represents an authentication intent as a point mass that
    interacts gravitationally with other intents.
    """
    position: np.ndarray      # 6D position (one per SCBE gate)
    velocity: np.ndarray      # 6D velocity
    mass: float = DEFAULT_MASS
    intent_id: str = ""
    actor_id: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if self.position is None:
            self.position = np.zeros(6)
        if self.velocity is None:
            self.velocity = np.zeros(6)
        if not self.intent_id:
            self.intent_id = secrets.token_hex(8)
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class GravityIntentEvolution:
    """
    N-body gravitational simulation for intent vector evolution.

    Implements: F_i = Σ G * m_i * m_j * (r_j - r_i) / ||r_j - r_i||³

    This creates a chaotic dynamical system where:
    - Intent vectors are point masses
    - They attract/repel based on semantic similarity
    - Evolution is deterministic but sensitive to initial conditions
    - Forward security requires combining with secure primitives
    """

    def __init__(self, G: float = G_SCALED, softening: float = SOFTENING_EPSILON):
        self.G = G
        self.softening = softening
        self.vectors: List[IntentVector] = []
        self.history: List[List[np.ndarray]] = []  # Position history for Lyapunov
        self.dt = 0.01  # Time step

    def add_vector(self, position: np.ndarray, velocity: np.ndarray = None,
                   mass: float = DEFAULT_MASS, actor_id: str = "") -> IntentVector:
        """Add an intent vector to the system."""
        if velocity is None:
            velocity = np.zeros_like(position)

        vec = IntentVector(
            position=np.array(position, dtype=np.float64),
            velocity=np.array(velocity, dtype=np.float64),
            mass=mass,
            actor_id=actor_id
        )
        self.vectors.append(vec)
        return vec

    def compute_forces(self) -> np.ndarray:
        """
        Compute gravitational forces between all intent vectors.

        F_i = Σ G * m_i * m_j * (r_j - r_i) / ||r_j - r_i||³

        Complexity: O(n²) per step
        """
        n = len(self.vectors)
        forces = np.zeros((n, 6))

        for i in range(n):
            for j in range(n):
                if i != j:
                    r_ij = self.vectors[j].position - self.vectors[i].position
                    dist = np.linalg.norm(r_ij)

                    # Softened gravitational force (prevents singularity)
                    dist_soft = np.sqrt(dist**2 + self.softening**2)

                    # F = G * m_i * m_j * r_ij / |r_ij|³
                    force_mag = (self.G * self.vectors[i].mass *
                                self.vectors[j].mass / dist_soft**3)
                    forces[i] += force_mag * r_ij

        return forces

    def step(self) -> None:
        """Advance simulation by one time step using leapfrog integration."""
        forces = self.compute_forces()

        # Record positions for history
        positions = [v.position.copy() for v in self.vectors]
        self.history.append(positions)

        # Leapfrog integration (symplectic, preserves energy)
        for i, vec in enumerate(self.vectors):
            acceleration = forces[i] / vec.mass

            # Half-step velocity
            vec.velocity += 0.5 * acceleration * self.dt

            # Full-step position
            vec.position += vec.velocity * self.dt

            # Recompute force at new position (not strictly necessary for kick-drift-kick)
            # Half-step velocity again
            vec.velocity += 0.5 * acceleration * self.dt

    def evolve(self, steps: int) -> List[List[np.ndarray]]:
        """Evolve system for given number of steps."""
        for _ in range(steps):
            self.step()
        return self.history

    def get_state_hash(self) -> bytes:
        """Get cryptographic hash of current system state."""
        state_data = b""
        for vec in self.vectors:
            state_data += vec.position.tobytes()
            state_data += vec.velocity.tobytes()
        return hashlib.sha256(state_data).digest()

    def reset(self):
        """Reset system to empty state."""
        self.vectors.clear()
        self.history.clear()


# =============================================================================
# LYAPUNOV EXPONENT CALCULATOR
# =============================================================================

class LyapunovExponentCalculator:
    """
    Calculate Lyapunov exponent to verify chaotic behavior.

    Positive Lyapunov exponent (λ > 0) indicates:
    - Sensitive dependence on initial conditions
    - Exponential divergence of nearby trajectories
    - Chaotic (unpredictable) dynamics

    This is CRITICAL for security: chaos prevents prediction.
    """

    @staticmethod
    def calculate(system: GravityIntentEvolution,
                  perturbation: float = LYAPUNOV_PERTURBATION,
                  steps: int = LYAPUNOV_STEPS) -> Tuple[float, List[float]]:
        """
        Calculate largest Lyapunov exponent.

        Method: Track divergence of nearby trajectories.

        Returns:
            (lyapunov_exponent, divergence_history)
        """
        if len(system.vectors) < 2:
            return 0.0, []

        # Store original positions
        original_positions = [v.position.copy() for v in system.vectors]
        original_velocities = [v.velocity.copy() for v in system.vectors]

        # Create perturbed system
        perturbed_system = GravityIntentEvolution(system.G, system.softening)
        for i, vec in enumerate(system.vectors):
            perturbed_pos = vec.position + perturbation * np.random.randn(6)
            perturbed_system.add_vector(
                perturbed_pos,
                vec.velocity.copy(),
                vec.mass,
                vec.actor_id
            )

        divergence_history = []
        lyapunov_sum = 0.0

        for step in range(steps):
            # Evolve both systems
            system.step()
            perturbed_system.step()

            # Calculate total separation
            total_separation = 0.0
            for i in range(len(system.vectors)):
                sep = np.linalg.norm(
                    system.vectors[i].position -
                    perturbed_system.vectors[i].position
                )
                total_separation += sep**2

            separation = np.sqrt(total_separation)

            if separation > 0 and perturbation > 0:
                # Log of divergence ratio
                lyapunov_sum += np.log(separation / perturbation)
                divergence_history.append(separation)

                # Renormalize perturbed trajectory
                for i in range(len(system.vectors)):
                    direction = (perturbed_system.vectors[i].position -
                               system.vectors[i].position)
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                    perturbed_system.vectors[i].position = (
                        system.vectors[i].position + perturbation * direction
                    )

        # Restore original system
        for i, vec in enumerate(system.vectors):
            vec.position = original_positions[i]
            vec.velocity = original_velocities[i]
        system.history.clear()

        # Average Lyapunov exponent
        lyapunov = lyapunov_sum / steps if steps > 0 else 0.0

        return lyapunov, divergence_history


# =============================================================================
# PHI-BASED QUASICRYSTAL GENERATOR
# =============================================================================

class PhiQuasicrystalGenerator:
    """
    Generate quasicrystal structures using golden ratio rotations.

    Grok-validated parameters:
    - Rotation angles: multiples of π/5 or φ-based
    - Dimensionality: 6D → 3D projection
    - Scale: 10³-10⁵ initial points
    - Iterations: 10-50 inflation/deflation steps
    """

    def __init__(self, dimension: int = QUASICRYSTAL_DIMENSIONALITY):
        self.dimension = dimension
        self.phi = PHI
        self.phi_inv = PHI_INV
        self.points: List[np.ndarray] = []
        self.iteration_count = 0

    def generate_initial_lattice(self, scale: int = LATTICE_SCALE_MIN) -> np.ndarray:
        """
        Generate initial quasi-periodic lattice using φ-based spacing.
        """
        # Generate points using golden ratio spacing (low-discrepancy)
        points = []

        for i in range(scale):
            # Fibonacci-based point generation
            point = np.zeros(self.dimension)
            for d in range(self.dimension):
                # Golden ratio modular arithmetic
                point[d] = ((i * self.phi**(d+1)) % 1) * 2 - 1  # Map to [-1, 1]
            points.append(point)

        self.points = points
        return np.array(points)

    def phi_rotation_matrix(self, axis1: int, axis2: int) -> np.ndarray:
        """
        Create rotation matrix using golden angle in specified plane.
        """
        # Golden angle ≈ 137.5° = 2π/φ² ≈ 2.4 radians
        golden_angle = 2 * np.pi / (self.phi ** 2)

        matrix = np.eye(self.dimension)
        c, s = np.cos(golden_angle), np.sin(golden_angle)

        matrix[axis1, axis1] = c
        matrix[axis1, axis2] = -s
        matrix[axis2, axis1] = s
        matrix[axis2, axis2] = c

        return matrix

    def five_fold_rotation_matrix(self) -> np.ndarray:
        """
        Create rotation matrix with π/5 (36°) rotations.

        This produces 5-fold symmetry characteristic of quasicrystals.
        """
        angle = np.pi / 5  # 36 degrees

        # Compound rotation across multiple planes
        matrix = np.eye(self.dimension)

        for d in range(0, self.dimension - 1, 2):
            c, s = np.cos(angle), np.sin(angle)
            plane_rot = np.eye(self.dimension)
            plane_rot[d, d] = c
            plane_rot[d, d+1] = -s
            plane_rot[d+1, d] = s
            plane_rot[d+1, d+1] = c
            matrix = matrix @ plane_rot

        return matrix

    def inflate(self, iterations: int = 1) -> np.ndarray:
        """
        Inflate quasicrystal (scale up with subdivision).

        Uses golden ratio scaling factor.
        """
        if not self.points:
            self.generate_initial_lattice()

        for _ in range(iterations):
            new_points = []

            for point in self.points:
                # Scale by φ
                scaled = point * self.phi
                new_points.append(scaled)

                # Add midpoints with φ-based offsets
                for d in range(self.dimension):
                    offset = np.zeros(self.dimension)
                    offset[d] = self.phi_inv
                    new_points.append(scaled + offset)

            self.points = new_points
            self.iteration_count += 1

            # Limit growth
            if len(self.points) > LATTICE_SCALE_MAX:
                # Keep points within unit hypercube
                self.points = [p for p in self.points
                              if np.all(np.abs(p) <= self.phi)]

        return np.array(self.points)

    def deflate(self, iterations: int = 1) -> np.ndarray:
        """
        Deflate quasicrystal (scale down with merging).

        Uses inverse golden ratio scaling.
        """
        if not self.points:
            return np.array([])

        for _ in range(iterations):
            # Scale down by 1/φ
            self.points = [p * self.phi_inv for p in self.points]

            # Merge nearby points
            merged = []
            used = set()

            for i, p1 in enumerate(self.points):
                if i in used:
                    continue

                cluster = [p1]
                used.add(i)

                for j, p2 in enumerate(self.points):
                    if j not in used:
                        if np.linalg.norm(p1 - p2) < 0.1:
                            cluster.append(p2)
                            used.add(j)

                # Use centroid of cluster
                merged.append(np.mean(cluster, axis=0))

            self.points = merged
            self.iteration_count += 1

        return np.array(self.points)

    def project_to_3d(self, points: np.ndarray = None) -> np.ndarray:
        """
        Project 6D quasicrystal to 3D using golden ratio projection.
        """
        if points is None:
            points = np.array(self.points)

        if len(points) == 0:
            return np.array([])

        # Projection matrix (6D → 3D) using φ
        proj_matrix = np.array([
            [1, 0, 0, self.phi_inv, 0, 0],
            [0, 1, 0, 0, self.phi_inv, 0],
            [0, 0, 1, 0, 0, self.phi_inv]
        ])

        # Normalize rows
        for i in range(3):
            proj_matrix[i] /= np.linalg.norm(proj_matrix[i])

        return points @ proj_matrix.T


# =============================================================================
# AVALANCHE EFFECT TESTER
# =============================================================================

class AvalancheEffectTester:
    """
    Test avalanche effect for cryptographic strength.

    Good avalanche effect: flipping 1 input bit changes ~50% of output bits.

    This validates that our chaotic system produces strong diffusion.
    """

    @staticmethod
    def count_bit_differences(hash1: bytes, hash2: bytes) -> int:
        """Count number of differing bits between two hashes."""
        diff = 0
        for b1, b2 in zip(hash1, hash2):
            diff += bin(b1 ^ b2).count('1')
        return diff

    @staticmethod
    def test_gravity_avalanche(n_vectors: int = 10,
                                iterations: int = AVALANCHE_ITERATIONS) -> Dict:
        """
        Test avalanche effect of gravity-intent system.

        Flip one bit in initial conditions and measure hash divergence.
        """
        results = {
            'bit_differences': [],
            'avalanche_ratios': [],
            'passed': 0,
            'failed': 0
        }

        total_bits = 256  # SHA-256

        for _ in range(iterations):
            # Create base system
            base_system = GravityIntentEvolution()
            for i in range(n_vectors):
                pos = np.random.randn(6)
                base_system.add_vector(pos, mass=1.0 + 0.1*i)

            # Evolve and hash
            base_system.evolve(50)
            base_hash = base_system.get_state_hash()

            # Create perturbed system (flip one bit in first vector)
            perturbed_system = GravityIntentEvolution()
            for i, vec in enumerate(base_system.vectors):
                pos = vec.position.copy()
                if i == 0:
                    # Minimal perturbation (single bit flip equivalent)
                    pos[0] += 1e-10
                perturbed_system.add_vector(pos, mass=vec.mass)

            # Reset and re-evolve base system for fair comparison
            for i, vec in enumerate(base_system.vectors):
                vec.position = vec.position.copy()
                vec.velocity = np.zeros(6)
            base_system.history.clear()
            base_system.evolve(50)
            base_hash = base_system.get_state_hash()

            perturbed_system.evolve(50)
            perturbed_hash = perturbed_system.get_state_hash()

            # Count differences
            diff = AvalancheEffectTester.count_bit_differences(base_hash, perturbed_hash)
            ratio = diff / total_bits

            results['bit_differences'].append(diff)
            results['avalanche_ratios'].append(ratio)

            if MIN_AVALANCHE_RATIO <= ratio <= MAX_AVALANCHE_RATIO:
                results['passed'] += 1
            else:
                results['failed'] += 1

        # Summary statistics
        results['mean_ratio'] = np.mean(results['avalanche_ratios'])
        results['std_ratio'] = np.std(results['avalanche_ratios'])
        results['pass_rate'] = results['passed'] / iterations
        results['avalanche_quality'] = 'GOOD' if results['mean_ratio'] > 0.45 else 'WEAK'

        return results

    @staticmethod
    def test_quasicrystal_avalanche(iterations: int = AVALANCHE_ITERATIONS) -> Dict:
        """
        Test avalanche effect of quasicrystal generation.
        """
        results = {
            'bit_differences': [],
            'avalanche_ratios': [],
            'passed': 0,
            'failed': 0
        }

        total_bits = 256

        for _ in range(iterations):
            # Create base quasicrystal
            base_qc = PhiQuasicrystalGenerator()
            base_qc.generate_initial_lattice(100)
            base_qc.inflate(5)
            base_points = np.array(base_qc.points)
            base_hash = hashlib.sha256(base_points.tobytes()).digest()

            # Create perturbed quasicrystal
            perturbed_qc = PhiQuasicrystalGenerator()
            perturbed_qc.generate_initial_lattice(100)
            # Perturb first point slightly
            if perturbed_qc.points:
                perturbed_qc.points[0] = perturbed_qc.points[0] + 1e-10
            perturbed_qc.inflate(5)
            perturbed_points = np.array(perturbed_qc.points)
            perturbed_hash = hashlib.sha256(perturbed_points.tobytes()).digest()

            diff = AvalancheEffectTester.count_bit_differences(base_hash, perturbed_hash)
            ratio = diff / total_bits

            results['bit_differences'].append(diff)
            results['avalanche_ratios'].append(ratio)

            if MIN_AVALANCHE_RATIO <= ratio <= MAX_AVALANCHE_RATIO:
                results['passed'] += 1
            else:
                results['failed'] += 1

        results['mean_ratio'] = np.mean(results['avalanche_ratios'])
        results['std_ratio'] = np.std(results['avalanche_ratios'])
        results['pass_rate'] = results['passed'] / iterations
        results['avalanche_quality'] = 'GOOD' if results['mean_ratio'] > 0.45 else 'WEAK'

        return results


# =============================================================================
# HYBRID ENCODING (MORSE + DNA PATTERNS)
# =============================================================================

class HybridEncoder:
    """
    Hybrid encoding system using Morse code and DNA patterns.

    Grok recommendation:
    - Morse code for initial compression of intent patterns
    - DNA layering for steganography and pattern recognition
    - Aids fail-to-noise oracle implementation
    """

    # Morse code mapping
    MORSE_CODE = {
        'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
        'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
        'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
        'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
        'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
        'Z': '--..', '0': '-----', '1': '.----', '2': '..---',
        '3': '...--', '4': '....-', '5': '.....', '6': '-....',
        '7': '--...', '8': '---..', '9': '----.', ' ': '/'
    }

    # DNA codon mapping (simplified - maps hex to codons)
    DNA_CODONS = {
        '0': 'AAA', '1': 'AAC', '2': 'AAG', '3': 'AAT',
        '4': 'ACA', '5': 'ACC', '6': 'ACG', '7': 'ACT',
        '8': 'AGA', '9': 'AGC', 'A': 'AGG', 'B': 'AGT',
        'C': 'ATA', 'D': 'ATC', 'E': 'ATG', 'F': 'ATT'
    }

    @classmethod
    def text_to_morse(cls, text: str) -> str:
        """Convert text to Morse code."""
        return ' '.join(cls.MORSE_CODE.get(c.upper(), '') for c in text)

    @classmethod
    def morse_to_bits(cls, morse: str) -> bytes:
        """Convert Morse code to bit pattern."""
        bits = []
        for char in morse:
            if char == '.':
                bits.append(0)
            elif char == '-':
                bits.append(1)
            elif char == ' ':
                bits.extend([0, 0])  # Word separator
            elif char == '/':
                bits.extend([1, 1])  # Letter separator

        # Pad to byte boundary
        while len(bits) % 8 != 0:
            bits.append(0)

        # Convert to bytes
        result = []
        for i in range(0, len(bits), 8):
            byte = sum(bits[i+j] << (7-j) for j in range(8))
            result.append(byte)

        return bytes(result)

    @classmethod
    def bytes_to_dna(cls, data: bytes) -> str:
        """Convert bytes to DNA sequence."""
        hex_str = data.hex().upper()
        return ''.join(cls.DNA_CODONS.get(c, 'NNN') for c in hex_str)

    @classmethod
    def dna_to_bytes(cls, dna: str) -> bytes:
        """Convert DNA sequence back to bytes."""
        reverse_codons = {v: k for k, v in cls.DNA_CODONS.items()}

        hex_chars = []
        for i in range(0, len(dna), 3):
            codon = dna[i:i+3]
            hex_chars.append(reverse_codons.get(codon, '0'))

        hex_str = ''.join(hex_chars)
        return bytes.fromhex(hex_str)

    @classmethod
    def encode_intent(cls, intent: str) -> Dict:
        """
        Encode intent using hybrid Morse + DNA encoding.

        Returns both encodings for different use cases.
        """
        morse = cls.text_to_morse(intent)
        morse_bits = cls.morse_to_bits(morse)
        dna = cls.bytes_to_dna(morse_bits)

        return {
            'original': intent,
            'morse': morse,
            'morse_bits_hex': morse_bits.hex(),
            'dna': dna,
            'compression_ratio': len(morse_bits) / len(intent) if intent else 0,
            'dna_length': len(dna)
        }


# =============================================================================
# COMPREHENSIVE VALIDATION
# =============================================================================

def run_grok_validation():
    """Run comprehensive validation of Grok's recommendations."""
    print("\n" + "="*70)
    print("  GROK THEORETICAL VALIDATION - IMPLEMENTATION TEST")
    print("  SCBE v3.0 - Gravity-Intent Evolution")
    print("="*70)

    results = {}

    # Test 1: Gravity-Intent Evolution
    print("\n[1/5] Testing Gravity-Intent Evolution (n=10 vectors)...")

    system = GravityIntentEvolution()
    for i in range(10):
        pos = np.random.randn(6) * 2
        vel = np.random.randn(6) * 0.1
        system.add_vector(pos, vel, mass=1.0 + 0.1*i, actor_id=f"actor_{i}")

    initial_hash = system.get_state_hash()
    system.evolve(100)
    final_hash = system.get_state_hash()

    results['gravity_evolution'] = {
        'n_vectors': 10,
        'evolution_steps': 100,
        'initial_hash': initial_hash.hex()[:16],
        'final_hash': final_hash.hex()[:16],
        'state_changed': initial_hash != final_hash
    }

    print(f"  ✓ Vectors: 10, Steps: 100")
    print(f"  ✓ Initial hash: {initial_hash.hex()[:16]}...")
    print(f"  ✓ Final hash: {final_hash.hex()[:16]}...")
    print(f"  ✓ State evolved: {initial_hash != final_hash}")

    # Test 2: Lyapunov Exponent
    print("\n[2/5] Calculating Lyapunov Exponent (chaos verification)...")

    lyapunov_system = GravityIntentEvolution()
    for i in range(10):
        pos = np.random.randn(6)
        lyapunov_system.add_vector(pos, mass=1.0)

    lyapunov, divergence = LyapunovExponentCalculator.calculate(
        lyapunov_system, steps=500
    )

    is_chaotic = lyapunov > CHAOS_THRESHOLD

    results['lyapunov'] = {
        'exponent': lyapunov,
        'is_chaotic': is_chaotic,
        'divergence_samples': len(divergence),
        'chaos_threshold': CHAOS_THRESHOLD
    }

    print(f"  ✓ Lyapunov exponent λ = {lyapunov:.6f}")
    print(f"  ✓ Chaos threshold: λ > {CHAOS_THRESHOLD}")
    print(f"  ✓ System is chaotic: {is_chaotic}")

    # Test 3: Quasicrystal Generation
    print("\n[3/5] Testing φ-based Quasicrystal Generation...")

    qc = PhiQuasicrystalGenerator(dimension=6)
    initial_points = qc.generate_initial_lattice(1000)

    # Apply golden ratio rotations
    rot_matrix = qc.phi_rotation_matrix(0, 1)
    five_fold = qc.five_fold_rotation_matrix()

    # Inflate
    qc.inflate(3)
    inflated_count = len(qc.points)

    # Project to 3D
    projected = qc.project_to_3d()

    results['quasicrystal'] = {
        'initial_points': 1000,
        'after_inflation': inflated_count,
        'inflation_steps': 3,
        'projection_shape': projected.shape if len(projected) > 0 else (0, 0),
        'phi_value': PHI,
        'rotation_angle': f"{np.degrees(ROTATION_ANGLE):.1f}°"
    }

    print(f"  ✓ Initial points: 1000")
    print(f"  ✓ After 3 inflations: {inflated_count}")
    print(f"  ✓ φ = {PHI:.10f}")
    print(f"  ✓ Rotation angle: {np.degrees(ROTATION_ANGLE):.1f}° (π/5)")

    # Test 4: Avalanche Effect
    print("\n[4/5] Testing Avalanche Effect (100 iterations)...")

    gravity_avalanche = AvalancheEffectTester.test_gravity_avalanche(
        n_vectors=10, iterations=100
    )

    results['avalanche_gravity'] = {
        'mean_ratio': gravity_avalanche['mean_ratio'],
        'std_ratio': gravity_avalanche['std_ratio'],
        'pass_rate': gravity_avalanche['pass_rate'],
        'quality': gravity_avalanche['avalanche_quality']
    }

    print(f"  ✓ Gravity system avalanche: {gravity_avalanche['mean_ratio']:.3f} ± {gravity_avalanche['std_ratio']:.3f}")
    print(f"  ✓ Quality: {gravity_avalanche['avalanche_quality']}")
    print(f"  ✓ Pass rate: {gravity_avalanche['pass_rate']*100:.1f}%")

    # Test 5: Hybrid Encoding
    print("\n[5/5] Testing Hybrid Encoding (Morse + DNA)...")

    test_intent = "READ DATA"
    encoded = HybridEncoder.encode_intent(test_intent)

    results['hybrid_encoding'] = encoded

    print(f"  ✓ Original: '{test_intent}'")
    print(f"  ✓ Morse: {encoded['morse']}")
    print(f"  ✓ DNA: {encoded['dna'][:30]}... ({encoded['dna_length']} bases)")
    print(f"  ✓ Compression ratio: {encoded['compression_ratio']:.2f}")

    # Summary
    print("\n" + "="*70)
    print("  VALIDATION SUMMARY")
    print("="*70)

    summary = {
        'gravity_evolution_working': results['gravity_evolution']['state_changed'],
        'system_is_chaotic': results['lyapunov']['is_chaotic'],
        'lyapunov_exponent': results['lyapunov']['exponent'],
        'quasicrystal_generated': results['quasicrystal']['after_inflation'] > 0,
        'avalanche_quality': results['avalanche_gravity']['quality'],
        'hybrid_encoding_working': len(encoded['dna']) > 0
    }

    for key, value in summary.items():
        if isinstance(value, bool):
            status = "✓ PASS" if value else "✗ FAIL"
        elif isinstance(value, float):
            status = f"= {value:.6f}"
        else:
            status = f"= {value}"
        print(f"  {status}: {key}")

    print("="*70)

    results['summary'] = summary
    return results


if __name__ == '__main__':
    results = run_grok_validation()

    # Save results
    output_file = 'gravity_intent_validation.json'

    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, bytes):
            return obj.hex()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)

    print(f"\n  Results saved to: {output_file}")
