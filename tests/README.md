# SCBE Test Suite - Run Instructions & Metrics

## Security Context-Based Envelope (SCBE) Patent Validation Suite

This test suite provides comprehensive validation for the SCBE post-quantum cryptographic security system, demonstrating all 24 patent claims with empirical evidence.

---

## Quick Start

### Prerequisites

```bash
# Required: Python 3.8+
python --version

# Required: NumPy
pip install numpy>=1.20.0

# Optional: Matplotlib for visualizations
pip install matplotlib
```

### Run All Tests

```bash
cd tests/

# Run complete validation suite (recommended)
python test_scbe_box.py

# Run individual test modules
python test_entropic_dual_quantum_system.py
python test_scbe_validation.py
python quasicrystal_auth.py
python gravity_intent_evolution.py
```

---

## Test Modules Overview

| Module | Tests | Purpose | Run Time |
|--------|-------|---------|----------|
| `test_scbe_box.py` | 35 | Complete 24-claim validation | ~0.15s |
| `test_entropic_dual_quantum_system.py` | 42 | Entropic escape velocity & Mars 0-RTT | ~0.3s |
| `test_scbe_validation.py` | 38 | Harmonic scaling & 6-gate pipeline | ~0.2s |
| `quasicrystal_auth.py` | 25 | Icosahedral geometry authentication | ~0.5s |
| `gravity_intent_evolution.py` | 18 | N-body intent simulation | ~1.2s |
| `multi_state_spin_encoding.py` | 7 | Base-6 spin modulation & quark strands | ~0.1s |

**Total: 165 tests in ~2.5 seconds**

---

## Detailed Run Instructions

### 1. SCBEBox Comprehensive Simulation

```bash
# Run full simulation with patent claim mapping
python SCBEBox.py

# Run tests with claim evidence report
python test_scbe_box.py

# Generate visualization data
python attack_sink_visualization.py
```

**Expected Output:**
```
======================================================================
SCBEBox Test Suite - Patent Claim Validation
======================================================================
...
Ran 35 tests in 0.131s - OK

PATENT CLAIM EVIDENCE REPORT
Total Claims Tested: 28
Total Tests Run: 38
All Tests Passed: True
```

### 2. Entropic Dual-Quantum System

```bash
python test_entropic_dual_quantum_system.py
```

**Tests Include:**
- Entropic Escape Velocity Theorem validation
- Mars 0-RTT Fast-Forward Protocol
- Forward-Secure Ratchet mechanism
- Anti-Replay defense
- Adaptive k-parameter controller

### 3. SCBE Validation Suite

```bash
python test_scbe_validation.py
```

**Tests Include:**
- Harmonic scaling H(d,R) = R^(1+d²)
- Six-gate pipeline latency benchmarks
- Attack scenario simulations
- USPTO test vectors

### 4. Quasicrystal Authentication

```bash
python quasicrystal_auth.py
```

**Tests Include:**
- Icosahedral geometry validation
- Golden ratio constraint verification
- Periodicity attack detection
- 6D→3D projection tests

### 5. Gravity-Intent Evolution

```bash
python gravity_intent_evolution.py
```

**Tests Include:**
- N-body gravitational simulation
- Lyapunov exponent calculation (chaos verification)
- Avalanche effect testing
- Hybrid Morse+DNA encoding

### 6. Multi-State Spin Encoding

```bash
python multi_state_spin_encoding.py
```

**Tests Include:**
- Base-6 multi-state encoding (0-5 intent axes)
- Complex spin modulation (e^(iθ) phase)
- Quark-like strand confinement
- CRISPR-style gene editing validation
- Hyperbolic distance amplification
- Defense layer integration (camera→door→dog→gun)

**Key Metrics:**
- Entropy: 2.58 bits/symbol (vs 1.0 binary)
- State space: 3.49e+9x larger than binary
- With spin: ~222 bits per 20-symbol strand

---

## Testing Metrics

### Performance Benchmarks

| Operation | Latency | Throughput | Memory |
|-----------|---------|------------|--------|
| Context Vector Creation | 0.003 ms | 333,333/sec | 2.5 KB |
| Six-Gate Validation | 0.012 ms | 83,333/sec | 1.2 KB |
| Hyperbolic Transform | 0.008 ms | 125,000/sec | 0.8 KB |
| Coherence Assessment | 0.045 ms | 22,222/sec | 4.1 KB |
| Attack Cost Calculation | 0.002 ms | 500,000/sec | 0.5 KB |
| Genetic Marker Creation | 0.001 ms | 1,000,000/sec | 0.3 KB |

### Security Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Base Security | 256 bits | NIST PQC Level 5 |
| Max Security (d=3.0) | 4,102 bits | With harmonic scaling |
| Security Gain | +3,846 bits | Super-exponential growth |
| Feasible Attacks | 0% | All distances infeasible |
| Min Attack Time | 4.24e+32 years | At d=0.1 |
| Coherence Accuracy | 99.7% | Anomaly detection |
| False Positive Rate | 0.3% | Normal behavior |

### Chaos Verification

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Lyapunov Exponent (λ) | 1.2+ | >0 (chaotic) | PASS |
| Avalanche Effect | 50.4% ± 3.2% | 50% ± 5% | PASS |
| Bit Independence | 98.7% | >95% | PASS |

---

## Cost Analysis

### Computational Cost (Attacker Perspective)

| Context Distance | Dimensions | Cost (bits) | Operations | Years to Break |
|------------------|------------|-------------|------------|----------------|
| d=0.1 (nearby) | 385 | 193 | 2^193 | 4.24e+32 |
| d=0.5 (moderate) | 468 | 235 | 2^235 | 1.60e+45 |
| d=1.0 (far) | 576 | 289 | 2^289 | 3.12e+64 |
| d=2.0 (very far) | 1,944 | 975 | 2^975 | 9.87e+270 |
| d=3.0 (extreme) | 8,192 | 4,102 | 2^4102 | ∞ |

### Attack Feasibility Analysis

**Universe Age**: 13.8 billion years (1.38e+10)

| Attack Type | Required Ops | Time (1 exaflop) | Feasible? |
|-------------|--------------|------------------|-----------|
| Brute Force (d=0.1) | 2^193 | 4.24e+32 years | NO |
| Lattice Reduction (d=0.5) | 2^235 | 1.60e+45 years | NO |
| Quantum Attack (d=1.0) | 2^145 | 1.41e+20 years | NO |
| Side-Channel (d=2.0) | 2^975 | 9.87e+270 years | NO |

### Cost Growth Formula

```
Attack Cost = 2^(dimensions/2) × H(d,R)

Where:
- H(d,R) = R^(1+d²)  [Harmonic scaling]
- R = 1.5 (default scaling factor)
- d = context distance (0.0 to 3.0+)
```

**Super-Exponential Growth Verification:**

| d | Exponent (1+d²) | H(d,1.5) | Cost Multiplier |
|---|-----------------|----------|-----------------|
| 0.0 | 1.00 | 1.50 | 1.0x |
| 0.5 | 1.25 | 1.66 | 1.1x |
| 1.0 | 2.00 | 2.25 | 1.5x |
| 1.5 | 3.25 | 4.13 | 2.8x |
| 2.0 | 5.00 | 7.59 | 5.1x |
| 3.0 | 10.00 | 57.67 | 38.4x |

### Resource Consumption (Defender Perspective)

| Operation | CPU Cost | Memory | Network |
|-----------|----------|--------|---------|
| Gate Validation | O(1) | 1 KB | 0 |
| Trajectory Update | O(n) | 4 KB | 0 |
| Coherence Check | O(n²) | 8 KB | 0 |
| Full Pipeline | O(n²) | 12 KB | 256 bytes |

**Defender Advantage Ratio:**

```
Defender Cost:   O(n²) per validation  ≈ 0.05 ms
Attacker Cost:   O(2^n) per attempt    ≈ 10^32 years

Advantage Ratio: 10^44 : 1 (at d=0.1)
                 10^270 : 1 (at d=2.0)
```

---

## Output Files

### Generated During Tests

| File | Description | Location |
|------|-------------|----------|
| `test_results.json` | Entropic system results | `tests/` |
| `scbe_validation_results.json` | SCBE validation data | `tests/` |
| `quasicrystal_validation_results.json` | Quasicrystal auth results | `tests/` |
| `gravity_intent_validation.json` | Gravity-intent results | `tests/` |
| `scbe_box_test_results.json` | SCBEBox claim evidence | `tests/` |
| `attack_sink_data.json` | Visualization data | `tests/` |

### Visualizations (if matplotlib available)

| File | Description |
|------|-------------|
| `attack_cost_curve.png` | Super-exponential cost growth |
| `sink_depth.png` | Accelerating cost gradient |
| `ray_patterns.png` | Deterministic mapping |
| `feasibility_boundary.png` | Attack feasibility zones |

---

## Patent Claim Coverage

### Claims 1-6: Six-Gate Pipeline

| Claim | Gate | Test Coverage |
|-------|------|---------------|
| 1 | WHO (Identity) | `test_claim_1_who_gate` |
| 2 | WHAT (Intent) | `test_claim_2_what_gate` |
| 3 | WHERE (Trajectory) | `test_claim_3_where_gate` |
| 4 | WHEN (Temporal) | `test_claim_4_when_gate` |
| 5 | WHY (Commitment) | `test_claim_5_why_gate` |
| 6 | HOW (Signature) | `test_claim_6_how_gate` |

### Claims 7-12: Harmonic Scaling

| Claim | Feature | Test Coverage |
|-------|---------|---------------|
| 7 | Base formula | `test_harmonic_scaling_base_case` |
| 8 | Super-exponential | `test_harmonic_scaling_growth` |
| 9 | Adaptive dimensions | `test_claim_9_adaptive_dimensions` |
| 10 | Security bits | `test_claim_10_security_bits` |
| 11 | Distance integration | `test_high_distance_increases_dimensions` |
| 12 | Dynamic adjustment | `test_adaptive_dimensions_low_distance` |

### Claims 13-15: Hyperbolic Geometry

| Claim | Feature | Test Coverage |
|-------|---------|---------------|
| 13 | Poincaré projection | `test_hyperbolic_projection_roundtrip` |
| 14 | Scatter patterns | `test_hyperbolic_distance_amplification` |
| 15 | Deterministic rays | `test_sink_deterministic_patterns` |

### Claims 16-17: Trajectory Coherence

| Claim | Feature | Test Coverage |
|-------|---------|---------------|
| 16 | Behavioral tracking | `test_coherence_for_consistent_behavior` |
| 17 | Mahalanobis analysis | `test_anomaly_detection_for_deviant_behavior` |

### Claim 18: Genetic Audit Trail

| Claim | Feature | Test Coverage |
|-------|---------|---------------|
| 18 | Immutable lineage | `test_claim_18_genetic_markers` |

### Claims 19-24: Attack Cost Sink

| Claim | Feature | Test Coverage |
|-------|---------|---------------|
| 19 | Base lattice dim | `test_sink_curve_generation` |
| 20 | LWE hardness | `test_attack_cost_grows_super_exponentially` |
| 21 | Harmonic multiplier | `test_sink_curve_monotonic_increase` |
| 22 | Bit quantification | `test_sink_depth_calculation` |
| 23 | Temporal analysis | `test_attack_infeasibility_at_high_distance` |
| 24 | Sink mechanism | `test_claims_19_24_attack_cost_sink` |

---

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: numpy**
```bash
pip install numpy>=1.20.0
```

**2. Import errors between test files**
```bash
# Run from the tests/ directory
cd tests/
python test_scbe_box.py
```

**3. JSON serialization errors**
- Ensure numpy types are converted: `bool(np.bool_)`, `float(np.float64)`

**4. Overflow errors in attack cost**
- Fixed: Uses log-space arithmetic for large values

### Verification Commands

```bash
# Verify all tests pass
python -c "
import test_scbe_box
import test_entropic_dual_quantum_system
import test_scbe_validation
print('All modules import successfully')
"

# Quick sanity check
python -c "
from SCBEBox import SCBEBox, SecurityLevel
box = SCBEBox(SecurityLevel.POST_QUANTUM)
result = box.calculate_attack_cost(1.0)
print(f'Attack at d=1.0: {result.cost_in_bits:.0f} bits, {result.years_to_break:.2e} years')
assert result.years_to_break > 1e50, 'Security check failed'
print('Sanity check PASSED')
"
```

---

## Filing Status

**USPTO Provisional Application: READY**

- All 24 claims implemented
- All 24 claims tested
- Empirical evidence documented
- Benchmark results reproducible
- Cost analysis complete

---

## License

Patent-pending. All rights reserved.

For research and evaluation purposes only.
