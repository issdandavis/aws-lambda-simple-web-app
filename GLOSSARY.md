# SCBE TECHNICAL GLOSSARY
## Alphabetical Index of Patent Terms

---

## A

### Adaptive K-Controller
- **Definition**: Dynamic cryptographic key length adjustment mechanism responding to quantum threat estimates
- **Formula**: k ∈ [128, 512] bits, adjusted based on threat level τ
- **Patent Claims**: 19, 20
- **Purpose**: Increases computational work factor preemptively when quantum attacks detected

### Anomalous Agent
- **Definition**: Agent exhibiting chaotic decimal drift patterns (δ increasing monotonically)
- **Detection**: Multi-dimensional correlation across 3+ context dimensions
- **Patent Claims**: 14(c), 18
- **Response**: Immediate quarantine + entropy sink deployment
- **See Also**: Decimal Drift, Explorer Tag

### Attack Cost Function
- **Definition**: Super-exponential computational cost for adversaries
- **Formula**: Cost(d) = base_cost × H(d,R) = base_cost × R^(1+d²)
- **Patent Claims**: 1, 6, 18
- **Measured**: 25,251× harder at d=10 dimensions
- **See Also**: Harmonic Scaling, Entropy Sink

---

## C

### Coherence Score
- **Definition**: Numerical measure of intent trajectory consistency over time
- **Formula**: S(τ) = Σᵢ wᵢ D(cᵢ, cᵢ₋₁) where D is weighted Euclidean distance
- **Patent Claims**: 1, 3, 10, 11
- **Threshold**: τ < 2.0 for valid trajectories
- **See Also**: Mahalanobis Distance, Metric Tensor

### Complex Spin Vector
- **Definition**: Phase-modulated component encoding emotional/priority states
- **Formula**: z = |z|e^(iθ) where θ encodes semantic domain
- **Patent Claims**: 2, 12, 13, 15
- **Projection**: f(z) = [|z|cosθ, |z|sinθ] preserves phase as geometric angle
- **See Also**: Phase Modulation, Wave Interference

### Context Vector (6D)
- **Definition**: Six-dimensional representation of agent state
- **Components**:
  1. **v₁**: Identity (hardware root-of-trust)
  2. **v₂**: Intent (complex value with phase/magnitude)
  3. **v₃**: Trajectory (coherence score)
  4. **v₄**: Timing (phase-lock value)
  5. **v₅**: Commitment (cryptographic hash)
  6. **v₆**: Signature (lattice-based digital signature)
- **Patent Claims**: 1, 7
- **Space**: Embedded in weighted geometric manifold
- **See Also**: Geometric Manifold, Metric Tensor

### Cox Constant (c)
- **Definition**: Self-referential equilibrium point where c = e^(π/c)
- **Value**: c ≈ 2.9265358979...
- **Derivation**: Newton-Raphson on f(x) = x - e^(π/x)
- **Patent Claims**: TAHS equilibrium
- **See Also**: Harmonic Scaling

---

## D

### Decimal Drift (δ)
- **Definition**: Non-integer component of state value indicating instability
- **Formula**: **δ = s - floor(s)** where s is observed state
- **Patent Claims**: 14, 15, 16, 17, 18
- **Classification**:
  - δ < 0.35: **STABLE**
  - 0.35 ≤ δ < 0.50: **DRIFTING**
  - δ ≥ 0.50: **ANOMALOUS**
- **Key Innovation**: Continuous signal vs binary classification
- **See Also**: Multi-State Encoding, Variable Drift

### Destructive Interference
- **Definition**: Wave cancellation technique for malicious intent signals
- **Formula**: counter_wave = -1 × threat_spectrum × amplification
- **Patent Claims**: 13, 15
- **Purpose**: Active neutralization of attack signals
- **See Also**: Wave Interference, Inverse Phase Injection

---

## E

### Emotional Spin Wave
- **Definition**: Phase-encoded representation of agent intent mood/urgency
- **Formula**: v(t) = A · e^(i(ωt + φ))
- **Parameters**:
  - A: Amplitude [0.2, 1.5]
  - ω: Frequency (0.1 stable, 1.0 reactive)
  - φ: Phase offset [0, 2π]
- **Patent Claims**: 12, 13, 15
- **See Also**: Complex Spin Vector, Intent Signature

### Entropic Expansion Sink
- **Definition**: Defensive mechanism that grows keyspace faster than attacks probe
- **Property**: dW/dt > quantum_search_rate
- **Formula**: N(t) = N₀ · e^(kt), Progress P(t) → 0 as t → ∞
- **Patent Claims**: 18, 19, 20
- **Analogy**: "Target moves faster than search beam"
- **See Also**: Attack Cost Function, Golden Ratio

### Explorer Tag
- **Definition**: Fractional trust weight (e.g., 0.3i) assigned to suspicious agents
- **Purpose**: Quarantine with monitoring (not immediate banishment)
- **Patent Claims**: 14, 16, 17
- **Self-Healing**: Can be restored to 1.0 if agent stabilizes
- **See Also**: Hive Triage, Anomalous Agent

---

## F

### FFT (Fast Fourier Transform)
- **Definition**: Spectral decomposition of intent trajectories into frequency components
- **Formula**: X[k] = Σₙ c[n] · e^(-i2πkn/N)
- **Application**: Detects periodicity patterns indicating hallucination/injection
- **Patent Claims**: 9, 10, 11
- **Anomaly Threshold**: High-freq ratio r > 0.4
- **See Also**: Spectral Intent Decomposition

### Forward-Secure Ratchet
- **Definition**: Cryptographic key evolution with automatic past-state deletion
- **Property**: Compromise at time t doesn't reveal keys at time t-1
- **Patent Claims**: 20
- **See Also**: Temporal Lattice, Phase Lock

---

## G

### Geometric Manifold
- **Definition**: Weighted 6D+ space where context vectors are embedded
- **Metric**: Riemannian metric with tensor G = diag([1,1,1,φ,φ²,φ³])
- **Patent Claims**: 1, 2, 3
- **Curvature**: Hyperbolic (κ > 0) for defensive amplification
- **See Also**: Hyperbolic Embedding, Metric Tensor

### Golden Ratio (φ)
- **Value**: φ = (1+√5)/2 ≈ 1.618033988749895
- **Properties**:
  - φ² = φ + 1
  - 1/φ = φ - 1
  - "Most irrational" number
- **Usage**: Geometric parameter R_g controlling metric tensor weights
- **Patent Claims**: 1, 7
- **See Also**: Harmonic Scaling, Metric Tensor

---

## H

### Harmonic Scaling Function H(d,R)
- **Definition**: Work-factor function producing super-exponential cost growth
- **Formula**: **H(d,R) = R^(1+d²)**
- **Hardened v4.0**: H(d,R_h) = R_h^(d²) (neutral at d=0)
- **Patent Claims**: 1, 2, 3, 4, 6, 7, 8
- **Parameters**:
  - R_h = 1.5 (locked)
  - R_g = φ (locked)
  - d: Divergence score
- **Measured Advantage**: +101.76 bits at d=1.0
- **See Also**: Attack Cost Function, Entropic Expansion

### Hyperbolic Embedding
- **Definition**: Projection of context vectors into negatively-curved space
- **Formula**: h(c) = c / (1 + κ||c||²)
- **Distance**: d_h = arccosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))
- **Patent Claims**: 3, 16
- **Property**: Exponential distance growth near boundary
- **Purpose**: Amplifies decimal drift for early anomaly detection
- **See Also**: Poincaré Disk, Geometric Amplification

---

## I

### Intent Signature
- **Definition**: Spectral FFT fingerprint of agent's behavioral trajectory
- **Components**: Frequency distribution, phase relationships, timbre metrics
- **Patent Claims**: 10, 11, 12
- **Detection**: Dissonant patterns indicate hallucination/injection
- **See Also**: FFT, Spectral Intent Decomposition

### Inverse Phase Injection
- **Definition**: Counter-attack technique generating 180° phase-shifted wave
- **Formula**: counter_wave = ifft(-1 × fft(threat_trajectory)) × 2.0
- **Effect**: Destructive interference cancels malicious signal
- **See Also**: Destructive Interference

---

## K

### Klein Bottle Topology
- **Definition**: Non-orientable 4D manifold for intent+time subspace
- **Properties**:
  - Orientation-reversing (automatic negative flip)
  - No inside/outside boundary
  - Resolves infinite loop paradoxes
- **Parametric**: x,y,z,w functions of (u,v) with R = φ²
- **Patent Claims**: 19 (proposed)
- **See Also**: Geometric Manifold, Reverse Gravity

---

## L

### Lattice-Based Signature
- **Definition**: Post-quantum digital signature using ML-DSA-65 (Dilithium)
- **Component**: v₆ of 6D context vector
- **Patent Claims**: 7, 14
- **Security**: Resistant to Shor's algorithm
- **See Also**: ML-DSA, NIST PQC Standards

### Lyapunov Exponent (λ)
- **Definition**: Measure of chaotic divergence in trajectory dynamics
- **Threshold**: λ > 0.1 indicates chaos (anomalous)
- **Patent Claims**: 21
- **Purpose**: Certifies chaotic containment of attacks
- **See Also**: Chaotic Trajectory Dynamics

---

## M

### Mahalanobis Distance
- **Definition**: Metric tensor-weighted distance between context vectors
- **Formula**: D(c₁,c₂) = √((c₁-c₂)* G (c₁-c₂))
- **Patent Claims**: 3, 7
- **See Also**: Metric Tensor, Coherence Score

### Mars Frequency
- **Value**: 144.72 Hz
- **Derivation**: (Mars orbital period in seconds)⁻¹ × 2³³
- **Purpose**: Natural harmonic reference frequency
- **See Also**: Harmonic Scaling

### Metric Tensor (G)
- **Definition**: Weight matrix controlling distance computation in manifold
- **Formula**: G = diag([1, 1, 1, R_g, R_g², R_g³])
- **Patent Claims**: 1, 7
- **Purpose**: Emphasizes temporal/commitment dimensions over identity
- **See Also**: Geometric Manifold, Golden Ratio

### ML-KEM (Kyber)
- **Definition**: NIST-standardized post-quantum key encapsulation mechanism
- **Parameter Set**: ML-KEM-768 (security level 3)
- **Patent Claims**: 7, 14, 16
- **Context Binding**: Decryption requires manifold validation
- **See Also**: Post-Quantum Cryptography

### Multi-State Encoding
- **Definition**: Base-6 state representation (0-5) instead of binary
- **Entropy**: ~2.58 bits per position vs. 1 bit for binary
- **Patent Claims**: 14, 15
- **Semantic Mapping**:
  - 0: Null/undefined
  - 1-2: Low priority
  - 3-4: Medium priority
  - 5: Critical/urgent
- **See Also**: Decimal Drift, Variable State

---

## O

### Omni-Directional Intent Propagation
- **Definition**: Broadcast of intent waves to all agents in swarm topology
- **Formula**: w(r,t) = (A/r) · sin(kr - ωt + φ)
- **Pattern**: Star-mesh hybrid with wave superposition
- **Patent Claims**: 15, 16
- **Coordination**: Constructive = harmony, destructive = conflict
- **See Also**: Wave Interference, Swarm Dynamics

---

## P

### Phase Lock
- **Definition**: Temporal synchronization between sender and receiver
- **Component**: v₄ of 6D context vector
- **Patent Claims**: 4, 15
- **Tolerance**: ±30ms deviation threshold
- **Failure Mode**: Produces decimal drift when misaligned
- **See Also**: Temporal Consistency Gate

### Phase Modulation
- **Definition**: Encoding semantic information as complex phase angle θ
- **Formula**: s_effective = round(s_base · cos(θ) + s_base · sin(θ) · w_imag)
- **Patent Claims**: 15
- **Purpose**: Creates cryptographic key dependency
- **See Also**: Complex Spin Vector, Decimal Drift

### Poincaré Disk
- **Definition**: Hyperbolic space model with boundary at unit circle
- **Embedding**: Projects 6D context vectors into curved space
- **Patent Claims**: 16
- **Property**: Distance → ∞ as radius → 1
- **See Also**: Hyperbolic Embedding

### Post-Quantum Cryptography (PQC)
- **Definition**: Cryptographic algorithms resistant to quantum attacks
- **Standards**: NIST FIPS 203 (ML-KEM), FIPS 204 (ML-DSA)
- **Patent Claims**: 7, 14, 25
- **See Also**: ML-KEM, Lattice-Based Signature

---

## Q

### Q16.16 Fixed-Point
- **Definition**: Deterministic cross-platform arithmetic format
- **Range**: [-32768, 32767.99998]
- **Precision**: 1/65536 ≈ 0.000015
- **Purpose**: Reproducible calculations across architectures
- **Patent Claims**: Implementation detail
- **See Also**: Deterministic Computation

### Quasicrystal Lattice
- **Definition**: Aperiodic tiling structure with φ-based spacing
- **Property**: Non-repeating but ordered (Penrose-like)
- **Patent Claims**: 25
- **Purpose**: Ray-tracing noise generation for lattice crypto
- **See Also**: Golden Ratio, Lattice-Based Signature

---

## R

### Rational/Irrational Flux
- **Definition**: Continuous oscillation between rational and irrational values
- **Mechanism**: φ-weighted metric spreads irrationality
- **Detection**: δ captures irrational residue
- **Patent Claims**: 14-18 (implicit)
- **Security**: Attackers can't forge stable rational states
- **See Also**: Decimal Drift, Golden Ratio

---

## S

### Sacred Tongues (Conlang)
- **Definition**: Six constructed language roots at 60° phase intervals
- **Mapping**:
  - KO (0°): Origin/identity
  - AV (60°): Affirmation
  - RU (120°): Query/reflection
  - CA (180°): Negation/opposition
  - UM (240°): Uncertainty
  - DR (300°): Completion/closure
- **Patent Claims**: 4, 5
- **See Also**: Phase Modulation, Emotional Spin

### Soliton Propagation
- **Definition**: Shape-preserving wave packet from NLSE
- **Formula**: ψ(x,t) = A · sech((x-vt)/w) · e^(i(kx-ωt))
- **Property**: Maintains coherence over distance
- **Patent Claims**: 8 (implicit)
- **See Also**: Wave Interference

### Spectral Intent Decomposition
- **Definition**: FFT analysis of behavioral trajectories
- **Output**: Power spectrum P[k] = |X[k]|²
- **Anomaly**: High-frequency ratio > 0.4
- **Patent Claims**: 9, 10, 11
- **See Also**: FFT, Intent Signature

---

## T

### Temporal Drift Accumulation
- **Definition**: Decay-weighted sum of historical drift values
- **Formula**: S_drift = Σᵢ (δᵢ × decay^(M-i))
- **decay_factor**: 0.9 (configurable)
- **Patent Claims**: 17
- **Classification**:
  - EXPLORATORY: Gradual rise then stable
  - OSCILLATORY: Fluctuates (probing)
  - CHAOTIC: Monotonic increase (attack)
- **See Also**: Decimal Drift

### Temporal Consistency Gate
- **Definition**: Phase-lock validation checkpoint
- **Tolerance**: Δt < 30ms
- **Response**: Reject or Explorer Tag if violated
- **Patent Claims**: 4
- **See Also**: Phase Lock

---

## V

### Variable Drift
- **Definition**: Continuous-valued instability metric (vs binary)
- **Range**: δ ∈ [0, 1)
- **Key Innovation**: Captures gradual state transitions
- **Patent Claims**: 14-18
- **See Also**: Decimal Drift, Multi-State Encoding

---

## W

### Wave Interference
- **Definition**: Superposition of multiple agent intent signals
- **Formula**: v_total(t) = Σⱼ vⱼ(t)
- **Coherence**: |v_total|/M (high = harmony, low = conflict)
- **Patent Claims**: 12, 13
- **See Also**: Emotional Spin Wave, Destructive Interference

---

## Constants Reference

| Constant | Value | Definition |
|----------|-------|------------|
| φ (phi) | 1.618033988749... | Golden ratio (1+√5)/2 |
| φ² | 2.618033988749... | φ + 1 |
| φ³ | 4.236067977499... | φ² + φ |
| R_h | 1.5 | Harmonic ratio (locked) |
| R_g | φ | Geometric ratio (locked) |
| κ | 1.0 | Hyperbolic curvature |
| τ_stable | 0.35 | Drift stability threshold |
| τ_anomaly | 0.50 | Drift anomaly threshold |
| c (Cox) | 2.9265358979... | e^(π/c) equilibrium |
| Mars freq | 144.72 Hz | Natural harmonic reference |

---

## Claim Cross-Reference

| Term | Primary Claims |
|------|----------------|
| Decimal Drift (δ) | 14, 15, 16, 17, 18 |
| Harmonic Scaling H(d,R) | 1, 2, 3, 4, 6, 7, 8 |
| Context Vector 6D | 1, 7 |
| Complex Spin | 2, 12, 13, 15 |
| Hyperbolic Embedding | 3, 16 |
| FFT Spectral | 9, 10, 11 |
| Phase Modulation | 4, 15 |
| Temporal Accumulation | 17 |
| Multi-Dimensional Correlation | 18 |
| Lattice Crypto (PQC) | 7, 14, 25 |

---

*This glossary is maintained as the authoritative reference for all SCBE patent terminology.*
