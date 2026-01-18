/**
 * SCBE-AETHERMOORE Type Definitions
 *
 * Core types for harmonic scaling, HAL attention, and vacuum acoustics.
 */

// ═══════════════════════════════════════════════════════════════
// Vector Types
// ═══════════════════════════════════════════════════════════════

/** 3D position vector */
export type Vector3D = [number, number, number];

/** 6D context vector for Langues metric */
export type Vector6D = [number, number, number, number, number, number];

/** Generic N-dimensional vector */
export type VectorND = number[];

/** 2D matrix */
export type Matrix2D = number[][];

/** 3D tensor (batch of matrices) */
export type Tensor2D = number[][];
export type Tensor3D = number[][][];

// ═══════════════════════════════════════════════════════════════
// Time Types (TypeScript's strong suit)
// ═══════════════════════════════════════════════════════════════

/** High-precision timestamp in milliseconds */
export type Timestamp = number;

/** Duration in milliseconds */
export type Duration = number;

/** Time window for consensus */
export interface TimeWindow {
  start: Timestamp;
  end: Timestamp;
  duration: Duration;
}

/** Breathing cycle phase */
export type BreathPhase = 'expansion' | 'contraction' | 'neutral';

/** Time-stamped event */
export interface TimedEvent<T> {
  timestamp: Timestamp;
  data: T;
  phase: BreathPhase;
}

// ═══════════════════════════════════════════════════════════════
// Harmonic Scaling Types
// ═══════════════════════════════════════════════════════════════

/** Harmonic scaling result */
export interface HarmonicScaleResult {
  H: number;           // H(d*, R)
  d_star: number;      // Input distance
  R: number;           // Harmonic ratio
  securityBits: number;
  overflow: boolean;
}

/** Risk assessment */
export interface RiskAssessment {
  riskBase: number;
  riskPrime: number;
  H: number;
  decision: 'ALLOW' | 'QUARANTINE' | 'DENY';
  timestamp: Timestamp;
}

// ═══════════════════════════════════════════════════════════════
// HAL (Harmonic Attention Layer) Types
// ═══════════════════════════════════════════════════════════════

/** HAL configuration */
export interface HALConfig {
  d_model: number;
  n_heads: number;
  R?: number;
  d_max?: number;
  normalize?: boolean;
}

/** Attention output with metadata */
export interface HALOutput {
  output: Tensor3D;
  attentionWeights: Tensor3D;
  couplingMatrix: Matrix2D;
  timestamp: Timestamp;
}

// ═══════════════════════════════════════════════════════════════
// Vacuum Acoustics Types
// ═══════════════════════════════════════════════════════════════

/** Vacuum acoustics configuration */
export interface VacuumAcousticsConfig {
  L?: number;         // Cavity length
  c?: number;         // Speed of sound
  gamma: number;      // Damping coefficient
  R?: number;         // Harmonic ratio
  resolution?: number;
}

/** Acoustic source for interference */
export interface AcousticSource {
  pos: Vector3D;
  phase: number;
  amplitude?: number;
}

/** Cymatic resonance result */
export interface ResonanceResult {
  isResonant: boolean;
  nodalValue: number;
  tolerance: number;
  frequency: number;
}

/** Flux redistribution result */
export interface FluxResult {
  canceled: number;
  corners: [number, number, number, number];
  totalEnergy: number;
}

// ═══════════════════════════════════════════════════════════════
// Cymatic Voxel Types
// ═══════════════════════════════════════════════════════════════

/** Voxel in 3D holographic storage */
export interface Voxel {
  position: Vector3D;
  intensity: number;
  phase: number;
  timestamp: Timestamp;
}

/** Holographic QR cube configuration */
export interface HoloCubeConfig {
  resolution: number;  // Voxels per side
  wavelength: number;
  sources: AcousticSource[];
}

// ═══════════════════════════════════════════════════════════════
// Consensus Types
// ═══════════════════════════════════════════════════════════════

/** Dual lattice consensus state */
export type ConsensusState =
  | 'pending'
  | 'kyber_only'
  | 'dilithium_only'
  | 'consensus'
  | 'failed'
  | 'timeout';

/** Consensus result */
export interface ConsensusResult {
  state: ConsensusState;
  settled: boolean;
  settlingKey?: Uint8Array;
  arrivalTime: Timestamp;
  deltaT: Duration;
  kyberValid: boolean;
  dilithiumValid: boolean;
}

// ═══════════════════════════════════════════════════════════════
// Utility Types
// ═══════════════════════════════════════════════════════════════

/** Result type for operations that can fail */
export type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E };

/** Async result */
export type AsyncResult<T, E = Error> = Promise<Result<T, E>>;

/** Validation status */
export type ValidationStatus = 'valid' | 'invalid' | 'pending' | 'expired';

/** Generic event emitter interface */
export interface EventEmitter<Events extends Record<string, unknown>> {
  on<K extends keyof Events>(event: K, handler: (data: Events[K]) => void): void;
  off<K extends keyof Events>(event: K, handler: (data: Events[K]) => void): void;
  emit<K extends keyof Events>(event: K, data: Events[K]): void;
}
