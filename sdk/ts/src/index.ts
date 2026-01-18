/**
 * SCBE-AETHERMOORE TypeScript SDK
 *
 * High-precision time-aware implementation of harmonic scaling,
 * HAL (Harmonic Attention Layer), and vacuum acoustics.
 *
 * @packageDocumentation
 */

// ═══════════════════════════════════════════════════════════════
// Type Exports
// ═══════════════════════════════════════════════════════════════

export type {
  // Vector types
  Vector3D,
  Vector6D,
  VectorND,
  Matrix2D,
  Tensor2D,
  Tensor3D,

  // Time types
  Timestamp,
  Duration,
  TimeWindow,
  BreathPhase,
  TimedEvent,

  // Harmonic scaling types
  HarmonicScaleResult,
  RiskAssessment,

  // HAL types
  HALConfig,
  HALOutput,

  // Vacuum acoustics types
  VacuumAcousticsConfig,
  AcousticSource,
  ResonanceResult,
  FluxResult,

  // Voxel types
  Voxel,
  HoloCubeConfig,

  // Consensus types
  ConsensusState,
  ConsensusResult,

  // Utility types
  Result,
  AsyncResult,
  ValidationStatus,
  EventEmitter,
} from './types';

// ═══════════════════════════════════════════════════════════════
// Harmonic Scaling Exports
// ═══════════════════════════════════════════════════════════════

export {
  CONSTANTS,
  harmonicScale,
  harmonicScaleBounded,
  securityBits,
  securityLevel,
  harmonicDistance,
  octaveTranspose,
  assessRisk,
  computeHarmonicScale,
} from './harmonic-scaling';

// ═══════════════════════════════════════════════════════════════
// HAL (Harmonic Attention Layer) Exports
// ═══════════════════════════════════════════════════════════════

export {
  harmonicCouplingMatrix,
  halAttention,
  multiHeadHAL,
  positionToDistance,
  sinusoidalPositions,
} from './hal';

// ═══════════════════════════════════════════════════════════════
// Vacuum Acoustics Exports
// ═══════════════════════════════════════════════════════════════

export {
  ACOUSTIC_CONSTANTS,
  // Nodal surface
  nodalSurface,
  isNodalPoint,
  // Resonance
  resonantFrequencies,
  checkCymaticResonance,
  dampedResonanceAmplitude,
  // Bottle beam
  bottleBeamIntensity,
  findTrapCenter,
  // Flux redistribution
  fluxRedistribution,
  energyConservation,
  // Holographic storage
  createHoloCube,
  readVoxel,
  sampleCube,
  encodeHolographic,
  decodeHolographic,
  // Time-synchronized
  timeVaryingField,
  breathSyncedIntensity,
} from './vacuum-acoustics';

// ═══════════════════════════════════════════════════════════════
// Version
// ═══════════════════════════════════════════════════════════════

export const VERSION = '2.1.0';
