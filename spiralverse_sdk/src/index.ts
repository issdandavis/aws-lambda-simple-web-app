/**
 * SCBE-AETHERMOORE: Unified Mathematical Specification SDK
 *
 * Spiralverse Context-Bound Encryption with AETHERMOORE Harmonic Physics Framework
 *
 * Document ID: SCBE-AETHER-UNIFIED-2026-001
 * Version: 2.0.0
 * Author: Isaac Davis
 */

// Constants
export {
  GOLDEN_RATIO,
  PERFECT_FIFTH,
  PHI_AETHER,
  LAMBDA_ISAAC,
  OMEGA_SPIRAL,
  ALPHA_ABH,
  EVENT_HORIZON_THRESHOLD,
  SOLITON_THRESHOLD,
  ENTROPY_EXPORT_RATE,
  PLANETARY_FREQUENCIES,
  PLANETARY_SEED,
  HARMONIC_FREQUENCY_MAP,
  HARMONIC_METRIC_TENSOR,
  PARAMETER_RANGES,
  validateConstants,
} from "./constants";

export type { PlanetaryFrequency } from "./constants";

// Harmonic Scaling
export {
  harmonicScaling,
  securityBits,
  harmonicScalingTable,
  harmonicMetricDistance,
  chaosIterations,
  dimensionalSeparability,
  inverseDuality,
} from "./harmonic";

export type { SecurityTableEntry } from "./harmonic";

// Context
export {
  contextToTuple,
  contextToAethermoore,
  weightedContext,
  validateContext,
  contextCommitment,
  harmonicContextCommitment,
  contextDistance,
  deriveChaosParams,
  contextCommitmentHex,
  verifyContextBinding,
} from "./context";

export type { ContextVector } from "./context";

// Physics Validation
export {
  timeDilation,
  eventHorizonDistance,
  timeDilationTable,
  solitonThresholdCheck,
  signalCoherence,
  oracleShift,
  groverIterationLimit,
  quantumAttackSimulation,
  entropyExport,
  entropyOverCycles,
  secondLawCompliance,
  runAllPhysicsTests,
  physicsSummary,
} from "./physics";

// Package metadata
export const VERSION = "2.0.0";
export const DOCUMENT_ID = "SCBE-AETHER-UNIFIED-2026-001";
export const AUTHOR = "Isaac Davis";
