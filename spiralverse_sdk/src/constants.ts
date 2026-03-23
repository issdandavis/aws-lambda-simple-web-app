/**
 * AETHERMOORE Constants
 *
 * Fundamental constants derived from the Golden Ratio (φ) and Perfect Fifth (R₅).
 *
 * Reference: Appendix A of SCBE-AETHER-UNIFIED-2026-001
 */

// Fundamental Mathematical Constants
export const GOLDEN_RATIO = (1 + Math.sqrt(5)) / 2; // φ ≈ 1.618
export const PERFECT_FIFTH = 3 / 2; // R₅ = 1.5 (exact)

// Derived AETHERMOORE Constants
export const PHI_AETHER = Math.pow(GOLDEN_RATIO, 1 / PERFECT_FIFTH); // 1.378...
export const LAMBDA_ISAAC = PERFECT_FIFTH * Math.pow(GOLDEN_RATIO, 2); // 3.927...
export const OMEGA_SPIRAL = (2 * Math.PI) / Math.pow(GOLDEN_RATIO, 3); // 0.934...
export const ALPHA_ABH = GOLDEN_RATIO + PERFECT_FIFTH; // 3.118...

// Critical Thresholds
export const EVENT_HORIZON_THRESHOLD = ALPHA_ABH * LAMBDA_ISAAC; // 12.24...
export const SOLITON_THRESHOLD = PHI_AETHER * (1 - OMEGA_SPIRAL); // 0.0909...
export const ENTROPY_EXPORT_RATE = 1 - OMEGA_SPIRAL; // 6.6%

/**
 * Planetary Frequency Data
 * The Solar System forms a D Major 7th chord
 */
export interface PlanetaryFrequency {
  name: string;
  periodDays: number;
  baseFrequencyHz: number;
  octaves: number;
  audibleHz: number;
  note: string;
  chordDegree: string;
}

export const PLANETARY_FREQUENCIES: Record<string, PlanetaryFrequency> = {
  mercury: {
    name: "Mercury",
    periodDays: 87.97,
    baseFrequencyHz: 1.316e-7,
    octaves: 30,
    audibleHz: 141.27,
    note: "C#3",
    chordDegree: "Major 7th (alt)",
  },
  venus: {
    name: "Venus",
    periodDays: 224.7,
    baseFrequencyHz: 5.151e-8,
    octaves: 32,
    audibleHz: 221.23,
    note: "A3",
    chordDegree: "Perfect 5th",
  },
  earth: {
    name: "Earth",
    periodDays: 365.25,
    baseFrequencyHz: 3.169e-8,
    octaves: 32,
    audibleHz: 136.1,
    note: "C#3",
    chordDegree: "Major 7th",
  },
  mars: {
    name: "Mars",
    periodDays: 687.0,
    baseFrequencyHz: 1.685e-8,
    octaves: 33,
    audibleHz: 144.72,
    note: "D3",
    chordDegree: "Root",
  },
  jupiter: {
    name: "Jupiter",
    periodDays: 4333,
    baseFrequencyHz: 2.671e-9,
    octaves: 36,
    audibleHz: 183.58,
    note: "F#3",
    chordDegree: "Major 3rd",
  },
  saturn: {
    name: "Saturn",
    periodDays: 10759,
    baseFrequencyHz: 1.076e-9,
    octaves: 37,
    audibleHz: 147.85,
    note: "D3",
    chordDegree: "Root (octave)",
  },
};

/**
 * D Major 7th Chord Seed (Claim 52)
 * Used to seed harmonic parameters in intent configuration
 */
export const PLANETARY_SEED = {
  root: 144.72, // Mars (D)
  third: 183.58, // Jupiter (F#)
  fifth: 221.23, // Venus (A)
  seventh: 136.1, // Earth (C#)
};

/**
 * Harmonic frequency mapping by security dimension
 */
export const HARMONIC_FREQUENCY_MAP: Record<number, number> = {
  1: PLANETARY_FREQUENCIES.mars.audibleHz, // 144.72 Hz
  2: PLANETARY_FREQUENCIES.jupiter.audibleHz, // 183.58 Hz
  3: PLANETARY_FREQUENCIES.venus.audibleHz, // 221.23 Hz
  4: PLANETARY_FREQUENCIES.earth.audibleHz, // 136.10 Hz
  5: PLANETARY_FREQUENCIES.saturn.audibleHz, // 147.85 Hz
  6: PLANETARY_FREQUENCIES.mercury.audibleHz, // 141.27 Hz
  7:
    (PLANETARY_SEED.root +
      PLANETARY_SEED.third +
      PLANETARY_SEED.fifth +
      PLANETARY_SEED.seventh) /
    4, // Full chord
};

/**
 * 6D Harmonic Metric Tensor: g = diag(1, 1, 1, R₅, R₅², R₅³)
 */
export const HARMONIC_METRIC_TENSOR: readonly number[] = [
  1.0, // time / x
  1.0, // device_id / y
  1.0, // threat_level / z
  PERFECT_FIFTH, // entropy / velocity: 1.5
  Math.pow(PERFECT_FIFTH, 2), // server_load / priority: 2.25
  Math.pow(PERFECT_FIFTH, 3), // behavior_stability / security: 3.375
] as const;

/**
 * Recommended parameter ranges (Appendix C)
 */
export const PARAMETER_RANGES = {
  rChaos: { min: 3.97, max: 4.0, default: 3.99 },
  RHarmonic: { value: PERFECT_FIFTH },
  dDimension: { min: 1, max: 7, default: 6 },
  fractalIterations: { min: 30, max: 100, default: 50 },
  escapeRadius: { min: 1.5, max: 3.0, default: 2.0 },
  energyThresholdK: { min: 2, max: 4, default: 3 },
  trustAlpha: { min: 0.8, max: 0.95, default: 0.9 },
  tauParticipate: { min: 0.2, max: 0.4, default: 0.3 },
  epsilonCoherence: { min: 0.1, max: 0.25, default: 0.15 },
  phasePeriodSeconds: { min: 30, max: 120, default: 60 },
} as const;

/**
 * Validate AETHERMOORE constants against expected values
 */
export function validateConstants(): Record<string, boolean> {
  return {
    goldenRatio: Math.abs(GOLDEN_RATIO - 1.6180339887) < 1e-9,
    perfectFifth: PERFECT_FIFTH === 1.5,
    phiAether: Math.abs(PHI_AETHER - 1.378240772) < 1e-8,
    lambdaIsaac: Math.abs(LAMBDA_ISAAC - 3.9270509831) < 1e-8,
    omegaSpiral: Math.abs(OMEGA_SPIRAL - 0.9340017595) < 1e-8,
    alphaAbh: Math.abs(ALPHA_ABH - 3.1180339887) < 1e-8,
    eventHorizon: Math.abs(EVENT_HORIZON_THRESHOLD - 12.2446) < 0.001,
  };
}
