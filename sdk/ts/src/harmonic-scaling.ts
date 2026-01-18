/**
 * SCBE-AETHERMOORE Harmonic Scaling
 *
 * Core mathematical functions for hyperbolic risk amplification.
 *
 * Primary Form: H(d*, R) = R^(d*²)
 * Bounded Form: H(d*, R) = 1 + α·tanh(β·d*)
 */

import type { Vector6D, HarmonicScaleResult, RiskAssessment } from './types';

// ═══════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════

export const CONSTANTS = {
  /** Perfect fifth ratio (3:2) */
  R_FIFTH: 1.5,

  /** Golden ratio φ */
  PHI: 1.618033988749895,

  /** Default R for scaling */
  DEFAULT_R: 1.5,

  /** Default cavity length */
  DEFAULT_L: 1.0,

  /** Default tolerance for resonance */
  DEFAULT_TOLERANCE: 0.01,

  /** Maximum safe exponent before overflow */
  MAX_SAFE_EXPONENT: 700,

  /** Bounded scaling alpha */
  ALPHA: 10.0,

  /** Bounded scaling beta */
  BETA: 0.5,

  /** ALLOW threshold */
  THRESHOLD_ALLOW: 0.30,

  /** DENY threshold */
  THRESHOLD_DENY: 0.70,
} as const;

// ═══════════════════════════════════════════════════════════════
// Assertions
// ═══════════════════════════════════════════════════════════════

function assertIntGE(name: string, value: number, min: number): void {
  if (!Number.isFinite(value) || value < min) {
    throw new RangeError(`${name} must be >= ${min}, got ${value}`);
  }
}

function assertFinite(value: number, message: string): void {
  if (!Number.isFinite(value)) {
    throw new RangeError(message);
  }
}

function log2(x: number): number {
  return Math.log(x) / Math.LN2;
}

// ═══════════════════════════════════════════════════════════════
// Core Math
// ═══════════════════════════════════════════════════════════════

/**
 * Harmonic scaling: H(d, R) = R^(d²)
 *
 * This is the PRIMARY scaling function that amplifies risk
 * super-exponentially as distance increases.
 *
 * @param d - Hyperbolic distance d*
 * @param R - Harmonic ratio (default: 3/2 = perfect fifth)
 * @returns H(d, R)
 * @throws RangeError if d < 1 or R <= 0 or overflow
 */
export function harmonicScale(d: number, R: number = CONSTANTS.DEFAULT_R): number {
  assertIntGE('d', d, 0);
  if (!(R > 0)) throw new RangeError('R must be > 0');

  const e = d * d * Math.log(R);
  if (e > CONSTANTS.MAX_SAFE_EXPONENT) {
    throw new RangeError('harmonicScale overflow');
  }

  const y = Math.exp(e);
  assertFinite(y, 'harmonicScale overflow');
  return y;
}

/**
 * Bounded harmonic scaling: H(d, R) = 1 + α·tanh(β·d)
 *
 * This bounded form prevents overflow for large d.
 * H ∈ [1, 1 + α]
 *
 * @param d - Hyperbolic distance d*
 * @param alpha - Maximum additional scaling (default: 10)
 * @param beta - Growth rate (default: 0.5)
 * @returns Bounded H(d)
 */
export function harmonicScaleBounded(
  d: number,
  alpha: number = CONSTANTS.ALPHA,
  beta: number = CONSTANTS.BETA
): number {
  if (d < 0) throw new RangeError('d must be >= 0');
  return 1.0 + alpha * Math.tanh(beta * d);
}

/**
 * Compute security bits from base security and distance.
 *
 * Security = baseBits + d² × log₂(R)
 *
 * @param baseBits - Base security level in bits
 * @param d - Hyperbolic distance
 * @param R - Harmonic ratio
 * @returns Total security bits
 */
export function securityBits(
  baseBits: number,
  d: number,
  R: number = CONSTANTS.DEFAULT_R
): number {
  assertIntGE('d', d, 0);
  if (!(R > 0)) throw new RangeError('R must be > 0');
  return baseBits + d * d * log2(R);
}

/**
 * Compute security level multiplier.
 *
 * @param base - Base security level
 * @param d - Hyperbolic distance
 * @param R - Harmonic ratio
 * @returns Security level = base × H(d, R)
 */
export function securityLevel(
  base: number,
  d: number,
  R: number = CONSTANTS.DEFAULT_R
): number {
  return base * harmonicScale(d, R);
}

/**
 * Compute weighted harmonic distance in 6D Langues space.
 *
 * d_H(u, v) = √(Σ gᵢ(uᵢ - vᵢ)²)
 *
 * where g = [1, 1, 1, R, R², R³] (Langues metric tensor diagonal)
 *
 * @param u - First 6D vector
 * @param v - Second 6D vector
 * @returns Weighted distance
 */
export function harmonicDistance(u: Vector6D, v: Vector6D): number {
  const R5 = CONSTANTS.R_FIFTH;
  const g: number[] = [1, 1, 1, R5, R5 * R5, R5 * R5 * R5];

  let s = 0;
  for (let i = 0; i < 6; i++) {
    const d = u[i] - v[i];
    s += g[i] * d * d;
  }
  return Math.sqrt(s);
}

/**
 * Transpose frequency by octaves.
 *
 * @param freq - Base frequency in Hz
 * @param octaves - Number of octaves (can be negative)
 * @returns Transposed frequency
 */
export function octaveTranspose(freq: number, octaves: number): number {
  if (!(freq > 0)) throw new RangeError('freq must be > 0');
  return freq * Math.pow(2, octaves);
}

// ═══════════════════════════════════════════════════════════════
// Risk Assessment
// ═══════════════════════════════════════════════════════════════

/**
 * Compute full risk assessment.
 *
 * Risk' = Risk_base × H(d*, R)
 *
 * @param riskBase - Base behavioral risk [0, 1]
 * @param dStar - Hyperbolic distance to nearest realm
 * @param R - Harmonic ratio
 * @param useBounded - Use bounded tanh form instead of exponential
 * @returns Full risk assessment
 */
export function assessRisk(
  riskBase: number,
  dStar: number,
  R: number = CONSTANTS.DEFAULT_R,
  useBounded: boolean = false
): RiskAssessment {
  const H = useBounded
    ? harmonicScaleBounded(dStar)
    : harmonicScale(Math.max(0, dStar), R);

  const riskPrime = riskBase * H;

  let decision: 'ALLOW' | 'QUARANTINE' | 'DENY';
  if (riskPrime < CONSTANTS.THRESHOLD_ALLOW) {
    decision = 'ALLOW';
  } else if (riskPrime > CONSTANTS.THRESHOLD_DENY) {
    decision = 'DENY';
  } else {
    decision = 'QUARANTINE';
  }

  return {
    riskBase,
    riskPrime,
    H,
    decision,
    timestamp: Date.now(),
  };
}

/**
 * Compute harmonic scale with full metadata.
 *
 * @param dStar - Hyperbolic distance
 * @param R - Harmonic ratio
 * @param baseBits - Base security bits for calculation
 * @returns Full harmonic scale result
 */
export function computeHarmonicScale(
  dStar: number,
  R: number = CONSTANTS.DEFAULT_R,
  baseBits: number = 128
): HarmonicScaleResult {
  let H: number;
  let overflow = false;

  try {
    H = harmonicScale(Math.max(0, dStar), R);
  } catch {
    // Fall back to bounded form on overflow
    H = harmonicScaleBounded(dStar);
    overflow = true;
  }

  return {
    H,
    d_star: dStar,
    R,
    securityBits: securityBits(baseBits, Math.max(0, dStar), R),
    overflow,
  };
}
