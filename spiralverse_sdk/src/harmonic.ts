/**
 * Harmonic Scaling Law Implementation
 *
 * The core mathematical foundation: H(d, R) = R^(d²)
 *
 * Reference: Section 0.1 of SCBE-AETHER-UNIFIED-2026-001
 * Claims: 18, 51, 53
 */

import { PERFECT_FIFTH, HARMONIC_METRIC_TENSOR } from "./constants";

/**
 * Compute the Harmonic Scaling Law: H(d, R) = R^(d²)
 *
 * This is the core formula shared by SCBE (Claim 18) and AETHERMOORE (Claim 51).
 *
 * @param d - Dimension/depth parameter (positive integer, typically 1-7)
 * @param R - Harmonic ratio (default: 1.5, the Perfect Fifth)
 * @returns The harmonic scaling factor R^(d²)
 *
 * Properties:
 * - Super-exponential growth: O(R^(d²)) >> O(e^d)
 * - Dimensional separability: H(d₁+d₂, R) includes cross-term R^(2d₁d₂)
 * - Inverse duality: H(d, R) × H(d, 1/R) = 1
 */
export function harmonicScaling(d: number, R: number = PERFECT_FIFTH): number {
  if (d < 1) {
    throw new Error(`Dimension d must be positive, got ${d}`);
  }
  if (R <= 0) {
    throw new Error(`Ratio R must be positive, got ${R}`);
  }
  return Math.pow(R, d * d);
}

/**
 * Calculate effective security bits with harmonic scaling.
 *
 * @param d - Security dimension (1-7)
 * @param baseBits - Base cryptographic strength (default: AES-128)
 * @param R - Harmonic ratio
 * @returns Total effective security bits
 */
export function securityBits(
  d: number,
  baseBits: number = 128,
  R: number = PERFECT_FIFTH
): number {
  const H = harmonicScaling(d, R);
  const addedBits = Math.log2(H);
  return baseBits + addedBits;
}

/**
 * Security scaling table entry
 */
export interface SecurityTableEntry {
  d: number;
  dSquared: number;
  H: number;
  bitsAdded: number;
  totalEffective: number;
  aesEquivalent: string;
}

/**
 * Generate the harmonic security scaling table.
 */
export function harmonicScalingTable(
  maxD: number = 7,
  R: number = PERFECT_FIFTH
): SecurityTableEntry[] {
  const results: SecurityTableEntry[] = [];

  for (let d = 1; d <= maxD; d++) {
    const H = harmonicScaling(d, R);
    const added = Math.log2(H);
    results.push({
      d,
      dSquared: d * d,
      H,
      bitsAdded: added,
      totalEffective: 128 + added,
      aesEquivalent: `AES-${Math.floor(128 + added)}`,
    });
  }

  return results;
}

/**
 * Compute distance using the 6D harmonic metric tensor.
 *
 * D_H = √(Σ gᵢᵢ × Δcᵢ²)
 *
 * The metric tensor weights the security dimension 3.375× more than position.
 */
export function harmonicMetricDistance(
  c1: readonly number[],
  c2: readonly number[],
  metric: readonly number[] = HARMONIC_METRIC_TENSOR
): number {
  if (c1.length !== c2.length) {
    throw new Error(`Vectors must have same length: ${c1.length} vs ${c2.length}`);
  }
  if (c1.length !== metric.length) {
    throw new Error(`Vector length ${c1.length} doesn't match metric ${metric.length}`);
  }

  let squaredSum = 0;
  for (let i = 0; i < c1.length; i++) {
    const delta = c1[i] - c2[i];
    squaredSum += metric[i] * delta * delta;
  }

  return Math.sqrt(squaredSum);
}

/**
 * Calculate chaos diffusion iterations scaled by harmonic depth.
 *
 * iterations = base × H(d, R)^(1/3)
 */
export function chaosIterations(
  d: number,
  baseIterations: number = 50,
  R: number = PERFECT_FIFTH
): number {
  const H = harmonicScaling(d, R);
  return Math.floor(baseIterations * Math.pow(H, 1 / 3));
}

/**
 * Demonstrate dimensional separability property.
 */
export function dimensionalSeparability(
  d1: number,
  d2: number,
  R: number = PERFECT_FIFTH
): {
  d1: number;
  d2: number;
  combinedD: number;
  HCombined: number;
  Hd1: number;
  Hd2: number;
  crossTerm: number;
  product: number;
  verification: boolean;
} {
  const combinedD = d1 + d2;
  const combinedH = harmonicScaling(combinedD, R);

  const H1 = harmonicScaling(d1, R);
  const H2 = harmonicScaling(d2, R);
  const crossTerm = Math.pow(R, 2 * d1 * d2);

  return {
    d1,
    d2,
    combinedD,
    HCombined: combinedH,
    Hd1: H1,
    Hd2: H2,
    crossTerm,
    product: H1 * crossTerm * H2,
    verification: Math.abs(combinedH - H1 * crossTerm * H2) < 1e-10,
  };
}

/**
 * Verify the inverse duality property: H(d, R) × H(d, 1/R) = 1
 */
export function inverseDuality(
  d: number,
  R: number = PERFECT_FIFTH
): {
  d: number;
  R: number;
  HdR: number;
  HdInvR: number;
  product: number;
  verification: boolean;
} {
  const HdR = harmonicScaling(d, R);
  const HdInvR = harmonicScaling(d, 1 / R);
  const product = HdR * HdInvR;

  return {
    d,
    R,
    HdR,
    HdInvR,
    product,
    verification: Math.abs(product - 1.0) < 1e-10,
  };
}
