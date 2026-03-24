/**
 * SCBE Context Commitment
 *
 * Turns 6 environmental measurements into a unique cryptographic fingerprint.
 *
 * Reference: Section 1.1 of SCBE-AETHER-UNIFIED-2026-001
 * Claims: 1(b), 53
 */

import { createHash } from "crypto";
import { PERFECT_FIFTH, HARMONIC_METRIC_TENSOR } from "./constants";
import { harmonicMetricDistance } from "./harmonic";

/**
 * The 6-dimensional context vector for SCBE authorization.
 *
 * SCBE dimensions:
 *   c₁: time - Unix timestamp
 *   c₂: device_id - Numeric device identifier
 *   c₃: threat_level - Current danger level (0-10)
 *   c₄: entropy - System randomness (0-1)
 *   c₅: server_load - System busyness (0-1)
 *   c₆: behavior_stability - How "normal" actions are (0-1)
 */
export interface ContextVector {
  time: number;
  deviceId: number;
  threatLevel: number;
  entropy: number;
  serverLoad: number;
  behaviorStability: number;
}

/**
 * Convert context to tuple for computation
 */
export function contextToTuple(
  ctx: ContextVector
): [number, number, number, number, number, number] {
  return [
    ctx.time,
    ctx.deviceId,
    ctx.threatLevel,
    ctx.entropy,
    ctx.serverLoad,
    ctx.behaviorStability,
  ];
}

/**
 * Map to AETHERMOORE 6D vector notation
 */
export function contextToAethermoore(ctx: ContextVector): {
  x: number;
  y: number;
  z: number;
  v: number;
  p: number;
  s: number;
} {
  return {
    x: ctx.time,
    y: ctx.deviceId,
    z: ctx.threatLevel,
    v: ctx.entropy,
    p: ctx.serverLoad,
    s: ctx.behaviorStability,
  };
}

/**
 * Apply harmonic weighting to context components.
 * Weights: (1, 1, 1, R, R², R³)
 */
export function weightedContext(
  ctx: ContextVector,
  R: number = PERFECT_FIFTH
): [number, number, number, number, number, number] {
  return [
    ctx.time,
    ctx.deviceId,
    ctx.threatLevel,
    R * ctx.entropy,
    Math.pow(R, 2) * ctx.serverLoad,
    Math.pow(R, 3) * ctx.behaviorStability,
  ];
}

/**
 * Validate context vector components are in expected ranges.
 */
export function validateContext(
  ctx: ContextVector
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (ctx.time < 0) {
    errors.push("time must be non-negative");
  }
  if (ctx.deviceId < 0) {
    errors.push("deviceId must be non-negative");
  }
  if (ctx.threatLevel < 0 || ctx.threatLevel > 10) {
    errors.push("threatLevel must be in [0, 10]");
  }
  if (ctx.entropy < 0 || ctx.entropy > 1) {
    errors.push("entropy must be in [0, 1]");
  }
  if (ctx.serverLoad < 0 || ctx.serverLoad > 1) {
    errors.push("serverLoad must be in [0, 1]");
  }
  if (ctx.behaviorStability < 0 || ctx.behaviorStability > 1) {
    errors.push("behaviorStability must be in [0, 1]");
  }

  return { valid: errors.length === 0, errors };
}

/**
 * Pack context values into bytes for hashing
 */
function packContext(values: number[]): Buffer {
  const buffer = Buffer.alloc(values.length * 8);
  values.forEach((val, i) => {
    buffer.writeDoubleLE(val, i * 8);
  });
  return buffer;
}

/**
 * Standard context commitment: SHA256(c₁, c₂, c₃, c₄, c₅, c₆)
 *
 * Produces a 256-bit cryptographic fingerprint of the context.
 */
export function contextCommitment(ctx: ContextVector): Buffer {
  const tuple = contextToTuple(ctx);
  const packed = packContext(tuple);
  return createHash("sha256").update(packed).digest();
}

/**
 * Harmonic-enhanced context commitment: SHA256(c₁, c₂, c₃, R·c₄, R²·c₅, R³·c₆)
 *
 * Weights higher dimensions more heavily, making behavior_stability
 * contribute 3.375× more entropy than time.
 */
export function harmonicContextCommitment(
  ctx: ContextVector,
  R: number = PERFECT_FIFTH
): Buffer {
  const weighted = weightedContext(ctx, R);
  const packed = packContext(weighted);
  return createHash("sha256").update(packed).digest();
}

/**
 * Compute distance between two context vectors
 */
export function contextDistance(
  c1: ContextVector,
  c2: ContextVector,
  useHarmonicMetric: boolean = true
): number {
  const t1 = contextToTuple(c1);
  const t2 = contextToTuple(c2);

  if (useHarmonicMetric) {
    return harmonicMetricDistance(t1, t2, HARMONIC_METRIC_TENSOR);
  } else {
    // Euclidean distance
    let sum = 0;
    for (let i = 0; i < 6; i++) {
      sum += Math.pow(t1[i] - t2[i], 2);
    }
    return Math.sqrt(sum);
  }
}

/**
 * Derive chaos map parameters (r, x₀) from context and key
 */
export function deriveChaosParams(
  ctx: ContextVector,
  key: Buffer,
  useHarmonic: boolean = true
): { r: number; x0: number } {
  const commitment = useHarmonic
    ? harmonicContextCommitment(ctx)
    : contextCommitment(ctx);

  // Combine commitment with key
  const combined = createHash("sha256")
    .update(Buffer.concat([commitment, key]))
    .digest();

  // Extract r from first 8 bytes, scale to [3.97, 4.0)
  const rRaw = combined.readBigUInt64LE(0);
  const rNormalized = Number(rRaw) / Number(BigInt(2) ** BigInt(64));
  const r = 3.97 + rNormalized * 0.03;

  // Extract x₀ from next 8 bytes, scale to (0, 1)
  const x0Raw = combined.readBigUInt64LE(8);
  const x0 = 0.001 + (Number(x0Raw) / Number(BigInt(2) ** BigInt(64))) * 0.998;

  return { r, x0 };
}

/**
 * Context commitment as hex string
 */
export function contextCommitmentHex(
  ctx: ContextVector,
  useHarmonic: boolean = true
): string {
  const commitment = useHarmonic
    ? harmonicContextCommitment(ctx)
    : contextCommitment(ctx);
  return commitment.toString("hex");
}

/**
 * Verify that a context matches an expected commitment
 */
export function verifyContextBinding(
  ctx: ContextVector,
  expectedCommitment: Buffer,
  useHarmonic: boolean = true
): boolean {
  const actual = useHarmonic
    ? harmonicContextCommitment(ctx)
    : contextCommitment(ctx);
  return actual.equals(expectedCommitment);
}
