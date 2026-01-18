/**
 * Vacuum Acoustics – Cymatic Resonance & Holographic Storage
 *
 * Implements cymatic nodal surface calculation, bottle beam interference,
 * and holographic voxel storage using acoustic standing wave patterns.
 *
 * Core equation for nodal surface:
 *   N(x, y, z) = Σᵢ Aᵢ · sin(kᵢ · r + φᵢ) = 0
 *
 * Bottle beam intensity:
 *   I(r) = |Σᵢ Aᵢ · exp(i(k·r + φᵢ))|²
 */

import type {
  Vector3D,
  AcousticSource,
  ResonanceResult,
  FluxResult,
  Voxel,
  HoloCubeConfig,
} from './types';
import { CONSTANTS } from './harmonic-scaling';

// ═══════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════

export const ACOUSTIC_CONSTANTS = {
  /** Speed of sound in air (m/s) */
  SPEED_OF_SOUND: 343.0,

  /** Reference wavelength (m) */
  WAVELENGTH_REF: 0.01,

  /** Default cavity length (m) */
  DEFAULT_L: 1.0,

  /** Default damping coefficient */
  DEFAULT_GAMMA: 0.01,

  /** Pi constant */
  PI: Math.PI,

  /** Two Pi */
  TWO_PI: 2 * Math.PI,

  /** Default voxel resolution */
  DEFAULT_RESOLUTION: 32,

  /** Tolerance for nodal detection */
  NODAL_TOLERANCE: 0.01,
} as const;

// ═══════════════════════════════════════════════════════════════
// Vector Operations
// ═══════════════════════════════════════════════════════════════

/**
 * Compute Euclidean distance between two 3D points.
 */
function distance3D(a: Vector3D, b: Vector3D): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}


// ═══════════════════════════════════════════════════════════════
// Nodal Surface Calculation
// ═══════════════════════════════════════════════════════════════

/**
 * Calculate nodal surface value at a point.
 *
 * For a standing wave, nodal surfaces occur where:
 *   N(r) = Σᵢ Aᵢ · sin(k · |r - rᵢ| + φᵢ) = 0
 *
 * This function returns the sum, not a boolean.
 * Values near zero indicate nodal points.
 *
 * @param position - 3D position to evaluate
 * @param sources - Array of acoustic sources
 * @param wavelength - Wavelength of the acoustic wave
 * @returns Sum of contributions (0 = nodal point)
 */
export function nodalSurface(
  position: Vector3D,
  sources: AcousticSource[],
  wavelength: number = ACOUSTIC_CONSTANTS.WAVELENGTH_REF
): number {
  if (wavelength <= 0) throw new RangeError('wavelength must be > 0');
  if (sources.length === 0) return 0;

  const k = ACOUSTIC_CONSTANTS.TWO_PI / wavelength;
  let sum = 0;

  for (const src of sources) {
    const r = distance3D(position, src.pos);
    const A = src.amplitude ?? 1.0;
    const phi = src.phase;
    sum += A * Math.sin(k * r + phi);
  }

  return sum;
}

/**
 * Check if a point is on a nodal surface.
 *
 * @param position - 3D position to check
 * @param sources - Array of acoustic sources
 * @param wavelength - Wavelength
 * @param tolerance - Tolerance for considering a point nodal
 * @returns True if point is within tolerance of nodal surface
 */
export function isNodalPoint(
  position: Vector3D,
  sources: AcousticSource[],
  wavelength: number = ACOUSTIC_CONSTANTS.WAVELENGTH_REF,
  tolerance: number = ACOUSTIC_CONSTANTS.NODAL_TOLERANCE
): boolean {
  const value = Math.abs(nodalSurface(position, sources, wavelength));
  return value < tolerance;
}

// ═══════════════════════════════════════════════════════════════
// Cymatic Resonance
// ═══════════════════════════════════════════════════════════════

/**
 * Calculate resonant frequencies for a cavity.
 *
 * f_n = n · c / (2L)
 *
 * @param L - Cavity length
 * @param c - Speed of sound
 * @param maxMode - Maximum mode number to compute
 * @returns Array of resonant frequencies
 */
export function resonantFrequencies(
  L: number = ACOUSTIC_CONSTANTS.DEFAULT_L,
  c: number = ACOUSTIC_CONSTANTS.SPEED_OF_SOUND,
  maxMode: number = 10
): number[] {
  if (L <= 0) throw new RangeError('L must be > 0');
  if (c <= 0) throw new RangeError('c must be > 0');

  const freqs: number[] = [];
  for (let n = 1; n <= maxMode; n++) {
    freqs.push((n * c) / (2 * L));
  }
  return freqs;
}

/**
 * Check if a frequency is resonant with a cavity.
 *
 * A frequency f is resonant if f ≈ n · c / (2L) for some integer n.
 *
 * @param frequency - Frequency to check (Hz)
 * @param L - Cavity length (m)
 * @param c - Speed of sound (m/s)
 * @param tolerance - Relative tolerance
 * @returns Resonance result with details
 */
export function checkCymaticResonance(
  frequency: number,
  L: number = ACOUSTIC_CONSTANTS.DEFAULT_L,
  c: number = ACOUSTIC_CONSTANTS.SPEED_OF_SOUND,
  tolerance: number = CONSTANTS.DEFAULT_TOLERANCE
): ResonanceResult {
  if (frequency <= 0) throw new RangeError('frequency must be > 0');
  if (L <= 0) throw new RangeError('L must be > 0');

  // Fundamental frequency
  const f1 = c / (2 * L);

  // Mode number (fractional)
  const n_frac = frequency / f1;

  // Nearest integer mode
  const n_int = Math.round(n_frac);

  // Deviation from resonance
  const deviation = Math.abs(n_frac - n_int);

  // Relative to mode spacing
  const nodalValue = deviation;

  return {
    isResonant: deviation < tolerance,
    nodalValue,
    tolerance,
    frequency,
  };
}

/**
 * Compute damped resonance amplitude.
 *
 * A(f) = A₀ / √((f² - f_r²)² + (γf)²)
 *
 * @param frequency - Driving frequency
 * @param resonantFreq - Resonant frequency
 * @param gamma - Damping coefficient
 * @param A0 - Peak amplitude
 * @returns Amplitude at given frequency
 */
export function dampedResonanceAmplitude(
  frequency: number,
  resonantFreq: number,
  gamma: number = ACOUSTIC_CONSTANTS.DEFAULT_GAMMA,
  A0: number = 1.0
): number {
  const f2 = frequency * frequency;
  const fr2 = resonantFreq * resonantFreq;
  const denominator = Math.sqrt(
    (f2 - fr2) * (f2 - fr2) + gamma * gamma * f2
  );

  if (denominator === 0) return A0;
  return A0 / denominator;
}

// ═══════════════════════════════════════════════════════════════
// Bottle Beam Interference
// ═══════════════════════════════════════════════════════════════

/**
 * Calculate bottle beam intensity at a point.
 *
 * Bottle beams create 3D acoustic traps using interference.
 *
 * I(r) = |Σᵢ Aᵢ · exp(i(k · |r - rᵢ| + φᵢ))|²
 *
 * @param position - 3D position to evaluate
 * @param sources - Array of acoustic sources
 * @param wavelength - Wavelength
 * @returns Intensity (|amplitude|²)
 */
export function bottleBeamIntensity(
  position: Vector3D,
  sources: AcousticSource[],
  wavelength: number = ACOUSTIC_CONSTANTS.WAVELENGTH_REF
): number {
  if (wavelength <= 0) throw new RangeError('wavelength must be > 0');
  if (sources.length === 0) return 0;

  const k = ACOUSTIC_CONSTANTS.TWO_PI / wavelength;

  // Sum complex amplitudes
  let realSum = 0;
  let imagSum = 0;

  for (const src of sources) {
    const r = distance3D(position, src.pos);
    const A = src.amplitude ?? 1.0;
    const phase = k * r + src.phase;

    realSum += A * Math.cos(phase);
    imagSum += A * Math.sin(phase);
  }

  // Intensity = |amplitude|²
  return realSum * realSum + imagSum * imagSum;
}

/**
 * Find intensity minimum (trap center) using gradient descent.
 *
 * @param sources - Acoustic sources
 * @param wavelength - Wavelength
 * @param initialGuess - Starting position
 * @param maxIter - Maximum iterations
 * @param stepSize - Gradient descent step size
 * @returns Position of local minimum
 */
export function findTrapCenter(
  sources: AcousticSource[],
  wavelength: number = ACOUSTIC_CONSTANTS.WAVELENGTH_REF,
  initialGuess: Vector3D = [0, 0, 0],
  maxIter: number = 100,
  stepSize: number = 0.001
): Vector3D {
  const pos: Vector3D = [...initialGuess];
  const delta = wavelength / 100;

  for (let iter = 0; iter < maxIter; iter++) {
    // Compute numerical gradient
    const I0 = bottleBeamIntensity(pos, sources, wavelength);

    const gradX =
      (bottleBeamIntensity([pos[0] + delta, pos[1], pos[2]], sources, wavelength) - I0) / delta;
    const gradY =
      (bottleBeamIntensity([pos[0], pos[1] + delta, pos[2]], sources, wavelength) - I0) / delta;
    const gradZ =
      (bottleBeamIntensity([pos[0], pos[1], pos[2] + delta], sources, wavelength) - I0) / delta;

    // Gradient descent step
    pos[0] -= stepSize * gradX;
    pos[1] -= stepSize * gradY;
    pos[2] -= stepSize * gradZ;

    // Check convergence
    const gradNorm = Math.sqrt(gradX * gradX + gradY * gradY + gradZ * gradZ);
    if (gradNorm < 1e-6) break;
  }

  return pos;
}

// ═══════════════════════════════════════════════════════════════
// Flux Redistribution
// ═══════════════════════════════════════════════════════════════

/**
 * Compute flux redistribution to corners.
 *
 * When flux is canceled at center, energy redistributes to corners.
 * This models the "corners get the flux" principle.
 *
 * @param centerIntensity - Intensity at center
 * @param cornerPositions - 4 corner positions
 * @param sources - Acoustic sources
 * @param wavelength - Wavelength
 * @returns Flux result with corner energies
 */
export function fluxRedistribution(
  centerIntensity: number,
  cornerPositions: [Vector3D, Vector3D, Vector3D, Vector3D],
  sources: AcousticSource[],
  wavelength: number = ACOUSTIC_CONSTANTS.WAVELENGTH_REF
): FluxResult {
  // Calculate corner intensities
  const corners: [number, number, number, number] = [
    bottleBeamIntensity(cornerPositions[0], sources, wavelength),
    bottleBeamIntensity(cornerPositions[1], sources, wavelength),
    bottleBeamIntensity(cornerPositions[2], sources, wavelength),
    bottleBeamIntensity(cornerPositions[3], sources, wavelength),
  ];

  const totalCornerEnergy = corners.reduce((a, b) => a + b, 0);

  return {
    canceled: centerIntensity,
    corners,
    totalEnergy: centerIntensity + totalCornerEnergy,
  };
}

/**
 * Calculate energy conservation factor.
 *
 * Verifies that total energy is conserved during redistribution.
 *
 * @param initialEnergy - Energy before redistribution
 * @param fluxResult - Result from fluxRedistribution
 * @returns Conservation factor (1.0 = perfect conservation)
 */
export function energyConservation(
  initialEnergy: number,
  fluxResult: FluxResult
): number {
  if (initialEnergy === 0) return 1.0;
  return fluxResult.totalEnergy / initialEnergy;
}

// ═══════════════════════════════════════════════════════════════
// Cymatic Voxel Storage
// ═══════════════════════════════════════════════════════════════

/**
 * Create a holographic voxel cube.
 *
 * Generates a 3D grid of voxels with intensity and phase
 * calculated from acoustic source interference.
 *
 * @param config - HoloCube configuration
 * @returns 3D array of voxels
 */
export function createHoloCube(config: HoloCubeConfig): Voxel[][][] {
  const { resolution, wavelength, sources } = config;
  const timestamp = Date.now();
  const k = ACOUSTIC_CONSTANTS.TWO_PI / wavelength;

  // Normalize to [-1, 1] cube
  const step = 2 / resolution;

  const cube: Voxel[][][] = [];

  for (let xi = 0; xi < resolution; xi++) {
    const layer: Voxel[][] = [];
    const x = -1 + (xi + 0.5) * step;

    for (let yi = 0; yi < resolution; yi++) {
      const row: Voxel[] = [];
      const y = -1 + (yi + 0.5) * step;

      for (let zi = 0; zi < resolution; zi++) {
        const z = -1 + (zi + 0.5) * step;
        const position: Vector3D = [x, y, z];

        // Calculate complex amplitude
        let realSum = 0;
        let imagSum = 0;

        for (const src of sources) {
          const r = distance3D(position, src.pos);
          const A = src.amplitude ?? 1.0;
          const phase = k * r + src.phase;
          realSum += A * Math.cos(phase);
          imagSum += A * Math.sin(phase);
        }

        const intensity = realSum * realSum + imagSum * imagSum;
        const phase = Math.atan2(imagSum, realSum);

        row.push({
          position,
          intensity,
          phase,
          timestamp,
        });
      }
      layer.push(row);
    }
    cube.push(layer);
  }

  return cube;
}

/**
 * Read voxel value at integer coordinates.
 *
 * @param cube - Holographic cube
 * @param x - X index
 * @param y - Y index
 * @param z - Z index
 * @returns Voxel or null if out of bounds
 */
export function readVoxel(
  cube: Voxel[][][],
  x: number,
  y: number,
  z: number
): Voxel | null {
  if (
    x < 0 || x >= cube.length ||
    y < 0 || y >= (cube[0]?.length ?? 0) ||
    z < 0 || z >= (cube[0]?.[0]?.length ?? 0)
  ) {
    return null;
  }
  return cube[x][y][z];
}

/**
 * Sample holographic cube at continuous position.
 *
 * Uses trilinear interpolation.
 *
 * @param cube - Holographic cube
 * @param position - Position in [-1, 1]³
 * @returns Interpolated intensity
 */
export function sampleCube(
  cube: Voxel[][][],
  position: Vector3D
): number {
  const res = cube.length;
  if (res === 0) return 0;

  // Convert position to voxel coordinates
  const step = 2 / res;
  const fx = (position[0] + 1) / step - 0.5;
  const fy = (position[1] + 1) / step - 0.5;
  const fz = (position[2] + 1) / step - 0.5;

  // Integer coordinates
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const z0 = Math.floor(fz);

  // Fractional parts
  const dx = fx - x0;
  const dy = fy - y0;
  const dz = fz - z0;

  // Trilinear interpolation
  let sum = 0;

  for (let di = 0; di <= 1; di++) {
    for (let dj = 0; dj <= 1; dj++) {
      for (let dk = 0; dk <= 1; dk++) {
        const v = readVoxel(cube, x0 + di, y0 + dj, z0 + dk);
        if (!v) continue;

        const wx = di === 0 ? 1 - dx : dx;
        const wy = dj === 0 ? 1 - dy : dy;
        const wz = dk === 0 ? 1 - dz : dz;

        sum += wx * wy * wz * v.intensity;
      }
    }
  }

  return sum;
}

/**
 * Encode data into holographic cube using phase modulation.
 *
 * Each bit is encoded by adjusting source phases to create
 * constructive (1) or destructive (0) interference at specific
 * voxel locations.
 *
 * @param data - Binary data to encode
 * @param baseConfig - Base HoloCube configuration
 * @returns Encoded holographic cube
 */
export function encodeHolographic(
  data: Uint8Array,
  baseConfig: Omit<HoloCubeConfig, 'sources'>
): Voxel[][][] {
  const { resolution, wavelength } = baseConfig;
  const totalVoxels = resolution * resolution * resolution;
  const totalBits = data.length * 8;

  if (totalBits > totalVoxels) {
    throw new RangeError(`Data too large: ${totalBits} bits > ${totalVoxels} voxels`);
  }

  // Create sources for encoding
  const sources: AcousticSource[] = [
    { pos: [1, 0, 0], phase: 0, amplitude: 1 },
    { pos: [-1, 0, 0], phase: 0, amplitude: 1 },
    { pos: [0, 1, 0], phase: 0, amplitude: 1 },
    { pos: [0, -1, 0], phase: 0, amplitude: 1 },
    { pos: [0, 0, 1], phase: 0, amplitude: 1 },
    { pos: [0, 0, -1], phase: 0, amplitude: 1 },
  ];

  const cube = createHoloCube({ resolution, wavelength, sources });

  // Modulate intensity based on data bits
  let bitIndex = 0;
  for (let x = 0; x < resolution && bitIndex < totalBits; x++) {
    for (let y = 0; y < resolution && bitIndex < totalBits; y++) {
      for (let z = 0; z < resolution && bitIndex < totalBits; z++) {
        const byteIndex = Math.floor(bitIndex / 8);
        const bitOffset = bitIndex % 8;
        const bit = (data[byteIndex] >> (7 - bitOffset)) & 1;

        // Modulate phase based on bit value
        if (bit === 0) {
          cube[x][y][z].intensity *= 0.1; // Suppress for 0
        }

        bitIndex++;
      }
    }
  }

  return cube;
}

/**
 * Decode data from holographic cube.
 *
 * @param cube - Encoded holographic cube
 * @param threshold - Intensity threshold for bit detection
 * @param numBytes - Number of bytes to decode
 * @returns Decoded data
 */
export function decodeHolographic(
  cube: Voxel[][][],
  numBytes: number,
  threshold: number = 0.5
): Uint8Array {
  const resolution = cube.length;
  const data = new Uint8Array(numBytes);

  // Find max intensity for normalization
  let maxIntensity = 0;
  for (let x = 0; x < resolution; x++) {
    for (let y = 0; y < resolution; y++) {
      for (let z = 0; z < resolution; z++) {
        maxIntensity = Math.max(maxIntensity, cube[x][y][z].intensity);
      }
    }
  }

  if (maxIntensity === 0) return data;

  // Decode bits
  let bitIndex = 0;
  const totalBits = numBytes * 8;

  for (let x = 0; x < resolution && bitIndex < totalBits; x++) {
    for (let y = 0; y < resolution && bitIndex < totalBits; y++) {
      for (let z = 0; z < resolution && bitIndex < totalBits; z++) {
        const normalized = cube[x][y][z].intensity / maxIntensity;
        const bit = normalized > threshold ? 1 : 0;

        const byteIndex = Math.floor(bitIndex / 8);
        const bitOffset = bitIndex % 8;
        data[byteIndex] |= bit << (7 - bitOffset);

        bitIndex++;
      }
    }
  }

  return data;
}

// ═══════════════════════════════════════════════════════════════
// Time-Synchronized Acoustics
// ═══════════════════════════════════════════════════════════════

/**
 * Create time-varying acoustic field.
 *
 * The field evolves according to the wave equation with
 * sources oscillating at given frequencies.
 *
 * @param position - Position to evaluate
 * @param sources - Acoustic sources with frequency info
 * @param time - Time in seconds
 * @param wavelength - Reference wavelength
 * @returns Complex amplitude at position and time
 */
export function timeVaryingField(
  position: Vector3D,
  sources: (AcousticSource & { frequency: number })[],
  time: number,
  wavelength: number = ACOUSTIC_CONSTANTS.WAVELENGTH_REF
): { real: number; imag: number; intensity: number } {
  let realSum = 0;
  let imagSum = 0;

  for (const src of sources) {
    const r = distance3D(position, src.pos);
    const k = ACOUSTIC_CONSTANTS.TWO_PI / wavelength;
    const omega = ACOUSTIC_CONSTANTS.TWO_PI * src.frequency;
    const A = src.amplitude ?? 1.0;

    // Phase includes spatial and temporal components
    const phase = k * r - omega * time + src.phase;

    realSum += A * Math.cos(phase);
    imagSum += A * Math.sin(phase);
  }

  return {
    real: realSum,
    imag: imagSum,
    intensity: realSum * realSum + imagSum * imagSum,
  };
}

/**
 * Calculate breathing-synchronized acoustic intensity.
 *
 * Modulates acoustic field with breathing rhythm.
 *
 * @param position - Position to evaluate
 * @param sources - Acoustic sources
 * @param breathPhase - Current breath phase [0, 2π]
 * @param wavelength - Wavelength
 * @returns Modulated intensity
 */
export function breathSyncedIntensity(
  position: Vector3D,
  sources: AcousticSource[],
  breathPhase: number,
  wavelength: number = ACOUSTIC_CONSTANTS.WAVELENGTH_REF
): number {
  const baseIntensity = bottleBeamIntensity(position, sources, wavelength);

  // Breathing modulation factor
  const breathMod = 1.0 + 0.1 * Math.sin(breathPhase);

  return baseIntensity * breathMod;
}
