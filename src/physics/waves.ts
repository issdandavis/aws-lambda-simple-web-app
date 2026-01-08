/**
 * Wave Simulation Module
 * Implements electromagnetic waves, sound waves, and quantum wave functions
 */

import { PhysicalConstants as PC } from './constants';

export interface WaveParameters {
  amplitude: number;       // A
  frequency: number;       // f (Hz)
  wavelength: number;      // λ (m)
  phase: number;           // φ (radians)
  velocity: number;        // v (m/s)
}

export interface WaveResult {
  displacement: number[];
  positions: number[];
  time: number;
  energy: number;
  intensity: number;
}

export interface InterferenceResult {
  resultantAmplitude: number[];
  positions: number[];
  constructivePoints: number[];
  destructivePoints: number[];
  fringeSpacing: number;
}

export interface DiffractionResult {
  intensity: number[];
  angles: number[];
  centralMaxWidth: number;
  maxima: number[];
  minima: number[];
}

export interface DopplerResult {
  observedFrequency: number;
  wavelengthShift: number;
  relativeVelocity: number;
  redshift: number;  // z parameter (positive for receding)
}

export interface StandingWaveResult {
  modes: number[];
  frequencies: number[];
  nodePositions: number[][];
  antinodePositions: number[][];
}

export interface BlackbodyResult {
  peakWavelength: number;          // Wien's law (m)
  peakFrequency: number;           // Hz
  totalPower: number;              // Stefan-Boltzmann (W/m²)
  spectralRadiance: number[];      // B(λ,T) array
  wavelengths: number[];           // corresponding wavelengths
}

/**
 * Wave Physics Calculator
 */
export class WaveSimulation {

  /**
   * Calculate wave parameters from given inputs
   */
  static calculateWaveParameters(
    amplitude: number,
    frequency: number,
    mediumVelocity: number = PC.c,
    phase: number = 0
  ): WaveParameters {
    const wavelength = mediumVelocity / frequency;

    return {
      amplitude,
      frequency,
      wavelength,
      phase,
      velocity: mediumVelocity,
    };
  }

  /**
   * Generate sinusoidal wave displacement
   * y(x,t) = A * sin(kx - ωt + φ)
   */
  static sinusoidalWave(
    params: WaveParameters,
    xMin: number,
    xMax: number,
    time: number,
    numPoints: number = 100
  ): WaveResult {
    const k = 2 * Math.PI / params.wavelength;  // Wave number
    const omega = 2 * Math.PI * params.frequency;  // Angular frequency

    const positions: number[] = [];
    const displacement: number[] = [];
    const dx = (xMax - xMin) / (numPoints - 1);

    for (let i = 0; i < numPoints; i++) {
      const x = xMin + i * dx;
      const y = params.amplitude * Math.sin(k * x - omega * time + params.phase);
      positions.push(x);
      displacement.push(y);
    }

    // Wave energy per unit length: E = 0.5 * μ * ω² * A²
    // Using simplified form for energy density
    const energy = 0.5 * Math.pow(params.amplitude * omega, 2);

    // Intensity (power per unit area) for EM waves: I = 0.5 * ε₀ * c * E₀²
    const intensity = 0.5 * PC.epsilon0 * PC.c * Math.pow(params.amplitude, 2);

    return { displacement, positions, time, energy, intensity };
  }

  /**
   * Two-source interference pattern
   */
  static twoSourceInterference(
    wavelength: number,
    sourceSpacing: number,     // d
    screenDistance: number,    // L
    screenWidth: number,
    numPoints: number = 200
  ): InterferenceResult {
    const positions: number[] = [];
    const resultantAmplitude: number[] = [];
    const constructivePoints: number[] = [];
    const destructivePoints: number[] = [];

    const dx = screenWidth / (numPoints - 1);
    const k = 2 * Math.PI / wavelength;

    for (let i = 0; i < numPoints; i++) {
      const y = -screenWidth / 2 + i * dx;
      positions.push(y);

      // Path difference: Δr = d * sin(θ) ≈ d * y / L for small angles
      const pathDiff = sourceSpacing * y / screenDistance;

      // Phase difference
      const phaseDiff = k * pathDiff;

      // Resultant amplitude: A_total = 2A * cos(φ/2)
      const amplitude = 2 * Math.cos(phaseDiff / 2);
      resultantAmplitude.push(amplitude * amplitude); // Intensity proportional to A²
    }

    // Find constructive and destructive interference points
    // Constructive: d*sin(θ) = mλ
    // Destructive: d*sin(θ) = (m + 0.5)λ

    for (let m = -10; m <= 10; m++) {
      const sinTheta = m * wavelength / sourceSpacing;
      if (Math.abs(sinTheta) <= 1) {
        const y = screenDistance * Math.tan(Math.asin(sinTheta));
        if (Math.abs(y) <= screenWidth / 2) {
          constructivePoints.push(y);
        }
      }

      const sinThetaDestructive = (m + 0.5) * wavelength / sourceSpacing;
      if (Math.abs(sinThetaDestructive) <= 1) {
        const y = screenDistance * Math.tan(Math.asin(sinThetaDestructive));
        if (Math.abs(y) <= screenWidth / 2) {
          destructivePoints.push(y);
        }
      }
    }

    // Fringe spacing: Δy = λL/d
    const fringeSpacing = wavelength * screenDistance / sourceSpacing;

    return {
      resultantAmplitude,
      positions,
      constructivePoints: constructivePoints.sort((a, b) => a - b),
      destructivePoints: destructivePoints.sort((a, b) => a - b),
      fringeSpacing,
    };
  }

  /**
   * Single slit diffraction pattern (Fraunhofer)
   * I(θ) = I₀ * [sin(β)/β]² where β = (πa/λ)sin(θ)
   */
  static singleSlitDiffraction(
    wavelength: number,
    slitWidth: number,         // a
    screenDistance: number,
    maxAngle: number = Math.PI / 6,  // radians
    numPoints: number = 200
  ): DiffractionResult {
    const angles: number[] = [];
    const intensity: number[] = [];
    const maxima: number[] = [];
    const minima: number[] = [];

    const dTheta = 2 * maxAngle / (numPoints - 1);

    for (let i = 0; i < numPoints; i++) {
      const theta = -maxAngle + i * dTheta;
      angles.push(theta);

      const beta = Math.PI * slitWidth * Math.sin(theta) / wavelength;

      // Handle β = 0 case (central maximum)
      let I: number;
      if (Math.abs(beta) < 1e-10) {
        I = 1.0;
      } else {
        I = Math.pow(Math.sin(beta) / beta, 2);
      }

      intensity.push(I);
    }

    // Calculate diffraction minima: a*sin(θ) = mλ
    for (let m = 1; m <= 10; m++) {
      const sinTheta = m * wavelength / slitWidth;
      if (Math.abs(sinTheta) <= 1) {
        minima.push(Math.asin(sinTheta));
        minima.push(-Math.asin(sinTheta));
      }
    }

    // Calculate secondary maxima (approximate: between minima)
    maxima.push(0); // Central maximum
    for (let m = 1; m <= 9; m++) {
      const sinTheta = (m + 0.5) * wavelength / slitWidth;
      if (Math.abs(sinTheta) <= 1) {
        maxima.push(Math.asin(sinTheta));
        maxima.push(-Math.asin(sinTheta));
      }
    }

    // Central maximum width: Δθ = 2λ/a
    const centralMaxWidth = 2 * wavelength / slitWidth;

    return {
      intensity,
      angles,
      centralMaxWidth,
      maxima: maxima.sort((a, b) => a - b),
      minima: minima.sort((a, b) => a - b),
    };
  }

  /**
   * Diffraction grating
   * Principal maxima at: d*sin(θ) = mλ
   */
  static diffractionGrating(
    wavelength: number,
    gratingSpacing: number,    // d (line spacing)
    numSlits: number,
    maxAngle: number = Math.PI / 3,
    numPoints: number = 500
  ): { intensity: number[]; angles: number[]; principalMaxima: number[] } {
    const angles: number[] = [];
    const intensity: number[] = [];
    const principalMaxima: number[] = [];

    const dTheta = 2 * maxAngle / (numPoints - 1);
    const k = 2 * Math.PI / wavelength;

    for (let i = 0; i < numPoints; i++) {
      const theta = -maxAngle + i * dTheta;
      angles.push(theta);

      const delta = k * gratingSpacing * Math.sin(theta);

      // N-slit interference: I = I₀ * [sin(Nδ/2)/sin(δ/2)]²
      let I: number;
      if (Math.abs(Math.sin(delta / 2)) < 1e-10) {
        I = numSlits * numSlits;
      } else {
        I = Math.pow(Math.sin(numSlits * delta / 2) / Math.sin(delta / 2), 2);
      }

      intensity.push(I / (numSlits * numSlits)); // Normalize
    }

    // Principal maxima
    for (let m = -10; m <= 10; m++) {
      const sinTheta = m * wavelength / gratingSpacing;
      if (Math.abs(sinTheta) <= 1) {
        principalMaxima.push(Math.asin(sinTheta));
      }
    }

    return {
      intensity,
      angles,
      principalMaxima: principalMaxima.sort((a, b) => a - b),
    };
  }

  /**
   * Doppler effect for sound waves
   */
  static dopplerSound(
    sourceFrequency: number,
    soundSpeed: number,        // v (m/s)
    sourceVelocity: number,    // v_s (positive = moving away)
    observerVelocity: number   // v_o (positive = moving toward source)
  ): DopplerResult {
    // f' = f * (v + v_o) / (v + v_s)
    const observedFrequency = sourceFrequency * (soundSpeed + observerVelocity) / (soundSpeed + sourceVelocity);

    const wavelengthOriginal = soundSpeed / sourceFrequency;
    const wavelengthObserved = soundSpeed / observedFrequency;
    const wavelengthShift = wavelengthObserved - wavelengthOriginal;

    const relativeVelocity = sourceVelocity - observerVelocity;

    // Redshift parameter (adapted for sound)
    const redshift = (observedFrequency - sourceFrequency) / sourceFrequency;

    return {
      observedFrequency,
      wavelengthShift,
      relativeVelocity,
      redshift: -redshift, // Positive for receding (lower frequency)
    };
  }

  /**
   * Relativistic Doppler effect for light
   * f_observed = f_source * sqrt((1 - β)/(1 + β)) for receding source
   */
  static dopplerRelativistic(
    sourceFrequency: number,
    relativeVelocity: number   // positive = receding
  ): DopplerResult {
    const beta = relativeVelocity / PC.c;

    if (Math.abs(beta) >= 1) {
      throw new Error('Relative velocity cannot equal or exceed speed of light');
    }

    // For receding source (positive velocity)
    const dopplerFactor = Math.sqrt((1 - beta) / (1 + beta));
    const observedFrequency = sourceFrequency * dopplerFactor;

    const wavelengthSource = PC.c / sourceFrequency;
    const wavelengthObserved = PC.c / observedFrequency;
    const wavelengthShift = wavelengthObserved - wavelengthSource;

    // Cosmological redshift: z = (λ_obs - λ_emit) / λ_emit
    const redshift = wavelengthShift / wavelengthSource;

    return {
      observedFrequency,
      wavelengthShift,
      relativeVelocity,
      redshift,
    };
  }

  /**
   * Standing waves on a string
   */
  static standingWaves(
    stringLength: number,
    tension: number,           // N
    linearDensity: number,     // kg/m
    numModes: number = 5
  ): StandingWaveResult {
    // Wave velocity: v = sqrt(T/μ)
    const velocity = Math.sqrt(tension / linearDensity);

    const modes: number[] = [];
    const frequencies: number[] = [];
    const nodePositions: number[][] = [];
    const antinodePositions: number[][] = [];

    for (let n = 1; n <= numModes; n++) {
      modes.push(n);

      // f_n = n * v / (2L)
      const frequency = n * velocity / (2 * stringLength);
      frequencies.push(frequency);

      // Nodes at x = m * L / n (m = 0, 1, 2, ..., n)
      const nodes: number[] = [];
      for (let m = 0; m <= n; m++) {
        nodes.push(m * stringLength / n);
      }
      nodePositions.push(nodes);

      // Antinodes at x = (m + 0.5) * L / n (m = 0, 1, ..., n-1)
      const antinodes: number[] = [];
      for (let m = 0; m < n; m++) {
        antinodes.push((m + 0.5) * stringLength / n);
      }
      antinodePositions.push(antinodes);
    }

    return {
      modes,
      frequencies,
      nodePositions,
      antinodePositions,
    };
  }

  /**
   * Blackbody radiation (Planck's law)
   */
  static blackbodyRadiation(
    temperature: number,       // Kelvin
    wavelengthMin: number = 1e-9,   // m
    wavelengthMax: number = 10e-6,  // m
    numPoints: number = 200
  ): BlackbodyResult {
    // Wien's displacement law: λ_max = b/T
    const wienConstant = 2.897771955e-3;  // m·K
    const peakWavelength = wienConstant / temperature;
    const peakFrequency = PC.c / peakWavelength;

    // Stefan-Boltzmann law: P = σT⁴
    const totalPower = PC.sigma * Math.pow(temperature, 4);

    const wavelengths: number[] = [];
    const spectralRadiance: number[] = [];

    // Use logarithmic spacing for better coverage
    const logMin = Math.log10(wavelengthMin);
    const logMax = Math.log10(wavelengthMax);
    const dLog = (logMax - logMin) / (numPoints - 1);

    for (let i = 0; i < numPoints; i++) {
      const lambda = Math.pow(10, logMin + i * dLog);
      wavelengths.push(lambda);

      // Planck's law: B(λ,T) = (2hc²/λ⁵) * 1/(exp(hc/λkT) - 1)
      const exponent = PC.h * PC.c / (lambda * PC.kB * temperature);

      let B: number;
      if (exponent > 700) {
        // Avoid overflow
        B = 0;
      } else {
        B = (2 * PC.h * PC.c * PC.c / Math.pow(lambda, 5)) *
            (1 / (Math.exp(exponent) - 1));
      }

      spectralRadiance.push(B);
    }

    return {
      peakWavelength,
      peakFrequency,
      totalPower,
      spectralRadiance,
      wavelengths,
    };
  }

  /**
   * Wave packet / group velocity
   * v_g = dω/dk
   */
  static wavePacket(
    centralFrequency: number,
    frequencySpread: number,   // Δf
    position: number[],
    time: number
  ): { amplitude: number[]; groupVelocity: number; phaseVelocity: number } {
    const omega0 = 2 * Math.PI * centralFrequency;
    const k0 = omega0 / PC.c;  // For light
    const deltaOmega = 2 * Math.PI * frequencySpread;

    const amplitude: number[] = [];

    for (const x of position) {
      // Gaussian wave packet
      const envelope = Math.exp(-Math.pow(x - PC.c * time, 2) / (2 * Math.pow(PC.c / deltaOmega, 2)));
      const carrier = Math.cos(k0 * x - omega0 * time);
      amplitude.push(envelope * carrier);
    }

    // For light in vacuum, group velocity = phase velocity = c
    return {
      amplitude,
      groupVelocity: PC.c,
      phaseVelocity: PC.c,
    };
  }

  /**
   * Electromagnetic wave energy and Poynting vector
   */
  static emWaveProperties(
    electricFieldAmplitude: number,  // E₀ (V/m)
    frequency: number                // Hz
  ): {
    magneticFieldAmplitude: number;
    intensity: number;
    energyDensity: number;
    momentum: number;
    poyntingVector: number;
  } {
    // B₀ = E₀/c
    const magneticFieldAmplitude = electricFieldAmplitude / PC.c;

    // Intensity: I = (1/2) * ε₀ * c * E₀²
    const intensity = 0.5 * PC.epsilon0 * PC.c * electricFieldAmplitude * electricFieldAmplitude;

    // Energy density: u = ε₀ * E₀² / 2 + B₀² / (2μ₀) = ε₀ * E₀²
    const energyDensity = PC.epsilon0 * electricFieldAmplitude * electricFieldAmplitude;

    // Radiation pressure / momentum flux: p = I/c = u
    const momentum = intensity / PC.c;

    // Poynting vector magnitude: S = E × B / μ₀ = E₀ * B₀ / μ₀
    const poyntingVector = electricFieldAmplitude * magneticFieldAmplitude / PC.mu0;

    return {
      magneticFieldAmplitude,
      intensity,
      energyDensity,
      momentum,
      poyntingVector,
    };
  }

  /**
   * Snell's law and refraction
   */
  static refraction(
    incidentAngle: number,     // radians
    n1: number,                // refractive index of medium 1
    n2: number                 // refractive index of medium 2
  ): {
    refractedAngle: number | null;
    reflectedAngle: number;
    criticalAngle: number | null;
    isTotalInternalReflection: boolean;
    reflectanceS: number;
    reflectanceP: number;
  } {
    const sinTheta1 = Math.sin(incidentAngle);
    const sinTheta2 = n1 * sinTheta1 / n2;

    // Check for total internal reflection
    const criticalAngle = n1 > n2 ? Math.asin(n2 / n1) : null;
    const isTotalInternalReflection = Math.abs(sinTheta2) > 1;

    let refractedAngle: number | null;
    let reflectanceS: number;
    let reflectanceP: number;

    if (isTotalInternalReflection) {
      refractedAngle = null;
      reflectanceS = 1;
      reflectanceP = 1;
    } else {
      refractedAngle = Math.asin(sinTheta2);
      const cosTheta1 = Math.cos(incidentAngle);
      const cosTheta2 = Math.cos(refractedAngle);

      // Fresnel equations
      const rs = (n1 * cosTheta1 - n2 * cosTheta2) / (n1 * cosTheta1 + n2 * cosTheta2);
      const rp = (n2 * cosTheta1 - n1 * cosTheta2) / (n2 * cosTheta1 + n1 * cosTheta2);

      reflectanceS = rs * rs;
      reflectanceP = rp * rp;
    }

    return {
      refractedAngle,
      reflectedAngle: incidentAngle,
      criticalAngle,
      isTotalInternalReflection,
      reflectanceS,
      reflectanceP,
    };
  }
}
