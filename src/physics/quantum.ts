/**
 * Quantum Mechanics Simulation Module
 * Implements real quantum mechanical calculations
 */

import { PhysicalConstants as PC } from './constants';

export interface QuantumState {
  n: number;           // Principal quantum number
  l: number;           // Orbital angular momentum quantum number
  m: number;           // Magnetic quantum number
  s: number;           // Spin quantum number (+1/2 or -1/2)
}

export interface WavefunctionResult {
  realPart: number[];
  imaginaryPart: number[];
  probability: number[];
  positions: number[];
  energy: number;
  normalization: number;
}

export interface PhotonProperties {
  energy: number;         // Joules
  frequency: number;      // Hz
  wavelength: number;     // meters
  momentum: number;       // kg·m/s
  angularMomentum: number; // J·s
}

export interface UncertaintyResult {
  deltaX: number;         // Position uncertainty (m)
  deltaP: number;         // Momentum uncertainty (kg·m/s)
  product: number;        // ΔxΔp product
  minimumProduct: number; // ħ/2
  isValid: boolean;       // Whether uncertainty principle is satisfied
}

export interface TunnelingResult {
  transmissionCoefficient: number;
  reflectionCoefficient: number;
  penetrationDepth: number;  // meters
  decayConstant: number;     // 1/m
}

export interface HarmonicOscillatorResult {
  energy: number;
  zeroPointEnergy: number;
  classicalTurningPoint: number;
  wavefunctionAtOrigin: number;
  expectationX: number;
  expectationX2: number;
}

/**
 * Quantum Mechanics Calculator
 */
export class QuantumMechanics {

  /**
   * Calculate photon properties from wavelength
   */
  static calculatePhotonProperties(wavelengthNm: number): PhotonProperties {
    const wavelength = wavelengthNm * 1e-9; // Convert nm to m
    const frequency = PC.c / wavelength;
    const energy = PC.h * frequency;
    const momentum = PC.h / wavelength;
    const angularMomentum = PC.hbar; // Spin-1 particle

    return {
      energy,
      frequency,
      wavelength,
      momentum,
      angularMomentum,
    };
  }

  /**
   * Calculate hydrogen atom energy levels using the Bohr model
   * E_n = -13.6 eV / n²
   */
  static hydrogenEnergyLevel(n: number): number {
    if (n < 1 || !Number.isInteger(n)) {
      throw new Error('Principal quantum number must be a positive integer');
    }

    const E1 = -(PC.me * Math.pow(PC.e, 4)) /
               (8 * Math.pow(PC.epsilon0, 2) * Math.pow(PC.h, 2));

    return E1 / (n * n);
  }

  /**
   * Calculate energy transition for hydrogen atom
   * Returns photon wavelength for transition from n_initial to n_final
   */
  static hydrogenTransition(nInitial: number, nFinal: number): {
    energyDifference: number;
    wavelength: number;
    frequency: number;
    seriesName: string;
  } {
    const E_initial = this.hydrogenEnergyLevel(nInitial);
    const E_final = this.hydrogenEnergyLevel(nFinal);
    const energyDifference = Math.abs(E_final - E_initial);

    const frequency = energyDifference / PC.h;
    const wavelength = PC.c / frequency;

    let seriesName = 'Unknown';
    if (nFinal === 1) seriesName = 'Lyman';
    else if (nFinal === 2) seriesName = 'Balmer';
    else if (nFinal === 3) seriesName = 'Paschen';
    else if (nFinal === 4) seriesName = 'Brackett';
    else if (nFinal === 5) seriesName = 'Pfund';

    return { energyDifference, wavelength, frequency, seriesName };
  }

  /**
   * Calculate Heisenberg uncertainty principle
   * ΔxΔp ≥ ħ/2
   */
  static calculateUncertainty(deltaX: number): UncertaintyResult {
    const minimumProduct = PC.hbar / 2;
    const deltaP = minimumProduct / deltaX;
    const product = deltaX * deltaP;

    return {
      deltaX,
      deltaP,
      product,
      minimumProduct,
      isValid: product >= minimumProduct,
    };
  }

  /**
   * Quantum tunneling through a rectangular barrier
   * Uses WKB approximation for thick barriers
   */
  static quantumTunneling(
    particleMass: number,      // kg
    particleEnergy: number,    // J
    barrierHeight: number,     // J
    barrierWidth: number       // m
  ): TunnelingResult {
    if (particleEnergy >= barrierHeight) {
      return {
        transmissionCoefficient: 1,
        reflectionCoefficient: 0,
        penetrationDepth: Infinity,
        decayConstant: 0,
      };
    }

    // Decay constant κ = sqrt(2m(V-E)) / ħ
    const kappa = Math.sqrt(2 * particleMass * (barrierHeight - particleEnergy)) / PC.hbar;
    const penetrationDepth = 1 / kappa;

    // Transmission coefficient T ≈ e^(-2κL) for κL >> 1
    const exponent = -2 * kappa * barrierWidth;
    const transmissionCoefficient = Math.exp(exponent);
    const reflectionCoefficient = 1 - transmissionCoefficient;

    return {
      transmissionCoefficient,
      reflectionCoefficient,
      penetrationDepth,
      decayConstant: kappa,
    };
  }

  /**
   * Quantum harmonic oscillator
   * E_n = ħω(n + 1/2)
   */
  static harmonicOscillator(
    mass: number,              // kg
    angularFrequency: number,  // rad/s
    n: number                  // quantum number (0, 1, 2, ...)
  ): HarmonicOscillatorResult {
    if (n < 0 || !Number.isInteger(n)) {
      throw new Error('Quantum number must be a non-negative integer');
    }

    const zeroPointEnergy = 0.5 * PC.hbar * angularFrequency;
    const energy = PC.hbar * angularFrequency * (n + 0.5);

    // Classical turning point: x_max = sqrt(2E/(mω²))
    const classicalTurningPoint = Math.sqrt(2 * energy / (mass * angularFrequency * angularFrequency));

    // Characteristic length: a = sqrt(ħ/(mω))
    const a = Math.sqrt(PC.hbar / (mass * angularFrequency));

    // Wavefunction at origin (only non-zero for even n)
    let wavefunctionAtOrigin = 0;
    if (n % 2 === 0) {
      wavefunctionAtOrigin = Math.pow(mass * angularFrequency / (Math.PI * PC.hbar), 0.25) *
        Math.pow(-1, n/2) * this.hermitePolynomial(n, 0) / Math.sqrt(Math.pow(2, n) * this.factorial(n));
    }

    // Expectation values
    const expectationX = 0; // Symmetric potential
    const expectationX2 = (n + 0.5) * PC.hbar / (mass * angularFrequency);

    return {
      energy,
      zeroPointEnergy,
      classicalTurningPoint,
      wavefunctionAtOrigin,
      expectationX,
      expectationX2,
    };
  }

  /**
   * Calculate radial wavefunction for hydrogen atom
   * R_nl(r) using Laguerre polynomials
   */
  static hydrogenRadialWavefunction(
    n: number,
    l: number,
    rValues: number[]  // in units of Bohr radius
  ): { r: number[]; R: number[]; probability: number[] } {
    if (l >= n || l < 0) {
      throw new Error('Invalid quantum numbers: 0 ≤ l < n required');
    }

    const R: number[] = [];
    const probability: number[] = [];

    for (const r of rValues) {
      const rho = 2 * r / n;

      // Normalization factor
      const norm = Math.sqrt(
        Math.pow(2 / n, 3) * this.factorial(n - l - 1) /
        (2 * n * this.factorial(n + l))
      );

      // Radial function
      const exp_part = Math.exp(-rho / 2);
      const rho_part = Math.pow(rho, l);
      const laguerre = this.associatedLaguerre(n - l - 1, 2 * l + 1, rho);

      const R_nl = norm * exp_part * rho_part * laguerre;
      R.push(R_nl);

      // Probability density r²|R|²
      probability.push(r * r * R_nl * R_nl);
    }

    return { r: rValues, R, probability };
  }

  /**
   * Calculate de Broglie wavelength
   * λ = h/p = h/(mv) for non-relativistic
   */
  static deBroglieWavelength(mass: number, velocity: number): number {
    const momentum = mass * velocity;
    return PC.h / momentum;
  }

  /**
   * Particle in a box energy levels
   * E_n = n²π²ħ²/(2mL²)
   */
  static particleInBox(mass: number, boxLength: number, n: number): {
    energy: number;
    wavelength: number;
    nodesCount: number;
  } {
    if (n < 1 || !Number.isInteger(n)) {
      throw new Error('Quantum number must be a positive integer');
    }

    const energy = (n * n * Math.PI * Math.PI * PC.hbar * PC.hbar) / (2 * mass * boxLength * boxLength);
    const wavelength = 2 * boxLength / n;
    const nodesCount = n - 1;

    return { energy, wavelength, nodesCount };
  }

  /**
   * Spin-orbit coupling energy
   * E_so = (α²/n³) * [j(j+1) - l(l+1) - s(s+1)] / (2l(l+1/2)(l+1))
   */
  static spinOrbitCoupling(n: number, l: number, j: number): number {
    if (l === 0) return 0; // No spin-orbit coupling for s orbitals

    const s = 0.5;
    const numerator = j * (j + 1) - l * (l + 1) - s * (s + 1);
    const denominator = 2 * l * (l + 0.5) * (l + 1);

    const E1 = this.hydrogenEnergyLevel(n);
    const relativeFactor = PC.alpha * PC.alpha / (n * n * n);

    return E1 * relativeFactor * numerator / denominator;
  }

  // Helper functions

  private static factorial(n: number): number {
    if (n <= 1) return 1;
    let result = 1;
    for (let i = 2; i <= n; i++) result *= i;
    return result;
  }

  private static hermitePolynomial(n: number, x: number): number {
    if (n === 0) return 1;
    if (n === 1) return 2 * x;

    let h_prev = 1;
    let h_curr = 2 * x;

    for (let i = 2; i <= n; i++) {
      const h_next = 2 * x * h_curr - 2 * (i - 1) * h_prev;
      h_prev = h_curr;
      h_curr = h_next;
    }

    return h_curr;
  }

  private static associatedLaguerre(n: number, alpha: number, x: number): number {
    if (n === 0) return 1;
    if (n === 1) return 1 + alpha - x;

    let L_prev = 1;
    let L_curr = 1 + alpha - x;

    for (let k = 2; k <= n; k++) {
      const L_next = ((2 * k - 1 + alpha - x) * L_curr - (k - 1 + alpha) * L_prev) / k;
      L_prev = L_curr;
      L_curr = L_next;
    }

    return L_curr;
  }
}
