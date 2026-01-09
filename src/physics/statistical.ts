/**
 * Statistical Mechanics Module
 *
 * Implements statistical mechanics and thermodynamic statistics:
 * - Maxwell-Boltzmann, Bose-Einstein, Fermi-Dirac distributions
 * - Partition functions and thermodynamic potentials
 * - Entropy calculations (Boltzmann, Gibbs, von Neumann)
 * - Phase transitions and critical phenomena
 * - Ideal gas and real gas models
 * - Ising model Monte Carlo simulation
 * - Fluctuation-dissipation relations
 */

import { PhysicalConstants } from './constants';

export interface EnsembleProperties {
  temperature: number;        // K
  pressure: number;           // Pa
  volume: number;             // m³
  particleNumber: number;     // N
  chemicalPotential: number;  // J
  internalEnergy: number;     // J
  entropy: number;            // J/K
  helmholtzFreeEnergy: number; // J
  gibbsFreeEnergy: number;    // J
  heatCapacityCv: number;     // J/K
  heatCapacityCp: number;     // J/K
}

export interface DistributionResult {
  energy: number[];
  occupation: number[];
  meanEnergy: number;
  variance: number;
  totalParticles: number;
}

export interface IsingState {
  lattice: number[][];
  magnetization: number;
  energy: number;
  temperature: number;
  correlationLength: number;
}

export interface PhaseTransitionResult {
  criticalTemperature: number;
  orderParameter: number[];
  susceptibility: number[];
  specificHeat: number[];
  temperatures: number[];
  criticalExponents: {
    beta: number;    // Order parameter
    gamma: number;   // Susceptibility
    alpha: number;   // Specific heat
    nu: number;      // Correlation length
  };
}

export interface FluctuationResult {
  energyFluctuation: number;
  particleFluctuation: number;
  compressibility: number;
  susceptibility: number;
}

const kB = PhysicalConstants.get('boltzmann_constant').value;
const h = PhysicalConstants.get('planck_constant').value;
const hbar = PhysicalConstants.get('reduced_planck_constant').value;
const NA = PhysicalConstants.get('avogadro_constant').value;

export class StatisticalMechanics {
  /**
   * Maxwell-Boltzmann velocity distribution
   */
  static maxwellBoltzmannVelocity(
    temperature: number,     // K
    mass: number,            // kg
    velocities?: number[]    // m/s (optional, generates if not provided)
  ): { velocities: number[]; probabilities: number[]; mostProbable: number; mean: number; rms: number } {
    const vp = Math.sqrt(2 * kB * temperature / mass);     // Most probable
    const vMean = Math.sqrt(8 * kB * temperature / (Math.PI * mass));  // Mean
    const vRms = Math.sqrt(3 * kB * temperature / mass);   // RMS

    if (!velocities) {
      // Generate velocities from 0 to 5 × vp
      velocities = Array.from({ length: 200 }, (_, i) => i * 5 * vp / 200);
    }

    const prefactor = 4 * Math.PI * Math.pow(mass / (2 * Math.PI * kB * temperature), 1.5);

    const probabilities = velocities.map(v => {
      const x = mass * v * v / (2 * kB * temperature);
      return prefactor * v * v * Math.exp(-x);
    });

    return {
      velocities,
      probabilities,
      mostProbable: vp,
      mean: vMean,
      rms: vRms
    };
  }

  /**
   * Maxwell-Boltzmann energy distribution
   */
  static maxwellBoltzmannEnergy(
    temperature: number,
    energies?: number[]      // J
  ): DistributionResult {
    const kT = kB * temperature;

    if (!energies) {
      energies = Array.from({ length: 200 }, (_, i) => i * 10 * kT / 200);
    }

    const occupation = energies.map(E => {
      return 2 * Math.PI * Math.pow(Math.PI * kT, -1.5) *
             Math.sqrt(E) * Math.exp(-E / kT);
    });

    // Normalize
    const total = occupation.reduce((a, b) => a + b, 0);
    const dE = energies[1] - energies[0];

    return {
      energy: energies,
      occupation: occupation.map(o => o / (total * dE)),
      meanEnergy: 1.5 * kT,
      variance: 1.5 * kT * kT,
      totalParticles: 1
    };
  }

  /**
   * Fermi-Dirac distribution
   */
  static fermiDirac(
    temperature: number,
    chemicalPotential: number,  // J (Fermi energy at T=0)
    energies?: number[]
  ): DistributionResult {
    const kT = kB * temperature;
    const mu = chemicalPotential;

    if (!energies) {
      const range = Math.max(5 * kT, 0.2 * Math.abs(mu));
      energies = Array.from({ length: 200 }, (_, i) =>
        mu - range + i * 2 * range / 200
      );
    }

    const occupation = energies.map(E => {
      if (temperature === 0) {
        return E <= mu ? 1 : 0;
      }
      const x = (E - mu) / kT;
      if (x > 100) return 0;
      if (x < -100) return 1;
      return 1 / (Math.exp(x) + 1);
    });

    // Calculate mean energy (for free electron gas)
    let meanEnergy = 0;
    let totalN = 0;
    const dE = energies[1] - energies[0];

    for (let i = 0; i < energies.length; i++) {
      const E = energies[i];
      if (E > 0) {
        const dos = Math.sqrt(E);  // Simplified DOS
        meanEnergy += E * occupation[i] * dos * dE;
        totalN += occupation[i] * dos * dE;
      }
    }

    if (totalN > 0) {
      meanEnergy /= totalN;
    }

    return {
      energy: energies,
      occupation,
      meanEnergy,
      variance: (Math.PI * kT) ** 2 / 3,  // Low-T Sommerfeld expansion
      totalParticles: totalN
    };
  }

  /**
   * Bose-Einstein distribution
   */
  static boseEinstein(
    temperature: number,
    chemicalPotential: number,  // J (must be < 0 for massive bosons)
    energies?: number[]
  ): DistributionResult {
    const kT = kB * temperature;
    const mu = chemicalPotential;

    if (!energies) {
      energies = Array.from({ length: 200 }, (_, i) =>
        i * 10 * kT / 200
      );
    }

    const occupation = energies.map(E => {
      const x = (E - mu) / kT;
      if (x > 100) return 0;
      if (x < 0) return Infinity;  // BEC singularity
      return 1 / (Math.exp(x) - 1);
    });

    // Mean energy for photon gas
    let meanEnergy = 0;
    let totalN = 0;
    const dE = energies[1] - energies[0];

    for (let i = 0; i < energies.length; i++) {
      const E = energies[i];
      if (E > 0 && isFinite(occupation[i])) {
        const dos = E * E;  // Photon DOS ∝ E²
        meanEnergy += E * occupation[i] * dos * dE;
        totalN += occupation[i] * dos * dE;
      }
    }

    if (totalN > 0) {
      meanEnergy /= totalN;
    }

    return {
      energy: energies,
      occupation,
      meanEnergy,
      variance: 0,  // Would need specific calculation
      totalParticles: totalN
    };
  }

  /**
   * Canonical partition function for harmonic oscillator
   */
  static harmonicOscillatorPartition(
    temperature: number,
    omega: number,           // Angular frequency (rad/s)
    numOscillators: number = 1
  ): {
    partitionFunction: number;
    internalEnergy: number;
    entropy: number;
    heatCapacity: number;
    helmholtzEnergy: number;
  } {
    const kT = kB * temperature;
    const beta = 1 / kT;
    const hw = hbar * omega;

    // Single oscillator partition function
    const z1 = 1 / (2 * Math.sinh(beta * hw / 2));

    // N oscillators
    const Z = Math.pow(z1, numOscillators);
    const lnZ = numOscillators * Math.log(z1);

    // Thermodynamic quantities
    const x = beta * hw;
    const cothx2 = 1 / Math.tanh(x / 2);

    const internalEnergy = numOscillators * hw * (0.5 * cothx2);
    const helmholtzEnergy = -kT * lnZ;

    // Heat capacity
    const expx = Math.exp(x);
    const heatCapacity = numOscillators * kB * (x * x * expx) / Math.pow(expx - 1, 2);

    // Entropy
    const entropy = (internalEnergy - helmholtzEnergy) / temperature;

    return {
      partitionFunction: Z,
      internalEnergy,
      entropy,
      heatCapacity,
      helmholtzEnergy
    };
  }

  /**
   * Ideal gas properties from statistical mechanics
   */
  static idealGasStatMech(
    temperature: number,
    volume: number,
    numParticles: number,
    mass: number             // Particle mass (kg)
  ): EnsembleProperties {
    const kT = kB * temperature;
    const N = numParticles;

    // Thermal de Broglie wavelength
    const lambda = h / Math.sqrt(2 * Math.PI * mass * kT);

    // Single-particle partition function
    const z1 = volume / Math.pow(lambda, 3);

    // N-particle partition function (using Stirling)
    const lnZ = N * Math.log(z1) - N * Math.log(N) + N;

    // Thermodynamic properties
    const internalEnergy = 1.5 * N * kT;
    const pressure = N * kT / volume;
    const entropy = kB * (lnZ + 2.5 * N);  // Sackur-Tetrode
    const helmholtzFreeEnergy = -kT * lnZ;
    const chemicalPotential = kT * Math.log(N * Math.pow(lambda, 3) / volume);
    const gibbsFreeEnergy = helmholtzFreeEnergy + pressure * volume;

    const heatCapacityCv = 1.5 * N * kB;
    const heatCapacityCp = 2.5 * N * kB;

    return {
      temperature,
      pressure,
      volume,
      particleNumber: N,
      chemicalPotential,
      internalEnergy,
      entropy,
      helmholtzFreeEnergy,
      gibbsFreeEnergy,
      heatCapacityCv,
      heatCapacityCp
    };
  }

  /**
   * Debye model for solid heat capacity
   */
  static debyeModel(
    temperature: number,
    debyeTemperature: number,
    numAtoms: number
  ): {
    heatCapacity: number;     // J/K
    internalEnergy: number;   // J
    entropy: number;          // J/K
    debyeFunction: number;
  } {
    const x = debyeTemperature / temperature;

    // Debye function D₃(x) = (3/x³) ∫₀ˣ t³/(eᵗ-1) dt
    const debyeFunction = this.debyeIntegral(x);

    // Heat capacity
    const heatCapacity = 3 * numAtoms * kB * debyeFunction;

    // Internal energy
    const internalEnergy = 3 * numAtoms * kB * temperature * (3 * debyeFunction / x);

    // Entropy
    const entropy = numAtoms * kB * (4 * 3 * debyeFunction / x - 3 * Math.log(1 - Math.exp(-x)));

    return {
      heatCapacity,
      internalEnergy,
      entropy,
      debyeFunction
    };
  }

  /**
   * Ising model Monte Carlo simulation (Metropolis algorithm)
   */
  static isingMonteCarlo(
    size: number,            // Lattice size (size × size)
    temperature: number,     // K
    couplingJ: number,       // Exchange coupling (J)
    externalField: number,   // External field (T)
    steps: number,           // MC steps
    equilibrationSteps: number = 1000
  ): IsingState {
    const beta = 1 / (kB * temperature);

    // Initialize random spin lattice
    const lattice: number[][] = Array(size).fill(null).map(() =>
      Array(size).fill(null).map(() => Math.random() < 0.5 ? 1 : -1)
    );

    // Calculate initial energy
    const calcEnergy = (): number => {
      let E = 0;
      for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
          const s = lattice[i][j];
          // Nearest neighbors (periodic boundary)
          const neighbors =
            lattice[(i + 1) % size][j] +
            lattice[(i - 1 + size) % size][j] +
            lattice[i][(j + 1) % size] +
            lattice[i][(j - 1 + size) % size];
          E -= couplingJ * s * neighbors / 2;  // Divide by 2 to avoid double counting
          E -= externalField * s;
        }
      }
      return E;
    };

    // Calculate magnetization
    const calcMagnetization = (): number => {
      let M = 0;
      for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
          M += lattice[i][j];
        }
      }
      return M / (size * size);
    };

    // Metropolis algorithm
    for (let step = 0; step < steps + equilibrationSteps; step++) {
      for (let k = 0; k < size * size; k++) {
        // Random site
        const i = Math.floor(Math.random() * size);
        const j = Math.floor(Math.random() * size);

        // Calculate energy change for spin flip
        const s = lattice[i][j];
        const neighbors =
          lattice[(i + 1) % size][j] +
          lattice[(i - 1 + size) % size][j] +
          lattice[i][(j + 1) % size] +
          lattice[i][(j - 1 + size) % size];

        const deltaE = 2 * s * (couplingJ * neighbors + externalField);

        // Metropolis acceptance
        if (deltaE <= 0 || Math.random() < Math.exp(-beta * deltaE)) {
          lattice[i][j] = -s;
        }
      }
    }

    const energy = calcEnergy();
    const magnetization = calcMagnetization();

    // Estimate correlation length (simplified)
    let correlationLength = 0;
    for (let r = 1; r < size / 2; r++) {
      let corr = 0;
      let count = 0;
      for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
          corr += lattice[i][j] * lattice[(i + r) % size][j];
          count++;
        }
      }
      corr /= count;
      if (corr < magnetization * magnetization * Math.exp(-1)) {
        correlationLength = r;
        break;
      }
    }

    return {
      lattice,
      magnetization,
      energy,
      temperature,
      correlationLength
    };
  }

  /**
   * Phase transition analysis using finite-size scaling
   */
  static analyzePhaseTransition(
    size: number,
    temperatureRange: [number, number],
    numTemperatures: number,
    couplingJ: number,
    mcSteps: number = 5000
  ): PhaseTransitionResult {
    const temperatures: number[] = [];
    const orderParameter: number[] = [];
    const susceptibility: number[] = [];
    const specificHeat: number[] = [];

    const Tc = 2 * couplingJ / (kB * Math.log(1 + Math.sqrt(2)));  // Exact 2D Ising

    for (let t = 0; t < numTemperatures; t++) {
      const T = temperatureRange[0] +
        (temperatureRange[1] - temperatureRange[0]) * t / (numTemperatures - 1);
      temperatures.push(T);

      // Run multiple MC simulations for statistics
      const runs = 5;
      let avgM = 0, avgM2 = 0, avgE = 0, avgE2 = 0;

      for (let run = 0; run < runs; run++) {
        const state = this.isingMonteCarlo(size, T, couplingJ, 0, mcSteps, 1000);
        const M = Math.abs(state.magnetization);
        const E = state.energy / (size * size);

        avgM += M;
        avgM2 += M * M;
        avgE += E;
        avgE2 += E * E;
      }

      avgM /= runs;
      avgM2 /= runs;
      avgE /= runs;
      avgE2 /= runs;

      orderParameter.push(avgM);

      // Susceptibility χ = N(⟨M²⟩ - ⟨M⟩²) / kT
      const chi = size * size * (avgM2 - avgM * avgM) / (kB * T);
      susceptibility.push(chi);

      // Specific heat C = (⟨E²⟩ - ⟨E⟩²) / kT²
      const C = (avgE2 - avgE * avgE) / (kB * T * T);
      specificHeat.push(C);
    }

    return {
      criticalTemperature: Tc,
      orderParameter,
      susceptibility,
      specificHeat,
      temperatures,
      criticalExponents: {
        beta: 0.125,    // 2D Ising exact
        gamma: 1.75,    // 2D Ising exact
        alpha: 0,       // Logarithmic divergence
        nu: 1           // 2D Ising exact
      }
    };
  }

  /**
   * Bose-Einstein condensation temperature
   */
  static becTemperature(
    density: number,         // particles/m³
    mass: number             // kg
  ): number {
    // Critical temperature for BEC
    // T_c = (2πℏ²/mk_B) × (n/ζ(3/2))^(2/3)
    const zeta32 = 2.612;  // Riemann zeta(3/2)

    const Tc = (2 * Math.PI * hbar * hbar / (mass * kB)) *
               Math.pow(density / zeta32, 2/3);

    return Tc;
  }

  /**
   * BEC ground state occupation
   */
  static becGroundStateOccupation(
    temperature: number,
    criticalTemperature: number,
    totalParticles: number
  ): number {
    if (temperature >= criticalTemperature) {
      return 0;
    }

    // N₀/N = 1 - (T/Tc)^(3/2)
    return totalParticles * (1 - Math.pow(temperature / criticalTemperature, 1.5));
  }

  /**
   * Fermi energy and temperature
   */
  static fermiEnergy(
    density: number,         // particles/m³
    mass: number             // kg
  ): { fermiEnergy: number; fermiTemperature: number; fermiVelocity: number; fermiWavelength: number } {
    // E_F = (ℏ²/2m)(3π²n)^(2/3)
    const EF = (hbar * hbar / (2 * mass)) * Math.pow(3 * Math.PI * Math.PI * density, 2/3);
    const TF = EF / kB;
    const vF = Math.sqrt(2 * EF / mass);
    const lambdaF = h / (mass * vF);

    return {
      fermiEnergy: EF,
      fermiTemperature: TF,
      fermiVelocity: vF,
      fermiWavelength: lambdaF
    };
  }

  /**
   * Fluctuation-dissipation theorem
   */
  static fluctuationDissipation(
    temperature: number,
    susceptibility: number,  // Response function
    volume: number
  ): FluctuationResult {
    const kT = kB * temperature;

    // Energy fluctuations ⟨(ΔE)²⟩ = kT² Cv
    const energyFluctuation = kT * kT * susceptibility / volume;

    // Particle number fluctuations ⟨(ΔN)²⟩ = kT × κ × V × n²
    // where κ is compressibility
    const compressibility = susceptibility;  // For isothermal
    const particleFluctuation = kT * compressibility * volume;

    return {
      energyFluctuation,
      particleFluctuation,
      compressibility,
      susceptibility
    };
  }

  /**
   * Entropy of mixing (ideal gases)
   */
  static entropyOfMixing(
    moleFractions: number[],
    totalMoles: number
  ): number {
    const R = PhysicalConstants.get('molar_gas_constant').value;

    let entropy = 0;
    for (const x of moleFractions) {
      if (x > 0) {
        entropy -= x * Math.log(x);
      }
    }

    return totalMoles * R * entropy;
  }

  /**
   * Gibbs entropy
   */
  static gibbsEntropy(probabilities: number[]): number {
    let S = 0;
    for (const p of probabilities) {
      if (p > 0) {
        S -= p * Math.log(p);
      }
    }
    return kB * S;
  }

  /**
   * Boltzmann entropy from microstates
   */
  static boltzmannEntropy(numMicrostates: number): number {
    return kB * Math.log(numMicrostates);
  }

  /**
   * Landau free energy expansion near phase transition
   */
  static landauFreeEnergy(
    orderParameter: number,
    temperature: number,
    criticalTemperature: number,
    coefficients: { a0: number; b: number; c?: number }
  ): {
    freeEnergy: number;
    equilibriumOrder: number;
    susceptibility: number;
  } {
    const { a0, b, c = 0 } = coefficients;
    const t = (temperature - criticalTemperature) / criticalTemperature;

    // F = F₀ + a₀t×φ² + b×φ⁴ + c×φ⁶
    const phi = orderParameter;
    const freeEnergy = a0 * t * phi * phi + b * Math.pow(phi, 4) + c * Math.pow(phi, 6);

    // Equilibrium order parameter (minimizing F)
    let equilibriumOrder: number;
    if (t > 0) {
      equilibriumOrder = 0;  // Disordered phase
    } else {
      // dF/dφ = 0: 2a₀t×φ + 4b×φ³ = 0 → φ² = -a₀t/(2b)
      equilibriumOrder = Math.sqrt(-a0 * t / (2 * b));
    }

    // Susceptibility χ = 1/(2a₀|t|) for t ≠ 0
    const susceptibility = t !== 0 ? 1 / (2 * a0 * Math.abs(t)) : Infinity;

    return {
      freeEnergy,
      equilibriumOrder,
      susceptibility
    };
  }

  /**
   * Van der Waals equation of state
   */
  static vanDerWaals(
    temperature: number,
    molarVolume: number,     // m³/mol
    a: number,               // Attraction parameter (Pa·m⁶/mol²)
    b: number                // Excluded volume (m³/mol)
  ): {
    pressure: number;
    compressibilityFactor: number;
    criticalPoint: { Tc: number; Pc: number; Vc: number };
  } {
    const R = PhysicalConstants.get('molar_gas_constant').value;

    // p = RT/(Vm - b) - a/Vm²
    const pressure = R * temperature / (molarVolume - b) - a / (molarVolume * molarVolume);

    // Compressibility factor Z = pVm/(RT)
    const compressibilityFactor = pressure * molarVolume / (R * temperature);

    // Critical point
    const Vc = 3 * b;
    const Pc = a / (27 * b * b);
    const Tc = 8 * a / (27 * R * b);

    return {
      pressure,
      compressibilityFactor,
      criticalPoint: { Tc, Pc, Vc }
    };
  }

  /**
   * Planck distribution (blackbody radiation)
   */
  static planckDistribution(
    temperature: number,
    frequencies?: number[]
  ): {
    frequencies: number[];
    spectralRadiance: number[];  // W/(m²·sr·Hz)
    peakFrequency: number;
    totalPower: number;          // W/m² (Stefan-Boltzmann)
  } {
    const c = PhysicalConstants.get('speed_of_light').value;

    // Peak frequency (Wien's law)
    const peakFrequency = 2.821 * kB * temperature / h;

    if (!frequencies) {
      frequencies = Array.from({ length: 200 }, (_, i) =>
        (i + 1) * 5 * peakFrequency / 200
      );
    }

    const spectralRadiance = frequencies.map(nu => {
      const x = h * nu / (kB * temperature);
      if (x > 100) return 0;
      return 2 * h * nu * nu * nu / (c * c * (Math.exp(x) - 1));
    });

    // Stefan-Boltzmann total power
    const sigma = 5.670374419e-8;  // Stefan-Boltzmann constant
    const totalPower = sigma * Math.pow(temperature, 4);

    return {
      frequencies,
      spectralRadiance,
      peakFrequency,
      totalPower
    };
  }

  // Helper: Debye integral using Gauss-Legendre quadrature
  private static debyeIntegral(x: number): number {
    if (x < 0.1) {
      // Low temperature limit
      return 1 - x * x / 20 + x * x * x * x / 560;
    }
    if (x > 20) {
      // High temperature limit (classical)
      const x3 = x * x * x;
      return (3 / x3) * (Math.PI * Math.PI * Math.PI * Math.PI / 15);
    }

    // Numerical integration
    const n = 100;
    let sum = 0;
    const dt = x / n;

    for (let i = 1; i < n; i++) {
      const t = i * dt;
      const et = Math.exp(t);
      if (et > 1) {
        sum += t * t * t * t * et / ((et - 1) * (et - 1));
      }
    }

    return (3 / (x * x * x)) * sum * dt;
  }
}
