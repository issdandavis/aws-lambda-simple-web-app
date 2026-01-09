/**
 * Nuclear and Particle Physics Module
 *
 * Implements nuclear and high-energy particle physics:
 * - Nuclear decay (alpha, beta, gamma)
 * - Binding energy and mass defect
 * - Cross sections and reaction rates
 * - Relativistic kinematics
 * - Particle interactions (Compton, pair production)
 * - Fusion and fission reactions
 * - Radiation transport basics
 */

import { PhysicalConstants } from './constants';

// Fundamental particle data
export interface ParticleData {
  name: string;
  symbol: string;
  mass: number;           // kg
  charge: number;         // elementary charges
  spin: number;           // units of ℏ
  lifetime?: number;      // seconds (undefined = stable)
  antiparticle?: string;
}

export interface NuclideData {
  Z: number;              // Atomic number
  A: number;              // Mass number
  symbol: string;
  mass: number;           // Atomic mass (u)
  bindingEnergy: number;  // MeV
  halfLife?: number;      // seconds
  decayModes?: DecayMode[];
  abundance?: number;     // Natural abundance (0-1)
}

export interface DecayMode {
  type: 'alpha' | 'beta-' | 'beta+' | 'gamma' | 'ec' | 'sf' | 'n' | 'p';
  branchingRatio: number;
  qValue: number;         // MeV
  daughter?: { Z: number; A: number };
}

export interface DecayResult {
  activity: number;       // Becquerel
  decayConstant: number;  // s⁻¹
  halfLife: number;       // s
  meanLifetime: number;   // s
  nucleiRemaining: number;
  nucleiDecayed: number;
  timeElapsed: number;
}

export interface CrossSectionResult {
  total: number;          // barns
  elastic: number;        // barns
  inelastic: number;      // barns
  absorption: number;     // barns
  meanFreePath: number;   // m
}

export interface ReactionResult {
  qValue: number;         // MeV
  threshold: number;      // MeV (kinetic energy)
  products: { particle: string; energy: number; angle: number }[];
  isExothermic: boolean;
}

export interface ComptonResult {
  scatteredPhotonEnergy: number;  // eV
  electronEnergy: number;          // eV
  wavelengthShift: number;         // m
  differentialCrossSection: number; // barns/sr
  totalCrossSection: number;       // barns
}

// Standard particle masses in kg
const PARTICLES: Record<string, ParticleData> = {
  electron: {
    name: 'Electron',
    symbol: 'e⁻',
    mass: 9.1093837015e-31,
    charge: -1,
    spin: 0.5,
    antiparticle: 'positron'
  },
  positron: {
    name: 'Positron',
    symbol: 'e⁺',
    mass: 9.1093837015e-31,
    charge: 1,
    spin: 0.5,
    antiparticle: 'electron'
  },
  proton: {
    name: 'Proton',
    symbol: 'p',
    mass: 1.67262192369e-27,
    charge: 1,
    spin: 0.5,
    antiparticle: 'antiproton'
  },
  neutron: {
    name: 'Neutron',
    symbol: 'n',
    mass: 1.67492749804e-27,
    charge: 0,
    spin: 0.5,
    lifetime: 879.4,
    antiparticle: 'antineutron'
  },
  muon: {
    name: 'Muon',
    symbol: 'μ⁻',
    mass: 1.883531627e-28,
    charge: -1,
    spin: 0.5,
    lifetime: 2.1969811e-6,
    antiparticle: 'antimuon'
  },
  pion_plus: {
    name: 'Pion+',
    symbol: 'π⁺',
    mass: 2.48808e-28,
    charge: 1,
    spin: 0,
    lifetime: 2.6033e-8,
    antiparticle: 'pion_minus'
  },
  pion_zero: {
    name: 'Pion0',
    symbol: 'π⁰',
    mass: 2.40618e-28,
    charge: 0,
    spin: 0,
    lifetime: 8.52e-17
  },
  alpha: {
    name: 'Alpha',
    symbol: 'α',
    mass: 6.6446573357e-27,
    charge: 2,
    spin: 0
  },
  photon: {
    name: 'Photon',
    symbol: 'γ',
    mass: 0,
    charge: 0,
    spin: 1
  }
};

// Common nuclides data
const NUCLIDES: Record<string, NuclideData> = {
  'H-1': { Z: 1, A: 1, symbol: 'H', mass: 1.007825, bindingEnergy: 0, abundance: 0.99985 },
  'H-2': { Z: 1, A: 2, symbol: 'D', mass: 2.014102, bindingEnergy: 2.224, abundance: 0.00015 },
  'H-3': { Z: 1, A: 3, symbol: 'T', mass: 3.016049, bindingEnergy: 8.482, halfLife: 3.888e8 },
  'He-3': { Z: 2, A: 3, symbol: 'He', mass: 3.016029, bindingEnergy: 7.718, abundance: 0.00000134 },
  'He-4': { Z: 2, A: 4, symbol: 'He', mass: 4.002603, bindingEnergy: 28.296, abundance: 0.99999866 },
  'Li-6': { Z: 3, A: 6, symbol: 'Li', mass: 6.015122, bindingEnergy: 31.995, abundance: 0.0759 },
  'Li-7': { Z: 3, A: 7, symbol: 'Li', mass: 7.016004, bindingEnergy: 39.245, abundance: 0.9241 },
  'C-12': { Z: 6, A: 12, symbol: 'C', mass: 12.000000, bindingEnergy: 92.162, abundance: 0.9893 },
  'C-14': { Z: 6, A: 14, symbol: 'C', mass: 14.003242, bindingEnergy: 105.285, halfLife: 1.807e11 },
  'N-14': { Z: 7, A: 14, symbol: 'N', mass: 14.003074, bindingEnergy: 104.659, abundance: 0.99636 },
  'O-16': { Z: 8, A: 16, symbol: 'O', mass: 15.994915, bindingEnergy: 127.619, abundance: 0.99757 },
  'Fe-56': { Z: 26, A: 56, symbol: 'Fe', mass: 55.934937, bindingEnergy: 492.254, abundance: 0.9175 },
  'U-235': { Z: 92, A: 235, symbol: 'U', mass: 235.043930, bindingEnergy: 1783.870, halfLife: 2.221e16, abundance: 0.0072 },
  'U-238': { Z: 92, A: 238, symbol: 'U', mass: 238.050788, bindingEnergy: 1801.695, halfLife: 1.409e17, abundance: 0.9927 },
  'Pu-239': { Z: 94, A: 239, symbol: 'Pu', mass: 239.052163, bindingEnergy: 1806.923, halfLife: 7.612e11 }
};

// Unit conversions
const eV_to_J = 1.602176634e-19;
const MeV_to_J = eV_to_J * 1e6;
const J_to_MeV = 1 / MeV_to_J;
const u_to_kg = 1.66053906660e-27;
const kg_to_u = 1 / u_to_kg;
const barn_to_m2 = 1e-28;

export class NuclearPhysics {
  /**
   * Calculate nuclear binding energy using semi-empirical mass formula (Bethe-Weizsäcker)
   */
  static bindingEnergySEMF(Z: number, A: number): {
    total: number;        // MeV
    perNucleon: number;   // MeV
    volume: number;
    surface: number;
    coulomb: number;
    asymmetry: number;
    pairing: number;
  } {
    const N = A - Z;

    // SEMF coefficients (MeV)
    const aV = 15.75;     // Volume
    const aS = 17.8;      // Surface
    const aC = 0.711;     // Coulomb
    const aA = 23.7;      // Asymmetry
    const aP = 11.18;     // Pairing

    // Volume term
    const volume = aV * A;

    // Surface term
    const surface = -aS * Math.pow(A, 2/3);

    // Coulomb term
    const coulomb = -aC * Z * (Z - 1) / Math.pow(A, 1/3);

    // Asymmetry term
    const asymmetry = -aA * Math.pow(N - Z, 2) / A;

    // Pairing term
    let pairing = 0;
    if (Z % 2 === 0 && N % 2 === 0) {
      pairing = aP / Math.sqrt(A);  // Even-even
    } else if (Z % 2 === 1 && N % 2 === 1) {
      pairing = -aP / Math.sqrt(A); // Odd-odd
    }

    const total = volume + surface + coulomb + asymmetry + pairing;

    return {
      total,
      perNucleon: total / A,
      volume,
      surface,
      coulomb,
      asymmetry,
      pairing
    };
  }

  /**
   * Calculate Q-value for nuclear reaction
   */
  static qValue(
    reactants: { Z: number; A: number; mass?: number }[],
    products: { Z: number; A: number; mass?: number }[]
  ): number {
    const c2 = PhysicalConstants.get('speed_of_light').value ** 2;

    // Get masses (use SEMF if not provided)
    const getMass = (particle: { Z: number; A: number; mass?: number }): number => {
      if (particle.mass) return particle.mass * u_to_kg;

      // Calculate from binding energy
      const BE = this.bindingEnergySEMF(particle.Z, particle.A).total * MeV_to_J;
      const mp = PARTICLES.proton.mass;
      const mn = PARTICLES.neutron.mass;
      return particle.Z * mp + (particle.A - particle.Z) * mn - BE / c2;
    };

    const reactantMass = reactants.reduce((sum, r) => sum + getMass(r), 0);
    const productMass = products.reduce((sum, p) => sum + getMass(p), 0);

    return (reactantMass - productMass) * c2 * J_to_MeV;
  }

  /**
   * Radioactive decay calculation
   */
  static decay(
    initialNuclei: number,
    halfLife: number,
    time: number
  ): DecayResult {
    const decayConstant = Math.LN2 / halfLife;
    const meanLifetime = 1 / decayConstant;

    const nucleiRemaining = initialNuclei * Math.exp(-decayConstant * time);
    const nucleiDecayed = initialNuclei - nucleiRemaining;

    const activity = decayConstant * nucleiRemaining;

    return {
      activity,
      decayConstant,
      halfLife,
      meanLifetime,
      nucleiRemaining,
      nucleiDecayed,
      timeElapsed: time
    };
  }

  /**
   * Activity from mass of radioactive material
   */
  static activityFromMass(
    massKg: number,
    atomicMass: number,    // u
    halfLife: number       // s
  ): number {
    const NA = PhysicalConstants.get('avogadro_constant').value;
    const numAtoms = massKg * NA / (atomicMass * 1e-3);
    const lambda = Math.LN2 / halfLife;
    return lambda * numAtoms;  // Bq
  }

  /**
   * Bateman equation for decay chain
   */
  static decayChain(
    initialNuclei: number[],      // Initial number of each species
    halfLives: number[],          // Half-lives of each species
    time: number
  ): number[] {
    const n = halfLives.length;
    const lambda = halfLives.map(t => Math.LN2 / t);
    const N = new Array(n).fill(0);

    // Bateman equations (general solution)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        // Calculate product of lambdas
        let lambdaProd = 1;
        for (let k = 0; k < i; k++) {
          lambdaProd *= lambda[k];
        }

        // Calculate sum term
        let sum = 0;
        for (let k = j; k <= i; k++) {
          let denom = 1;
          for (let l = j; l <= i; l++) {
            if (l !== k) {
              denom *= lambda[l] - lambda[k];
            }
          }
          if (Math.abs(denom) > 1e-30) {
            sum += Math.exp(-lambda[k] * time) / denom;
          }
        }

        N[i] += initialNuclei[j] * lambdaProd * sum;
      }
    }

    return N;
  }

  /**
   * Compton scattering
   */
  static comptonScattering(
    photonEnergyEV: number,
    scatteringAngle: number    // radians
  ): ComptonResult {
    const c = PhysicalConstants.get('speed_of_light').value;
    const h = PhysicalConstants.get('planck_constant').value;
    const me = PARTICLES.electron.mass;
    const re = 2.8179403262e-15;  // Classical electron radius

    // Compton wavelength of electron
    const lambdaC = h / (me * c);

    // Initial wavelength
    const lambda0 = h * c / (photonEnergyEV * eV_to_J);

    // Wavelength shift
    const wavelengthShift = lambdaC * (1 - Math.cos(scatteringAngle));

    // Final wavelength
    const lambda1 = lambda0 + wavelengthShift;

    // Scattered photon energy
    const scatteredPhotonEnergy = h * c / lambda1 / eV_to_J;

    // Electron kinetic energy
    const electronEnergy = photonEnergyEV - scatteredPhotonEnergy;

    // Klein-Nishina differential cross section
    const P = lambda0 / lambda1;
    const differentialCrossSection = 0.5 * re * re * P * P *
      (P + 1/P - Math.sin(scatteringAngle) ** 2) * 1e28;  // barns/sr

    // Total cross section (Thomson limit for low energy)
    const x = photonEnergyEV * eV_to_J / (me * c * c);
    let totalCrossSection: number;

    if (x < 0.01) {
      // Thomson scattering
      totalCrossSection = (8 * Math.PI / 3) * re * re * 1e28;
    } else {
      // Klein-Nishina formula
      totalCrossSection = 2 * Math.PI * re * re * (
        (1 + x) / (x * x * x) * (
          2 * x * (1 + x) / (1 + 2 * x) - Math.log(1 + 2 * x)
        ) +
        Math.log(1 + 2 * x) / (2 * x) -
        (1 + 3 * x) / Math.pow(1 + 2 * x, 2)
      ) * 1e28;
    }

    return {
      scatteredPhotonEnergy,
      electronEnergy,
      wavelengthShift,
      differentialCrossSection,
      totalCrossSection
    };
  }

  /**
   * Pair production threshold and kinematics
   */
  static pairProduction(
    photonEnergyEV: number
  ): {
    canOccur: boolean;
    threshold: number;
    electronEnergy: number;
    positronEnergy: number;
    crossSection: number;
  } {
    const me = PARTICLES.electron.mass;
    const c = PhysicalConstants.get('speed_of_light').value;
    const mec2 = me * c * c / eV_to_J;  // 511 keV

    const threshold = 2 * mec2;  // 1.022 MeV
    const canOccur = photonEnergyEV > threshold;

    let electronEnergy = 0;
    let positronEnergy = 0;
    let crossSection = 0;

    if (canOccur) {
      // Excess energy distributed between electron and positron
      const excessEnergy = photonEnergyEV - threshold;

      // Simple symmetric case (actual distribution varies)
      electronEnergy = mec2 + excessEnergy / 2;
      positronEnergy = mec2 + excessEnergy / 2;

      // Approximate cross section (Bethe-Heitler, high energy)
      const alpha = 1 / 137.036;  // Fine structure constant
      const re = 2.8179403262e-15;
      const Z = 26;  // Assume iron target

      if (photonEnergyEV > 10 * mec2) {
        crossSection = alpha * re * re * Z * Z *
          (28/9 * Math.log(2 * photonEnergyEV / mec2) - 218/27) * 1e28;
      }
    }

    return {
      canOccur,
      threshold,
      electronEnergy,
      positronEnergy,
      crossSection
    };
  }

  /**
   * Electron-positron annihilation
   */
  static annihilation(
    electronKE: number,   // eV
    positronKE: number    // eV
  ): {
    photonEnergies: number[];
    totalEnergy: number;
    isAtRest: boolean;
  } {
    const me = PARTICLES.electron.mass;
    const c = PhysicalConstants.get('speed_of_light').value;
    const mec2 = me * c * c / eV_to_J;

    const totalEnergy = 2 * mec2 + electronKE + positronKE;
    const isAtRest = electronKE < 1000 && positronKE < 1000;  // < 1 keV

    let photonEnergies: number[];

    if (isAtRest) {
      // Two 511 keV photons emitted back-to-back
      photonEnergies = [mec2, mec2];
    } else {
      // Non-trivial kinematics - simplified two-photon case
      // Energy split depends on angles and momenta
      const gamma = (totalEnergy) / (2 * mec2);
      const beta = Math.sqrt(1 - 1 / (gamma * gamma));

      // Forward and backward photon energies in CM frame transformed
      const E_forward = mec2 * gamma * (1 + beta);
      const E_backward = mec2 * gamma * (1 - beta);

      photonEnergies = [E_forward, E_backward];
    }

    return {
      photonEnergies,
      totalEnergy,
      isAtRest
    };
  }

  /**
   * Neutron cross sections (simplified parametric model)
   */
  static neutronCrossSection(
    energyEV: number,
    targetZ: number,
    targetA: number
  ): CrossSectionResult {
    // Very simplified model - real data comes from ENDF/B libraries

    const R = 1.2e-15 * Math.pow(targetA, 1/3);  // Nuclear radius
    const geometricCS = Math.PI * R * R / barn_to_m2;

    // Thermal neutrons (1/v law for absorption)
    const thermalEnergy = 0.0253;  // eV (room temp)
    const vRatio = Math.sqrt(thermalEnergy / energyEV);

    // Absorption cross section (simplified)
    let absorption = 0;
    if (targetZ === 5 && targetA === 10) {
      absorption = 3840 * vRatio;  // B-10
    } else if (targetZ === 48) {
      absorption = 2520 * vRatio;  // Cd
    } else if (targetZ === 64) {
      absorption = 49000 * vRatio; // Gd
    } else if (targetZ === 92 && targetA === 235) {
      absorption = 585 * vRatio;   // U-235 fission
    } else {
      absorption = 0.1 * targetA * vRatio;  // Generic
    }

    // Elastic scattering
    const elastic = geometricCS * (1 + (energyEV < 1 ? 1 : 0));

    // Inelastic (above threshold)
    const inelastic = energyEV > 1e6 ? geometricCS * 0.3 : 0;

    const total = elastic + inelastic + absorption;

    // Mean free path (assume typical density)
    const density = 8000;  // kg/m³
    const NA = PhysicalConstants.get('avogadro_constant').value;
    const n = density * NA / (targetA * 1e-3);  // Number density
    const meanFreePath = 1 / (n * total * barn_to_m2);

    return {
      total,
      elastic,
      inelastic,
      absorption,
      meanFreePath
    };
  }

  /**
   * Bethe-Bloch stopping power for charged particles
   */
  static stoppingPower(
    particleMass: number,     // kg
    particleCharge: number,   // units of e
    kineticEnergy: number,    // eV
    targetZ: number,
    targetA: number,
    targetDensity: number     // kg/m³
  ): {
    electronicStoppingPower: number;  // MeV cm²/g
    range: number;                     // cm
    energyLoss: number;               // MeV/cm
  } {
    const c = PhysicalConstants.get('speed_of_light').value;
    const me = PARTICLES.electron.mass;
    const re = 2.8179403262e-15;
    const NA = PhysicalConstants.get('avogadro_constant').value;

    // Particle velocity
    const E0 = particleMass * c * c;
    const gamma = 1 + kineticEnergy * eV_to_J / E0;
    const beta = Math.sqrt(1 - 1 / (gamma * gamma));
    const beta2 = beta * beta;

    // Mean excitation energy (approximate)
    const I = targetZ * 10 * eV_to_J;  // ~10Z eV

    // Maximum energy transfer
    const Tmax = 2 * me * c * c * beta2 * gamma * gamma /
      (1 + 2 * gamma * me / particleMass + (me / particleMass) ** 2);

    // Bethe-Bloch formula
    const K = 4 * Math.PI * NA * re * re * me * c * c;
    const factor = K * targetZ / targetA * (particleCharge ** 2) / beta2;

    const ln_term = Math.log(2 * me * c * c * beta2 * gamma * gamma * Tmax / (I * I));
    const dEdx = factor * (0.5 * ln_term - beta2) * J_to_MeV * 100 / (targetDensity / 1000);

    // Energy loss per cm
    const energyLoss = dEdx * targetDensity / 1000;

    // Approximate range (simplified - integrate dE/dx)
    const range = kineticEnergy * eV_to_J * J_to_MeV / energyLoss;

    return {
      electronicStoppingPower: dEdx,
      range,
      energyLoss
    };
  }

  /**
   * Nuclear fusion reaction rates
   */
  static fusionReactionRate(
    reaction: 'DT' | 'DD' | 'DHe3',
    temperatureKeV: number,
    densityN: number          // particles/m³
  ): {
    crossSection: number;     // barns
    reactivity: number;       // m³/s
    powerDensity: number;     // W/m³
    lawsonProduct: number;    // keV s/m³
  } {
    // Parametric fits to fusion cross sections
    let crossSection: number;
    let qValue: number;

    const T = temperatureKeV;

    switch (reaction) {
      case 'DT':
        // D + T → He-4 + n
        qValue = 17.6;  // MeV
        // Bosch-Hale parametrization
        const BG_DT = 34.3827;
        const A1 = 6.927e4, A2 = 7.454e8, A3 = 2.050e6, A4 = 5.2002e4, A5 = 0;
        const B1 = 6.38e1, B2 = -9.95e-1, B3 = 6.981e-5, B4 = 1.728e-4;
        const theta = T / (1 - (T * (B1 + T * (B2 + T * (B3 + T * B4)))) /
                              (A1 + T * (A2 + T * (A3 + T * (A4 + T * A5)))));
        const xi = Math.pow(BG_DT * BG_DT / (4 * theta), 1/3);
        crossSection = (A1 + T * (A2 + T * (A3 + T * (A4 + T * A5)))) /
                       (T * Math.exp(BG_DT / Math.sqrt(T))) * 1e-3;
        break;

      case 'DD':
        // D + D → T + p or D + D → He-3 + n
        qValue = 3.65;  // MeV (average)
        crossSection = 3.7e-2 * Math.exp(-18.76 / Math.sqrt(T)) / Math.sqrt(T);
        break;

      case 'DHe3':
        // D + He-3 → He-4 + p
        qValue = 18.3;
        crossSection = 5.4e-2 * Math.exp(-21.2 / Math.sqrt(T)) / Math.sqrt(T);
        break;

      default:
        crossSection = 0;
        qValue = 0;
    }

    // Reactivity <σv> (Maxwell-averaged)
    const v_thermal = Math.sqrt(2 * T * 1000 * eV_to_J / (2 * u_to_kg));  // Approximate
    const reactivity = crossSection * barn_to_m2 * v_thermal;

    // Power density
    const powerDensity = 0.25 * densityN * densityN * reactivity * qValue * MeV_to_J;

    // Lawson criterion product (nτE > threshold for ignition)
    const lawsonProduct = densityN * T;  // Simplified

    return {
      crossSection,
      reactivity,
      powerDensity,
      lawsonProduct
    };
  }

  /**
   * Fission energy release
   */
  static fissionEnergy(
    nuclide: 'U-235' | 'U-238' | 'Pu-239'
  ): {
    totalEnergy: number;          // MeV
    kineticFragments: number;     // MeV
    promptNeutrons: number;       // MeV
    promptGamma: number;          // MeV
    delayedBetaGamma: number;    // MeV
    neutrinos: number;            // MeV
    neutronYield: number;         // neutrons/fission
  } {
    switch (nuclide) {
      case 'U-235':
        return {
          totalEnergy: 202.5,
          kineticFragments: 169.1,
          promptNeutrons: 4.8,
          promptGamma: 7.0,
          delayedBetaGamma: 13.4,
          neutrinos: 8.8,
          neutronYield: 2.43
        };

      case 'U-238':
        return {
          totalEnergy: 205.0,
          kineticFragments: 170.0,
          promptNeutrons: 5.5,
          promptGamma: 7.5,
          delayedBetaGamma: 13.5,
          neutrinos: 9.0,
          neutronYield: 2.80
        };

      case 'Pu-239':
        return {
          totalEnergy: 210.0,
          kineticFragments: 175.0,
          promptNeutrons: 5.9,
          promptGamma: 7.8,
          delayedBetaGamma: 12.5,
          neutrinos: 8.6,
          neutronYield: 2.88
        };

      default:
        return {
          totalEnergy: 0,
          kineticFragments: 0,
          promptNeutrons: 0,
          promptGamma: 0,
          delayedBetaGamma: 0,
          neutrinos: 0,
          neutronYield: 0
        };
    }
  }

  /**
   * Relativistic kinematics for two-body decay
   */
  static twoBodyDecay(
    parentMass: number,       // kg
    product1Mass: number,     // kg
    product2Mass: number      // kg
  ): {
    product1Energy: number;   // J
    product2Energy: number;   // J
    product1Momentum: number; // kg m/s
    qValue: number;           // J
  } {
    const c = PhysicalConstants.get('speed_of_light').value;
    const c2 = c * c;

    const M = parentMass;
    const m1 = product1Mass;
    const m2 = product2Mass;

    // Q-value
    const qValue = (M - m1 - m2) * c2;

    // Product energies in parent rest frame
    const E1 = (M * M + m1 * m1 - m2 * m2) * c2 / (2 * M);
    const E2 = (M * M + m2 * m2 - m1 * m1) * c2 / (2 * M);

    // Momentum magnitude (same for both, opposite directions)
    const p = Math.sqrt(E1 * E1 - m1 * m1 * c2 * c2) / c;

    return {
      product1Energy: E1,
      product2Energy: E2,
      product1Momentum: p,
      qValue
    };
  }

  /**
   * Relativistic invariant mass
   */
  static invariantMass(
    particles: { energy: number; momentum: { x: number; y: number; z: number } }[]
  ): number {
    const c = PhysicalConstants.get('speed_of_light').value;

    let totalE = 0;
    let totalPx = 0;
    let totalPy = 0;
    let totalPz = 0;

    for (const p of particles) {
      totalE += p.energy;
      totalPx += p.momentum.x;
      totalPy += p.momentum.y;
      totalPz += p.momentum.z;
    }

    const p2 = totalPx * totalPx + totalPy * totalPy + totalPz * totalPz;
    const E2 = totalE * totalE;

    return Math.sqrt(E2 - p2 * c * c) / (c * c);
  }

  // Export particle and nuclide data
  static getParticle(name: string): ParticleData | undefined {
    return PARTICLES[name];
  }

  static getNuclide(name: string): NuclideData | undefined {
    return NUCLIDES[name];
  }

  static getAllParticles(): Record<string, ParticleData> {
    return { ...PARTICLES };
  }

  static getAllNuclides(): Record<string, NuclideData> {
    return { ...NUCLIDES };
  }
}

export { PARTICLES, NUCLIDES };
