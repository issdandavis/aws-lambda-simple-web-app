/**
 * CODATA 2018 Physical Constants
 * Source: NIST Reference on Constants, Units, and Uncertainty
 * https://physics.nist.gov/cuu/Constants/
 */

export const PhysicalConstants = {
  // Fundamental Constants
  c: 299792458,                          // Speed of light in vacuum (m/s) - exact
  h: 6.62607015e-34,                     // Planck constant (J·s) - exact
  hbar: 1.054571817e-34,                 // Reduced Planck constant (J·s)
  G: 6.67430e-11,                        // Gravitational constant (m³/(kg·s²))
  e: 1.602176634e-19,                    // Elementary charge (C) - exact

  // Electromagnetic Constants
  epsilon0: 8.8541878128e-12,            // Vacuum electric permittivity (F/m)
  mu0: 1.25663706212e-6,                 // Vacuum magnetic permeability (H/m)
  ke: 8.9875517923e9,                    // Coulomb constant (N·m²/C²)

  // Atomic and Nuclear Constants
  me: 9.1093837015e-31,                  // Electron mass (kg)
  mp: 1.67262192369e-27,                 // Proton mass (kg)
  mn: 1.67492749804e-27,                 // Neutron mass (kg)
  mmu: 1.883531627e-28,                  // Muon mass (kg)
  alpha: 7.2973525693e-3,                // Fine-structure constant (dimensionless)
  a0: 5.29177210903e-11,                 // Bohr radius (m)
  Rinf: 10973731.568160,                 // Rydberg constant (1/m)

  // Thermodynamic Constants
  kB: 1.380649e-23,                      // Boltzmann constant (J/K) - exact
  NA: 6.02214076e23,                     // Avogadro constant (1/mol) - exact
  R: 8.314462618,                        // Molar gas constant (J/(mol·K))
  sigma: 5.670374419e-8,                 // Stefan-Boltzmann constant (W/(m²·K⁴))

  // Quantum Mechanics
  muB: 9.2740100783e-24,                 // Bohr magneton (J/T)
  muN: 5.0507837461e-27,                 // Nuclear magneton (J/T)
  ge: -2.00231930436256,                 // Electron g-factor

  // Particle Physics
  MeV_c2_to_kg: 1.78266192e-30,          // MeV/c² to kg conversion
  eV_to_J: 1.602176634e-19,              // eV to Joules conversion

  // Cosmological Constants
  H0: 67.4,                              // Hubble constant (km/s/Mpc) - Planck 2018
  rhoC: 9.47e-27,                        // Critical density (kg/m³)

  // Mathematical Constants used in physics
  pi: Math.PI,
  e_math: Math.E,
  sqrt2: Math.SQRT2,
} as const;

export type ConstantName = keyof typeof PhysicalConstants;

/**
 * Get constant with uncertainty information
 */
export interface ConstantWithUncertainty {
  value: number;
  uncertainty: number;
  unit: string;
  description: string;
}

export const ConstantDetails: Record<string, ConstantWithUncertainty> = {
  c: { value: 299792458, uncertainty: 0, unit: 'm/s', description: 'Speed of light in vacuum' },
  h: { value: 6.62607015e-34, uncertainty: 0, unit: 'J·s', description: 'Planck constant' },
  hbar: { value: 1.054571817e-34, uncertainty: 0, unit: 'J·s', description: 'Reduced Planck constant' },
  G: { value: 6.67430e-11, uncertainty: 1.5e-15, unit: 'm³/(kg·s²)', description: 'Gravitational constant' },
  e: { value: 1.602176634e-19, uncertainty: 0, unit: 'C', description: 'Elementary charge' },
  me: { value: 9.1093837015e-31, uncertainty: 2.8e-40, unit: 'kg', description: 'Electron mass' },
  mp: { value: 1.67262192369e-27, uncertainty: 5.1e-37, unit: 'kg', description: 'Proton mass' },
  kB: { value: 1.380649e-23, uncertainty: 0, unit: 'J/K', description: 'Boltzmann constant' },
  NA: { value: 6.02214076e23, uncertainty: 0, unit: '1/mol', description: 'Avogadro constant' },
  alpha: { value: 7.2973525693e-3, uncertainty: 1.1e-12, unit: '', description: 'Fine-structure constant' },
  a0: { value: 5.29177210903e-11, uncertainty: 8.0e-21, unit: 'm', description: 'Bohr radius' },
};
