/**
 * Physical Constants Tests
 * Validates CODATA 2018 values
 */

import { PhysicalConstants, ConstantDetails } from '../../../src/physics/constants';

describe('PhysicalConstants', () => {
  describe('Fundamental Constants (Exact)', () => {
    test('speed of light should be exactly 299792458 m/s', () => {
      expect(PhysicalConstants.c).toBe(299792458);
    });

    test('Planck constant should be exactly 6.62607015e-34 J·s', () => {
      expect(PhysicalConstants.h).toBe(6.62607015e-34);
    });

    test('elementary charge should be exactly 1.602176634e-19 C', () => {
      expect(PhysicalConstants.e).toBe(1.602176634e-19);
    });

    test('Boltzmann constant should be exactly 1.380649e-23 J/K', () => {
      expect(PhysicalConstants.kB).toBe(1.380649e-23);
    });

    test('Avogadro constant should be exactly 6.02214076e23 /mol', () => {
      expect(PhysicalConstants.NA).toBe(6.02214076e23);
    });
  });

  describe('Derived Constants', () => {
    test('reduced Planck constant should be h/(2π)', () => {
      const expectedHbar = PhysicalConstants.h / (2 * Math.PI);
      expect(PhysicalConstants.hbar).toBeCloseTo(expectedHbar, 20);
    });

    test('vacuum permeability should be approximately 1.257e-6 H/m', () => {
      expect(PhysicalConstants.mu0).toBeCloseTo(1.25663706212e-6, 15);
    });

    test('vacuum permittivity should be approximately 8.854e-12 F/m', () => {
      expect(PhysicalConstants.epsilon0).toBeCloseTo(8.8541878128e-12, 20);
    });

    test('Coulomb constant should satisfy ke = 1/(4πε₀)', () => {
      const expectedKe = 1 / (4 * Math.PI * PhysicalConstants.epsilon0);
      expect(PhysicalConstants.ke).toBeCloseTo(expectedKe, 3);
    });
  });

  describe('Particle Masses', () => {
    test('electron mass should be approximately 9.109e-31 kg', () => {
      expect(PhysicalConstants.me).toBeCloseTo(9.1093837015e-31, 40);
    });

    test('proton mass should be approximately 1.673e-27 kg', () => {
      expect(PhysicalConstants.mp).toBeCloseTo(1.67262192369e-27, 37);
    });

    test('proton should be about 1836 times heavier than electron', () => {
      const ratio = PhysicalConstants.mp / PhysicalConstants.me;
      expect(ratio).toBeCloseTo(1836.15, 1);
    });

    test('neutron should be slightly heavier than proton', () => {
      expect(PhysicalConstants.mn).toBeGreaterThan(PhysicalConstants.mp);
    });
  });

  describe('Atomic Constants', () => {
    test('fine-structure constant should be approximately 1/137', () => {
      expect(PhysicalConstants.alpha).toBeCloseTo(1 / 137, 3);
    });

    test('Bohr radius should be approximately 5.29e-11 m', () => {
      expect(PhysicalConstants.a0).toBeCloseTo(5.29177210903e-11, 20);
    });

    test('Rydberg constant should be approximately 1.097e7 /m', () => {
      expect(PhysicalConstants.Rinf).toBeCloseTo(10973731.568160, 3);
    });
  });

  describe('Consistency Checks', () => {
    test('c² should equal 1/(ε₀μ₀)', () => {
      const cSquared = PhysicalConstants.c * PhysicalConstants.c;
      const fromEM = 1 / (PhysicalConstants.epsilon0 * PhysicalConstants.mu0);
      expect(cSquared).toBeCloseTo(fromEM, -8);
    });

    test('Bohr radius formula: a₀ = ħ/(mₑcα)', () => {
      const calculatedA0 = PhysicalConstants.hbar /
        (PhysicalConstants.me * PhysicalConstants.c * PhysicalConstants.alpha);
      expect(calculatedA0).toBeCloseTo(PhysicalConstants.a0, 12);
    });
  });
});

describe('ConstantDetails', () => {
  test('should have details for commonly used constants', () => {
    const expectedConstants = ['c', 'h', 'hbar', 'G', 'e', 'me', 'mp', 'kB', 'NA', 'alpha', 'a0'];
    for (const name of expectedConstants) {
      expect(ConstantDetails[name]).toBeDefined();
      expect(ConstantDetails[name].value).toBeDefined();
      expect(ConstantDetails[name].unit).toBeDefined();
      expect(ConstantDetails[name].description).toBeDefined();
    }
  });

  test('exact constants should have zero uncertainty', () => {
    const exactConstants = ['c', 'h', 'hbar', 'e', 'kB', 'NA'];
    for (const name of exactConstants) {
      expect(ConstantDetails[name].uncertainty).toBe(0);
    }
  });

  test('G should have non-zero uncertainty (least precisely known)', () => {
    expect(ConstantDetails.G.uncertainty).toBeGreaterThan(0);
  });
});
