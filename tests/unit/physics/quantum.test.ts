/**
 * Quantum Mechanics Tests
 */

import { QuantumMechanics } from '../../../src/physics/quantum';
import { PhysicalConstants } from '../../../src/physics/constants';

describe('QuantumMechanics', () => {
  describe('calculatePhotonProperties', () => {
    test('visible light (550nm) should have energy ~3.6e-19 J', () => {
      const photon = QuantumMechanics.calculatePhotonProperties(550);
      expect(photon.energy).toBeCloseTo(3.61e-19, 21);
    });

    test('wavelength and frequency should satisfy c = λf', () => {
      const photon = QuantumMechanics.calculatePhotonProperties(500);
      const calculatedC = photon.wavelength * photon.frequency;
      expect(calculatedC).toBeCloseTo(PhysicalConstants.c, 3);
    });

    test('energy should satisfy E = hf', () => {
      const photon = QuantumMechanics.calculatePhotonProperties(600);
      const expectedEnergy = PhysicalConstants.h * photon.frequency;
      expect(photon.energy).toBeCloseTo(expectedEnergy, 30);
    });

    test('momentum should satisfy p = h/λ', () => {
      const photon = QuantumMechanics.calculatePhotonProperties(700);
      const expectedMomentum = PhysicalConstants.h / photon.wavelength;
      expect(photon.momentum).toBeCloseTo(expectedMomentum, 35);
    });

    test('X-ray (0.1nm) should have much higher energy than visible light', () => {
      const xray = QuantumMechanics.calculatePhotonProperties(0.1);
      const visible = QuantumMechanics.calculatePhotonProperties(500);
      expect(xray.energy).toBeGreaterThan(visible.energy * 1000);
    });
  });

  describe('hydrogenEnergyLevel', () => {
    test('ground state (n=1) should be approximately -13.6 eV', () => {
      const E1 = QuantumMechanics.hydrogenEnergyLevel(1);
      const E1_eV = E1 / PhysicalConstants.eV_to_J;
      expect(E1_eV).toBeCloseTo(-13.6, 0);
    });

    test('first excited state (n=2) should be -3.4 eV', () => {
      const E2 = QuantumMechanics.hydrogenEnergyLevel(2);
      const E2_eV = E2 / PhysicalConstants.eV_to_J;
      expect(E2_eV).toBeCloseTo(-3.4, 0);
    });

    test('energy should scale as 1/n²', () => {
      const E1 = QuantumMechanics.hydrogenEnergyLevel(1);
      const E2 = QuantumMechanics.hydrogenEnergyLevel(2);
      const E3 = QuantumMechanics.hydrogenEnergyLevel(3);

      expect(E2 / E1).toBeCloseTo(1 / 4, 10);
      expect(E3 / E1).toBeCloseTo(1 / 9, 10);
    });

    test('should throw for invalid quantum number', () => {
      expect(() => QuantumMechanics.hydrogenEnergyLevel(0)).toThrow();
      expect(() => QuantumMechanics.hydrogenEnergyLevel(-1)).toThrow();
      expect(() => QuantumMechanics.hydrogenEnergyLevel(1.5)).toThrow();
    });
  });

  describe('hydrogenTransition', () => {
    test('Balmer alpha (3→2) should emit visible red light (~656nm)', () => {
      const transition = QuantumMechanics.hydrogenTransition(3, 2);
      const wavelengthNm = transition.wavelength * 1e9;
      expect(wavelengthNm).toBeCloseTo(656, 0);
      expect(transition.seriesName).toBe('Balmer');
    });

    test('Lyman alpha (2→1) should emit UV (~121nm)', () => {
      const transition = QuantumMechanics.hydrogenTransition(2, 1);
      const wavelengthNm = transition.wavelength * 1e9;
      expect(wavelengthNm).toBeCloseTo(121.5, 0);
      expect(transition.seriesName).toBe('Lyman');
    });

    test('energy difference should equal photon energy', () => {
      const transition = QuantumMechanics.hydrogenTransition(4, 2);
      const E4 = QuantumMechanics.hydrogenEnergyLevel(4);
      const E2 = QuantumMechanics.hydrogenEnergyLevel(2);
      expect(transition.energyDifference).toBeCloseTo(Math.abs(E4 - E2), 25);
    });
  });

  describe('calculateUncertainty', () => {
    test('should satisfy Heisenberg uncertainty principle', () => {
      const result = QuantumMechanics.calculateUncertainty(1e-10);
      expect(result.product).toBeGreaterThanOrEqual(result.minimumProduct);
      expect(result.isValid).toBe(true);
    });

    test('smaller position uncertainty should give larger momentum uncertainty', () => {
      const small = QuantumMechanics.calculateUncertainty(1e-12);
      const large = QuantumMechanics.calculateUncertainty(1e-10);
      expect(small.deltaP).toBeGreaterThan(large.deltaP);
    });

    test('minimum product should be ħ/2', () => {
      const result = QuantumMechanics.calculateUncertainty(1e-9);
      expect(result.minimumProduct).toBeCloseTo(PhysicalConstants.hbar / 2, 44);
    });
  });

  describe('quantumTunneling', () => {
    test('transmission should be less than 1 for barrier above particle energy', () => {
      const result = QuantumMechanics.quantumTunneling(
        PhysicalConstants.me,
        1e-19,  // particle energy
        2e-19,  // barrier height
        1e-10   // barrier width
      );
      expect(result.transmissionCoefficient).toBeLessThan(1);
      expect(result.transmissionCoefficient).toBeGreaterThan(0);
    });

    test('wider barrier should give lower transmission', () => {
      const narrow = QuantumMechanics.quantumTunneling(
        PhysicalConstants.me, 1e-19, 2e-19, 1e-10
      );
      const wide = QuantumMechanics.quantumTunneling(
        PhysicalConstants.me, 1e-19, 2e-19, 2e-10
      );
      expect(wide.transmissionCoefficient).toBeLessThan(narrow.transmissionCoefficient);
    });

    test('transmission + reflection should equal 1', () => {
      const result = QuantumMechanics.quantumTunneling(
        PhysicalConstants.me, 1e-19, 3e-19, 5e-11
      );
      expect(result.transmissionCoefficient + result.reflectionCoefficient).toBeCloseTo(1, 10);
    });

    test('particle energy above barrier should give full transmission', () => {
      const result = QuantumMechanics.quantumTunneling(
        PhysicalConstants.me, 3e-19, 2e-19, 1e-10
      );
      expect(result.transmissionCoefficient).toBe(1);
    });
  });

  describe('harmonicOscillator', () => {
    test('ground state energy should be ħω/2', () => {
      const omega = 1e15;
      const result = QuantumMechanics.harmonicOscillator(PhysicalConstants.me, omega, 0);
      expect(result.energy).toBeCloseTo(PhysicalConstants.hbar * omega / 2, 30);
      expect(result.zeroPointEnergy).toBeCloseTo(result.energy, 30);
    });

    test('energy levels should be equally spaced', () => {
      const omega = 1e14;
      const E0 = QuantumMechanics.harmonicOscillator(PhysicalConstants.me, omega, 0).energy;
      const E1 = QuantumMechanics.harmonicOscillator(PhysicalConstants.me, omega, 1).energy;
      const E2 = QuantumMechanics.harmonicOscillator(PhysicalConstants.me, omega, 2).energy;

      const spacing1 = E1 - E0;
      const spacing2 = E2 - E1;
      expect(spacing1).toBeCloseTo(PhysicalConstants.hbar * omega, 30);
      expect(spacing2).toBeCloseTo(spacing1, 30);
    });

    test('expectation value of x should be zero (symmetric)', () => {
      const result = QuantumMechanics.harmonicOscillator(PhysicalConstants.me, 1e15, 5);
      expect(result.expectationX).toBe(0);
    });

    test('should throw for negative quantum number', () => {
      expect(() => QuantumMechanics.harmonicOscillator(PhysicalConstants.me, 1e15, -1)).toThrow();
    });
  });

  describe('particleInBox', () => {
    test('ground state should have energy E₁ = π²ħ²/(2mL²)', () => {
      const L = 1e-9;
      const result = QuantumMechanics.particleInBox(PhysicalConstants.me, L, 1);
      const expected = Math.PI * Math.PI * PhysicalConstants.hbar * PhysicalConstants.hbar /
        (2 * PhysicalConstants.me * L * L);
      expect(result.energy).toBeCloseTo(expected, 20);
    });

    test('energy should scale as n²', () => {
      const L = 1e-9;
      const E1 = QuantumMechanics.particleInBox(PhysicalConstants.me, L, 1).energy;
      const E2 = QuantumMechanics.particleInBox(PhysicalConstants.me, L, 2).energy;
      const E3 = QuantumMechanics.particleInBox(PhysicalConstants.me, L, 3).energy;

      expect(E2 / E1).toBeCloseTo(4, 10);
      expect(E3 / E1).toBeCloseTo(9, 10);
    });

    test('number of nodes should be n-1', () => {
      expect(QuantumMechanics.particleInBox(PhysicalConstants.me, 1e-9, 1).nodesCount).toBe(0);
      expect(QuantumMechanics.particleInBox(PhysicalConstants.me, 1e-9, 3).nodesCount).toBe(2);
      expect(QuantumMechanics.particleInBox(PhysicalConstants.me, 1e-9, 5).nodesCount).toBe(4);
    });
  });

  describe('deBroglieWavelength', () => {
    test('electron at 1e6 m/s should have wavelength ~7e-10 m', () => {
      const lambda = QuantumMechanics.deBroglieWavelength(PhysicalConstants.me, 1e6);
      expect(lambda).toBeCloseTo(7.27e-10, 12);
    });

    test('wavelength should satisfy λ = h/mv', () => {
      const m = PhysicalConstants.me;
      const v = 5e5;
      const lambda = QuantumMechanics.deBroglieWavelength(m, v);
      expect(lambda).toBeCloseTo(PhysicalConstants.h / (m * v), 35);
    });

    test('heavier particle should have shorter wavelength', () => {
      const v = 1e6;
      const electronLambda = QuantumMechanics.deBroglieWavelength(PhysicalConstants.me, v);
      const protonLambda = QuantumMechanics.deBroglieWavelength(PhysicalConstants.mp, v);
      expect(protonLambda).toBeLessThan(electronLambda);
    });
  });

  describe('spinOrbitCoupling', () => {
    test('s orbitals (l=0) should have no spin-orbit coupling', () => {
      const energy = QuantumMechanics.spinOrbitCoupling(2, 0, 0.5);
      expect(energy).toBe(0);
    });

    test('p orbitals should have non-zero spin-orbit coupling', () => {
      const energyJ32 = QuantumMechanics.spinOrbitCoupling(2, 1, 1.5);
      const energyJ12 = QuantumMechanics.spinOrbitCoupling(2, 1, 0.5);
      expect(energyJ32).not.toBe(0);
      expect(energyJ12).not.toBe(0);
      expect(energyJ32).not.toEqual(energyJ12);
    });
  });

  describe('hydrogenRadialWavefunction', () => {
    test('1s orbital should have maximum at r=0', () => {
      const rValues = [0.1, 0.5, 1, 2, 3, 4, 5];
      const result = QuantumMechanics.hydrogenRadialWavefunction(1, 0, rValues);

      // R²(r) * r² peaks at r = a₀ for 1s
      // But R(r) itself is maximum at r=0
      expect(result.R[0]).toBeGreaterThan(result.R[3]);
    });

    test('should throw for invalid quantum numbers', () => {
      expect(() => QuantumMechanics.hydrogenRadialWavefunction(1, 1, [1])).toThrow();
      expect(() => QuantumMechanics.hydrogenRadialWavefunction(2, 2, [1])).toThrow();
      expect(() => QuantumMechanics.hydrogenRadialWavefunction(2, -1, [1])).toThrow();
    });
  });
});
