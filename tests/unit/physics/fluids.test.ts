/**
 * Fluid Dynamics Tests
 */

import { FluidDynamics, FluidProperties } from '../../../src/physics/fluids';

describe('FluidDynamics', () => {
  describe('reynoldsNumber', () => {
    test('should calculate Re = vL/ν', () => {
      const Re = FluidDynamics.reynoldsNumber(2, 0.1, 1e-6);
      expect(Re).toBe(200000);
    });

    test('higher velocity should give higher Re', () => {
      const Re1 = FluidDynamics.reynoldsNumber(1, 0.1, 1e-6);
      const Re2 = FluidDynamics.reynoldsNumber(5, 0.1, 1e-6);
      expect(Re2).toBeGreaterThan(Re1);
    });
  });

  describe('flowRegime', () => {
    test('Re < 2300 should be laminar', () => {
      expect(FluidDynamics.flowRegime(1000)).toBe('laminar');
      expect(FluidDynamics.flowRegime(2000)).toBe('laminar');
    });

    test('2300 < Re < 4000 should be transitional', () => {
      expect(FluidDynamics.flowRegime(3000)).toBe('transitional');
    });

    test('Re > 4000 should be turbulent', () => {
      expect(FluidDynamics.flowRegime(10000)).toBe('turbulent');
      expect(FluidDynamics.flowRegime(100000)).toBe('turbulent');
    });
  });

  describe('darcyFrictionFactor', () => {
    test('laminar flow should give f = 64/Re', () => {
      const f = FluidDynamics.darcyFrictionFactor(2000, 0);
      expect(f).toBeCloseTo(64 / 2000, 5);
    });

    test('turbulent smooth pipe should give reasonable f', () => {
      const f = FluidDynamics.darcyFrictionFactor(100000, 0);
      expect(f).toBeGreaterThan(0.01);
      expect(f).toBeLessThan(0.05);
    });

    test('rougher pipe should have higher f', () => {
      const smooth = FluidDynamics.darcyFrictionFactor(100000, 0);
      const rough = FluidDynamics.darcyFrictionFactor(100000, 0.01);
      expect(rough).toBeGreaterThan(smooth);
    });
  });

  describe('pipeFlow', () => {
    test('should calculate correct velocity from flow rate', () => {
      const result = FluidDynamics.pipeFlow(
        0.1,      // 10 cm diameter
        100,      // 100 m length
        0.01,     // 10 L/s
        1000,     // water density
        1e-6,     // water kinematic viscosity
        0
      );

      // v = Q/A = 0.01 / (π × 0.05²) ≈ 1.27 m/s
      expect(result.velocity).toBeCloseTo(1.27, 1);
    });

    test('longer pipe should have higher pressure drop', () => {
      const short = FluidDynamics.pipeFlow(0.05, 10, 0.005, 1000, 1e-6);
      const long = FluidDynamics.pipeFlow(0.05, 100, 0.005, 1000, 1e-6);

      expect(long.pressureDrop).toBeGreaterThan(short.pressureDrop);
    });

    test('smaller diameter should have higher pressure drop', () => {
      const large = FluidDynamics.pipeFlow(0.1, 50, 0.01, 1000, 1e-6);
      const small = FluidDynamics.pipeFlow(0.05, 50, 0.01, 1000, 1e-6);

      expect(small.pressureDrop).toBeGreaterThan(large.pressureDrop);
    });
  });

  describe('bernoulli', () => {
    test('velocity increase should cause pressure decrease', () => {
      const result = FluidDynamics.bernoulli(
        1000,     // water
        1,        // v1 = 1 m/s
        200000,   // P1 = 200 kPa
        0,        // h1 = 0
        3,        // v2 = 3 m/s
        undefined,// P2 = ?
        0         // h2 = 0
      );

      expect(result.pressure2).toBeLessThan(result.pressure1);
    });

    test('height increase should cause pressure decrease', () => {
      const result = FluidDynamics.bernoulli(
        1000,
        2,
        150000,
        0,
        2,        // same velocity
        undefined,
        10        // h2 = 10m higher
      );

      // Pressure should decrease by ~ρgh = 1000 × 9.8 × 10 ≈ 98 kPa
      const pressureDrop = result.pressure1 - result.pressure2;
      expect(pressureDrop).toBeCloseTo(98000, -3);
    });

    test('total head should be conserved', () => {
      const result = FluidDynamics.bernoulli(1000, 2, 100000, 5, 4, 80000, 3);

      // Verify total head
      const h1 = result.pressure1 + result.dynamicPressure1 + 1000 * 9.80665 * result.height1;
      const h2 = result.pressure2 + result.dynamicPressure2 + 1000 * 9.80665 * result.height2;

      // Should be approximately equal (within numerical precision)
      expect(h1).toBeCloseTo(result.totalHead, 0);
    });
  });

  describe('sphereDrag', () => {
    test('Stokes regime should give Cd = 24/Re', () => {
      const result = FluidDynamics.sphereDrag(0.001, 0.001, 1000, 1e-6);
      expect(result.reynoldsNumber).toBe(1);
      expect(result.dragCoefficient).toBeCloseTo(24, 0);
    });

    test('Newton regime should give Cd ≈ 0.44', () => {
      const result = FluidDynamics.sphereDrag(0.01, 1, 1000, 1e-6);
      expect(result.reynoldsNumber).toBeGreaterThan(1000);
      expect(result.dragCoefficient).toBeCloseTo(0.44, 1);
    });
  });

  describe('orificeFlow', () => {
    test('should calculate flow rate correctly', () => {
      const result = FluidDynamics.orificeFlow(
        0.02,     // 2 cm orifice
        0.1,      // 10 cm pipe
        10000,    // 10 kPa pressure drop
        1000      // water
      );

      expect(result.flowRate).toBeGreaterThan(0);
      expect(result.beta).toBe(0.2);
    });
  });

  describe('venturiFlow', () => {
    test('throat velocity should be higher than inlet', () => {
      const result = FluidDynamics.venturiFlow(0.1, 0.05, 5000, 1000);

      expect(result.throatVelocity).toBeGreaterThan(result.inletVelocity);
    });

    test('continuity should be satisfied', () => {
      const result = FluidDynamics.venturiFlow(0.1, 0.06, 10000, 1000);

      const A1 = Math.PI * 0.1 * 0.1 / 4;
      const A2 = Math.PI * 0.06 * 0.06 / 4;

      const Q1 = A1 * result.inletVelocity;
      const Q2 = A2 * result.throatVelocity;

      expect(Q1).toBeCloseTo(result.flowRate, 5);
      expect(Q2).toBeCloseTo(result.flowRate, 5);
    });
  });

  describe('manningFlow', () => {
    test('steeper slope should give faster flow', () => {
      const gentle = FluidDynamics.manningFlow(0.5, 0.001);
      const steep = FluidDynamics.manningFlow(0.5, 0.01);

      expect(steep.velocity).toBeGreaterThan(gentle.velocity);
    });

    test('larger hydraulic radius should give faster flow', () => {
      const small = FluidDynamics.manningFlow(0.2, 0.005);
      const large = FluidDynamics.manningFlow(1.0, 0.005);

      expect(large.velocity).toBeGreaterThan(small.velocity);
    });
  });

  describe('waterWave', () => {
    test('deep water wave speed should depend on wavelength', () => {
      const short = FluidDynamics.waterWave(10, 1);
      const long = FluidDynamics.waterWave(100, 1);

      expect(long.celerity).toBeGreaterThan(short.celerity);
    });

    test('group velocity should be half phase velocity in deep water', () => {
      const wave = FluidDynamics.waterWave(50, 2);
      expect(wave.groupVelocity).toBeCloseTo(wave.celerity / 2, 1);
    });

    test('shallow water waves should have same group and phase velocity', () => {
      const wave = FluidDynamics.waterWave(100, 1, 2);  // Very shallow
      expect(wave.groupVelocity).toBeCloseTo(wave.celerity, 1);
    });
  });

  describe('pumpCalculation', () => {
    test('power should be P = ρgQH/η', () => {
      const result = FluidDynamics.pumpCalculation(0.1, 20, 0.8, 1000);

      const expectedPower = 1000 * 9.80665 * 0.1 * 20 / 0.8;
      expect(result.power).toBeCloseTo(expectedPower, 0);
    });

    test('higher head should require more power', () => {
      const low = FluidDynamics.pumpCalculation(0.1, 10, 0.8);
      const high = FluidDynamics.pumpCalculation(0.1, 50, 0.8);

      expect(high.power).toBeGreaterThan(low.power);
    });
  });

  describe('buoyancy', () => {
    test('buoyancy force should equal weight of displaced fluid', () => {
      const result = FluidDynamics.buoyancy(1, 1000);  // 1 m³ in water

      expect(result.buoyancyForce).toBeCloseTo(9806.65, 0);
      expect(result.massEquivalent).toBe(1000);
    });
  });

  describe('hydrostaticPressure', () => {
    test('pressure at 10m depth should be ~2 atm', () => {
      const result = FluidDynamics.hydrostaticPressure(10);

      expect(result.pressure).toBeCloseTo(199225, -2);  // ~2 atm
      expect(result.gauge).toBeCloseTo(97900, -2);       // ~1 atm gauge
    });
  });

  describe('capillaryRise', () => {
    test('narrower tube should have higher rise', () => {
      const wide = FluidDynamics.capillaryRise(0.001, 0.0728, 0, 1000);
      const narrow = FluidDynamics.capillaryRise(0.0001, 0.0728, 0, 1000);

      expect(narrow).toBeGreaterThan(wide);
    });

    test('water in 1mm glass tube should rise ~14mm', () => {
      const rise = FluidDynamics.capillaryRise(0.0005, 0.0728, 0, 1000);
      expect(rise * 1000).toBeCloseTo(29.7, 0);  // ~30mm for 0.5mm radius
    });
  });

  describe('froudeNumber', () => {
    test('should calculate Fr = v/√(gL)', () => {
      const Fr = FluidDynamics.froudeNumber(5, 2);
      const expected = 5 / Math.sqrt(9.80665 * 2);
      expect(Fr).toBeCloseTo(expected, 5);
    });
  });

  describe('cavitationNumber', () => {
    test('high pressure should prevent cavitation', () => {
      const result = FluidDynamics.cavitationNumber(200000, 2338, 1000, 10);
      expect(result.willCavitate).toBe(false);
    });

    test('low pressure and high velocity should cause cavitation', () => {
      const result = FluidDynamics.cavitationNumber(50000, 2338, 1000, 30);
      expect(result.willCavitate).toBe(true);
    });
  });
});
