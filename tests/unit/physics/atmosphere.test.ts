/**
 * Atmospheric Physics Tests
 */

import { AtmosphericPhysics } from '../../../src/physics/atmosphere';

describe('AtmosphericPhysics', () => {
  describe('getAtmosphericState', () => {
    test('sea level should have standard conditions', () => {
      const state = AtmosphericPhysics.getAtmosphericState(0);

      expect(state.temperature).toBeCloseTo(288.15, 1);
      expect(state.pressure).toBeCloseTo(101325, -1);
      expect(state.density).toBeCloseTo(1.225, 2);
      expect(state.layer).toBe('Troposphere');
    });

    test('temperature should decrease in troposphere', () => {
      const low = AtmosphericPhysics.getAtmosphericState(0);
      const mid = AtmosphericPhysics.getAtmosphericState(5000);
      const high = AtmosphericPhysics.getAtmosphericState(10000);

      expect(mid.temperature).toBeLessThan(low.temperature);
      expect(high.temperature).toBeLessThan(mid.temperature);
    });

    test('pressure should decrease with altitude', () => {
      const p0 = AtmosphericPhysics.getAtmosphericState(0).pressure;
      const p5 = AtmosphericPhysics.getAtmosphericState(5000).pressure;
      const p10 = AtmosphericPhysics.getAtmosphericState(10000).pressure;

      expect(p5).toBeLessThan(p0);
      expect(p10).toBeLessThan(p5);
    });

    test('tropopause should be near 11km', () => {
      const state = AtmosphericPhysics.getAtmosphericState(11000);
      expect(state.temperature).toBeCloseTo(216.65, 0);
    });

    test('stratosphere should be isothermal in lower part', () => {
      const s12 = AtmosphericPhysics.getAtmosphericState(12000);
      const s20 = AtmosphericPhysics.getAtmosphericState(20000);

      expect(s12.temperature).toBeCloseTo(s20.temperature, 0);
      expect(s12.layer).toContain('Stratosphere');
    });

    test('speed of sound should decrease with altitude in troposphere', () => {
      const a0 = AtmosphericPhysics.getAtmosphericState(0).speedOfSound;
      const a10k = AtmosphericPhysics.getAtmosphericState(10000).speedOfSound;

      expect(a0).toBeCloseTo(340, 0);
      expect(a10k).toBeLessThan(a0);
    });
  });

  describe('calculateAerodynamics', () => {
    test('dynamic pressure should follow q = 0.5ρv²', () => {
      const result = AtmosphericPhysics.calculateAerodynamics(
        100,    // 100 m/s
        0,      // sea level
        1,      // 1 m² area
        0.5     // Cd
      );

      // q = 0.5 × 1.225 × 100² = 6125 Pa
      expect(result.dynamicPressure).toBeCloseTo(6125, -1);
    });

    test('drag force should be D = qACd', () => {
      const result = AtmosphericPhysics.calculateAerodynamics(
        50,     // velocity
        0,      // sea level
        2,      // 2 m² area
        0.3     // Cd
      );

      const expectedQ = 0.5 * 1.225 * 50 * 50;
      const expectedD = expectedQ * 2 * 0.3;

      expect(result.drag).toBeCloseTo(expectedD, 0);
    });

    test('Mach number should be v/a', () => {
      const state = AtmosphericPhysics.getAtmosphericState(0);
      const velocity = state.speedOfSound * 0.8;  // Mach 0.8

      const result = AtmosphericPhysics.calculateAerodynamics(
        velocity, 0, 1, 0.5
      );

      expect(result.machNumber).toBeCloseTo(0.8, 2);
    });
  });

  describe('machDragCoefficient', () => {
    test('subsonic should increase slightly with Mach', () => {
      const cd0 = AtmosphericPhysics.machDragCoefficient(0.3, 0);
      const cd05 = AtmosphericPhysics.machDragCoefficient(0.3, 0.5);

      expect(cd05).toBeGreaterThan(cd0);
    });

    test('transonic should have peak drag', () => {
      const cdSubsonic = AtmosphericPhysics.machDragCoefficient(0.3, 0.7);
      const cdTransonic = AtmosphericPhysics.machDragCoefficient(0.3, 1.0);
      const cdSupersonic = AtmosphericPhysics.machDragCoefficient(0.3, 2.0);

      expect(cdTransonic).toBeGreaterThan(cdSubsonic);
      expect(cdTransonic).toBeGreaterThan(cdSupersonic);
    });
  });

  describe('calculateReentryHeating', () => {
    test('high velocity should produce high heat flux', () => {
      const result = AtmosphericPhysics.calculateReentryHeating(
        7000,   // 7 km/s
        60000,  // 60 km altitude
        0.5,    // 0.5 m nose radius
        100     // 100 kg mass
      );

      expect(result.heatFlux).toBeGreaterThan(1e6);  // MW/m² range
    });

    test('heat flux should scale with v³', () => {
      const result1 = AtmosphericPhysics.calculateReentryHeating(4000, 60000, 0.5, 100);
      const result2 = AtmosphericPhysics.calculateReentryHeating(8000, 60000, 0.5, 100);

      // 8000³ / 4000³ = 8
      expect(result2.heatFlux / result1.heatFlux).toBeCloseTo(8, 0);
    });

    test('larger nose radius should reduce heating', () => {
      const small = AtmosphericPhysics.calculateReentryHeating(6000, 50000, 0.3, 100);
      const large = AtmosphericPhysics.calculateReentryHeating(6000, 50000, 1.0, 100);

      expect(large.heatFlux).toBeLessThan(small.heatFlux);
    });
  });

  describe('terminalVelocity', () => {
    test('skydiver should have terminal velocity ~50-60 m/s', () => {
      const v = AtmosphericPhysics.terminalVelocity(
        80,     // 80 kg person
        1.0,    // belly-down Cd
        0.7,    // ~0.7 m² cross section
        2000    // 2000 m altitude
      );

      expect(v).toBeGreaterThan(40);
      expect(v).toBeLessThan(70);
    });

    test('heavier object should fall faster', () => {
      const light = AtmosphericPhysics.terminalVelocity(50, 0.5, 0.5);
      const heavy = AtmosphericPhysics.terminalVelocity(100, 0.5, 0.5);

      expect(heavy).toBeGreaterThan(light);
    });
  });

  describe('simulateBallisticTrajectory', () => {
    test('trajectory should end at ground level', () => {
      const trajectory = AtmosphericPhysics.simulateBallisticTrajectory(
        10000,      // 10 km initial altitude
        500,        // 500 m/s velocity
        -0.5,       // 30° descent
        100,        // 100 kg
        0.1,        // 0.1 m² area
        0.3,        // Cd
        0.1,        // 0.1s timestep
        1000        // max 1000s
      );

      expect(trajectory.length).toBeGreaterThan(0);
      expect(trajectory[trajectory.length - 1].altitude).toBeLessThanOrEqual(0);
    });

    test('velocity should decrease due to drag', () => {
      const trajectory = AtmosphericPhysics.simulateBallisticTrajectory(
        5000, 400, -0.3, 50, 0.2, 0.5, 0.1, 500
      );

      const initial = trajectory[0].velocity;
      const final = trajectory[trajectory.length - 1].velocity;

      expect(final).toBeLessThan(initial);
    });
  });

  describe('pressureAltitude', () => {
    test('standard pressure should give sea level', () => {
      const altitude = AtmosphericPhysics.pressureAltitude(101325);
      expect(altitude).toBeCloseTo(0, 0);
    });

    test('lower pressure should give higher altitude', () => {
      const alt1 = AtmosphericPhysics.pressureAltitude(90000);
      const alt2 = AtmosphericPhysics.pressureAltitude(70000);

      expect(alt2).toBeGreaterThan(alt1);
    });
  });

  describe('windGradient', () => {
    test('wind should increase with height', () => {
      const w10 = AtmosphericPhysics.windGradient(10, 10);
      const w100 = AtmosphericPhysics.windGradient(10, 100);

      expect(w100).toBeGreaterThan(w10);
    });

    test('wind at reference height should equal surface wind', () => {
      const w = AtmosphericPhysics.windGradient(15, 10);
      expect(w).toBeCloseTo(15, 1);
    });
  });

  describe('airDensity', () => {
    test('standard conditions should give ~1.2 kg/m³', () => {
      const density = AtmosphericPhysics.airDensity(288.15, 101325, 0);
      expect(density).toBeCloseTo(1.225, 2);
    });

    test('higher temperature should give lower density', () => {
      const cold = AtmosphericPhysics.airDensity(273, 101325);
      const hot = AtmosphericPhysics.airDensity(313, 101325);

      expect(hot).toBeLessThan(cold);
    });

    test('humidity should slightly reduce density', () => {
      const dry = AtmosphericPhysics.airDensity(293, 101325, 0);
      const humid = AtmosphericPhysics.airDensity(293, 101325, 0.9);

      expect(humid).toBeLessThan(dry);
    });
  });
});
