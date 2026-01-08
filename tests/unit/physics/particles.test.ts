/**
 * Particle Dynamics Tests
 */

import { ParticleDynamics, Vector3D } from '../../../src/physics/particles';
import { PhysicalConstants } from '../../../src/physics/constants';

describe('ParticleDynamics', () => {
  describe('Vector Operations', () => {
    test('vectorAdd should add vectors correctly', () => {
      const a: Vector3D = { x: 1, y: 2, z: 3 };
      const b: Vector3D = { x: 4, y: 5, z: 6 };
      const result = ParticleDynamics.vectorAdd(a, b);
      expect(result).toEqual({ x: 5, y: 7, z: 9 });
    });

    test('vectorSubtract should subtract vectors correctly', () => {
      const a: Vector3D = { x: 5, y: 7, z: 9 };
      const b: Vector3D = { x: 1, y: 2, z: 3 };
      const result = ParticleDynamics.vectorSubtract(a, b);
      expect(result).toEqual({ x: 4, y: 5, z: 6 });
    });

    test('vectorMagnitude should calculate correct length', () => {
      const v: Vector3D = { x: 3, y: 4, z: 0 };
      expect(ParticleDynamics.vectorMagnitude(v)).toBe(5);
    });

    test('vectorDot should calculate dot product', () => {
      const a: Vector3D = { x: 1, y: 2, z: 3 };
      const b: Vector3D = { x: 4, y: 5, z: 6 };
      expect(ParticleDynamics.vectorDot(a, b)).toBe(32);
    });

    test('vectorCross should calculate cross product', () => {
      const a: Vector3D = { x: 1, y: 0, z: 0 };
      const b: Vector3D = { x: 0, y: 1, z: 0 };
      const result = ParticleDynamics.vectorCross(a, b);
      expect(result).toEqual({ x: 0, y: 0, z: 1 });
    });

    test('vectorNormalize should return unit vector', () => {
      const v: Vector3D = { x: 3, y: 4, z: 0 };
      const normalized = ParticleDynamics.vectorNormalize(v);
      expect(ParticleDynamics.vectorMagnitude(normalized)).toBeCloseTo(1, 10);
    });
  });

  describe('gravitationalForce', () => {
    test('Earth-Sun gravitational force should be ~3.5e22 N', () => {
      const earthMass = 5.972e24;
      const sunMass = 1.989e30;
      const r: Vector3D = { x: 1.496e11, y: 0, z: 0 }; // 1 AU

      const force = ParticleDynamics.gravitationalForce(earthMass, sunMass, r);
      const magnitude = ParticleDynamics.vectorMagnitude(force);

      expect(magnitude).toBeCloseTo(3.54e22, 20);
    });

    test('force should be attractive (negative direction)', () => {
      const r: Vector3D = { x: 1e6, y: 0, z: 0 };
      const force = ParticleDynamics.gravitationalForce(1e10, 1e10, r);
      expect(force.x).toBeLessThan(0);
    });

    test('force should follow inverse square law', () => {
      const m1 = 1e20, m2 = 1e20;
      const r1: Vector3D = { x: 1e6, y: 0, z: 0 };
      const r2: Vector3D = { x: 2e6, y: 0, z: 0 };

      const F1 = ParticleDynamics.vectorMagnitude(ParticleDynamics.gravitationalForce(m1, m2, r1));
      const F2 = ParticleDynamics.vectorMagnitude(ParticleDynamics.gravitationalForce(m1, m2, r2));

      expect(F1 / F2).toBeCloseTo(4, 10);
    });
  });

  describe('electrostaticForce', () => {
    test('two protons at 1e-15 m should feel ~230 N', () => {
      const q = PhysicalConstants.e;
      const r: Vector3D = { x: 1e-15, y: 0, z: 0 };

      const force = ParticleDynamics.electrostaticForce(q, q, r);
      const magnitude = ParticleDynamics.vectorMagnitude(force);

      expect(magnitude).toBeCloseTo(230, 0);
    });

    test('opposite charges should attract', () => {
      const r: Vector3D = { x: 1e-10, y: 0, z: 0 };
      const force = ParticleDynamics.electrostaticForce(PhysicalConstants.e, -PhysicalConstants.e, r);
      expect(force.x).toBeLessThan(0);
    });

    test('like charges should repel', () => {
      const r: Vector3D = { x: 1e-10, y: 0, z: 0 };
      const force = ParticleDynamics.electrostaticForce(PhysicalConstants.e, PhysicalConstants.e, r);
      expect(force.x).toBeGreaterThan(0);
    });
  });

  describe('lorentzForce', () => {
    test('stationary charge in magnetic field should feel no force', () => {
      const force = ParticleDynamics.lorentzForce(
        PhysicalConstants.e,
        { x: 0, y: 0, z: 0 },
        { x: 0, y: 0, z: 0 },
        { x: 0, y: 0, z: 1 }
      );
      expect(ParticleDynamics.vectorMagnitude(force)).toBe(0);
    });

    test('velocity parallel to B should give no magnetic force', () => {
      const force = ParticleDynamics.lorentzForce(
        PhysicalConstants.e,
        { x: 0, y: 0, z: 1e6 },
        { x: 0, y: 0, z: 0 },
        { x: 0, y: 0, z: 1 }
      );
      expect(ParticleDynamics.vectorMagnitude(force)).toBeCloseTo(0, 10);
    });

    test('velocity perpendicular to B should give maximum magnetic force', () => {
      const v = 1e6;
      const B = 1;
      const force = ParticleDynamics.lorentzForce(
        PhysicalConstants.e,
        { x: v, y: 0, z: 0 },
        { x: 0, y: 0, z: 0 },
        { x: 0, y: 0, z: B }
      );
      const expectedMagnitude = PhysicalConstants.e * v * B;
      expect(ParticleDynamics.vectorMagnitude(force)).toBeCloseTo(expectedMagnitude, 25);
    });
  });

  describe('calculateOrbitalElements', () => {
    test('circular orbit should have eccentricity near 0', () => {
      const sunMass = 1.989e30;
      const earthMass = 5.972e24;
      const r = 1.496e11; // 1 AU
      const v = Math.sqrt(PhysicalConstants.G * sunMass / r); // Circular velocity

      const elements = ParticleDynamics.calculateOrbitalElements(
        sunMass,
        earthMass,
        { x: r, y: 0, z: 0 },
        { x: 0, y: v, z: 0 }
      );

      expect(elements.eccentricity).toBeCloseTo(0, 5);
    });

    test('Earth orbit should have period ~1 year', () => {
      const sunMass = 1.989e30;
      const earthMass = 5.972e24;
      const r = 1.496e11;
      const v = Math.sqrt(PhysicalConstants.G * sunMass / r);

      const elements = ParticleDynamics.calculateOrbitalElements(
        sunMass,
        earthMass,
        { x: r, y: 0, z: 0 },
        { x: 0, y: v, z: 0 }
      );

      const periodDays = elements.orbitalPeriod / (24 * 3600);
      expect(periodDays).toBeCloseTo(365, 0);
    });
  });

  describe('nBodyStep', () => {
    test('total momentum should be conserved', () => {
      const particles = [
        { mass: 1e10, charge: 0, position: { x: 0, y: 0, z: 0 }, velocity: { x: 1, y: 0, z: 0 } },
        { mass: 1e10, charge: 0, position: { x: 1e6, y: 0, z: 0 }, velocity: { x: -1, y: 0, z: 0 } },
      ];

      const initialMomentum = ParticleDynamics.totalMomentum(particles);
      const newParticles = ParticleDynamics.nBodyStep(particles, 1);
      const finalMomentum = ParticleDynamics.totalMomentum(newParticles);

      expect(finalMomentum.x).toBeCloseTo(initialMomentum.x, 5);
      expect(finalMomentum.y).toBeCloseTo(initialMomentum.y, 5);
      expect(finalMomentum.z).toBeCloseTo(initialMomentum.z, 5);
    });
  });

  describe('elasticCollision', () => {
    test('equal masses should exchange velocities', () => {
      const result = ParticleDynamics.elasticCollision(1, 5, 1, 0);
      expect(result.v1Final).toBeCloseTo(0, 10);
      expect(result.v2Final).toBeCloseTo(5, 10);
    });

    test('momentum should be conserved', () => {
      const m1 = 2, v1 = 3, m2 = 1, v2 = -1;
      const result = ParticleDynamics.elasticCollision(m1, v1, m2, v2);

      const initialMomentum = m1 * v1 + m2 * v2;
      const finalMomentum = m1 * result.v1Final + m2 * result.v2Final;

      expect(finalMomentum).toBeCloseTo(initialMomentum, 10);
    });

    test('kinetic energy should be conserved', () => {
      const m1 = 2, v1 = 3, m2 = 1, v2 = -1;
      const result = ParticleDynamics.elasticCollision(m1, v1, m2, v2);

      const initialKE = 0.5 * m1 * v1 * v1 + 0.5 * m2 * v2 * v2;
      const finalKE = 0.5 * m1 * result.v1Final * result.v1Final + 0.5 * m2 * result.v2Final * result.v2Final;

      expect(finalKE).toBeCloseTo(initialKE, 10);
    });
  });

  describe('inelasticCollision', () => {
    test('perfectly inelastic (e=0) should give maximum energy loss', () => {
      const result = ParticleDynamics.inelasticCollision(1, 5, 1, -5, 0);
      expect(result.v1Final).toBeCloseTo(result.v2Final, 10);
      expect(result.energyLoss).toBeGreaterThan(0);
    });

    test('coefficient of restitution 1 should behave like elastic', () => {
      const m1 = 2, v1 = 3, m2 = 1, v2 = -1;
      const inelastic = ParticleDynamics.inelasticCollision(m1, v1, m2, v2, 1);
      const elastic = ParticleDynamics.elasticCollision(m1, v1, m2, v2);

      expect(inelastic.v1Final).toBeCloseTo(elastic.v1Final, 10);
      expect(inelastic.v2Final).toBeCloseTo(elastic.v2Final, 10);
    });

    test('momentum should always be conserved', () => {
      const m1 = 3, v1 = 4, m2 = 2, v2 = -2;
      const result = ParticleDynamics.inelasticCollision(m1, v1, m2, v2, 0.5);

      const initialMomentum = m1 * v1 + m2 * v2;
      const finalMomentum = m1 * result.v1Final + m2 * result.v2Final;

      expect(finalMomentum).toBeCloseTo(initialMomentum, 10);
    });
  });

  describe('calculateRelativistic', () => {
    test('low velocity should give lorentz factor ~1', () => {
      const result = ParticleDynamics.calculateRelativistic(PhysicalConstants.me, 1000);
      expect(result.lorentzFactor).toBeCloseTo(1, 10);
    });

    test('0.6c should give lorentz factor = 1.25', () => {
      const v = 0.6 * PhysicalConstants.c;
      const result = ParticleDynamics.calculateRelativistic(PhysicalConstants.me, v);
      expect(result.lorentzFactor).toBeCloseTo(1.25, 5);
    });

    test('0.99c should give high lorentz factor', () => {
      const v = 0.99 * PhysicalConstants.c;
      const result = ParticleDynamics.calculateRelativistic(PhysicalConstants.me, v);
      expect(result.lorentzFactor).toBeGreaterThan(7);
    });

    test('should throw for velocity >= c', () => {
      expect(() => ParticleDynamics.calculateRelativistic(PhysicalConstants.me, PhysicalConstants.c)).toThrow();
      expect(() => ParticleDynamics.calculateRelativistic(PhysicalConstants.me, 1.1 * PhysicalConstants.c)).toThrow();
    });

    test('total energy should equal rest energy + kinetic energy', () => {
      const v = 0.5 * PhysicalConstants.c;
      const result = ParticleDynamics.calculateRelativistic(PhysicalConstants.me, v);
      const restEnergy = PhysicalConstants.me * PhysicalConstants.c * PhysicalConstants.c;
      expect(result.totalEnergy).toBeCloseTo(restEnergy + result.kineticEnergy, 15);
    });
  });

  describe('escapeVelocity', () => {
    test('Earth escape velocity should be ~11.2 km/s', () => {
      const earthMass = 5.972e24;
      const earthRadius = 6.371e6;
      const vEscape = ParticleDynamics.escapeVelocity(earthMass, earthRadius);
      expect(vEscape / 1000).toBeCloseTo(11.2, 0);
    });
  });

  describe('schwarzschildRadius', () => {
    test('Sun schwarzschild radius should be ~3 km', () => {
      const sunMass = 1.989e30;
      const rs = ParticleDynamics.schwarzschildRadius(sunMass);
      expect(rs / 1000).toBeCloseTo(2.95, 0);
    });

    test('should satisfy rs = 2GM/c²', () => {
      const mass = 1e30;
      const rs = ParticleDynamics.schwarzschildRadius(mass);
      const expected = 2 * PhysicalConstants.G * mass / (PhysicalConstants.c * PhysicalConstants.c);
      expect(rs).toBeCloseTo(expected, 5);
    });
  });

  describe('pendulumPeriod', () => {
    test('1m pendulum should have period ~2s', () => {
      const period = ParticleDynamics.pendulumPeriod(1);
      expect(period).toBeCloseTo(2.006, 2);
    });

    test('period should scale as sqrt(L)', () => {
      const T1 = ParticleDynamics.pendulumPeriod(1);
      const T4 = ParticleDynamics.pendulumPeriod(4);
      expect(T4 / T1).toBeCloseTo(2, 5);
    });
  });
});
