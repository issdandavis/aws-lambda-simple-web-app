/**
 * Simulation Engine Tests
 */

import { SimulationEngine } from '../../../src/lambda/simulation-engine';
import { SimulationRequest } from '../../../src/lambda/types';

describe('SimulationEngine', () => {
  let engine: SimulationEngine;

  beforeEach(() => {
    engine = new SimulationEngine();
  });

  describe('Quantum Simulations', () => {
    test('should execute photon_properties simulation', async () => {
      const request: SimulationRequest = {
        simulationType: 'quantum',
        operation: 'photon_properties',
        parameters: { wavelengthNm: 550 },
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(true);
      expect(result.result).toHaveProperty('energy');
      expect(result.result).toHaveProperty('frequency');
      expect(result.result).toHaveProperty('wavelength');
      expect(result.result).toHaveProperty('momentum');
    });

    test('should execute hydrogen_energy simulation', async () => {
      const request: SimulationRequest = {
        simulationType: 'quantum',
        operation: 'hydrogen_energy',
        parameters: { n: 2 },
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(true);
      expect(result.result).toHaveProperty('n', 2);
      expect(result.result).toHaveProperty('energy');
      expect(result.result).toHaveProperty('energyEV');
    });

    test('should execute hydrogen_transition simulation', async () => {
      const request: SimulationRequest = {
        simulationType: 'quantum',
        operation: 'hydrogen_transition',
        parameters: { nInitial: 3, nFinal: 2 },
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(true);
      expect(result.result).toHaveProperty('energyDifference');
      expect(result.result).toHaveProperty('wavelength');
      expect(result.result).toHaveProperty('seriesName', 'Balmer');
    });

    test('should execute tunneling simulation', async () => {
      const request: SimulationRequest = {
        simulationType: 'quantum',
        operation: 'tunneling',
        parameters: {
          particleMass: 9.1e-31,
          particleEnergy: 1e-19,
          barrierHeight: 2e-19,
          barrierWidth: 1e-10,
        },
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(true);
      expect(result.result).toHaveProperty('transmissionCoefficient');
      expect(result.result).toHaveProperty('penetrationDepth');
    });

    test('should execute harmonic_oscillator simulation', async () => {
      const request: SimulationRequest = {
        simulationType: 'quantum',
        operation: 'harmonic_oscillator',
        parameters: {
          mass: 9.1e-31,
          angularFrequency: 1e15,
          n: 3,
        },
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(true);
      expect(result.result).toHaveProperty('energy');
      expect(result.result).toHaveProperty('zeroPointEnergy');
    });

    test('should handle invalid quantum number gracefully', async () => {
      const request: SimulationRequest = {
        simulationType: 'quantum',
        operation: 'hydrogen_energy',
        parameters: { n: 0 },
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });
  });

  describe('Particle Simulations', () => {
    test('should execute gravitational_force simulation', async () => {
      const request: SimulationRequest = {
        simulationType: 'particle',
        operation: 'gravitational_force',
        parameters: {
          m1: 5.972e24,
          m2: 7.342e22,
          r: { x: 3.844e8, y: 0, z: 0 },
        },
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(true);
      expect(result.result).toHaveProperty('force');
      expect(result.result).toHaveProperty('magnitude');
    });

    test('should execute orbital_elements simulation', async () => {
      const request: SimulationRequest = {
        simulationType: 'particle',
        operation: 'orbital_elements',
        parameters: {
          centralMass: 1.989e30,
          orbiterMass: 5.972e24,
          position: { x: 1.496e11, y: 0, z: 0 },
          velocity: { x: 0, y: 29780, z: 0 },
        },
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(true);
      expect(result.result).toHaveProperty('semiMajorAxis');
      expect(result.result).toHaveProperty('eccentricity');
      expect(result.result).toHaveProperty('orbitalPeriod');
    });

    test('should execute relativistic simulation', async () => {
      const request: SimulationRequest = {
        simulationType: 'particle',
        operation: 'relativistic',
        parameters: {
          restMass: 9.1e-31,
          velocity: 1.5e8,
        },
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(true);
      expect(result.result).toHaveProperty('lorentzFactor');
      expect(result.result).toHaveProperty('kineticEnergy');
      expect(result.result).toHaveProperty('totalEnergy');
    });

    test('should execute n_body_simulation', async () => {
      const request: SimulationRequest = {
        simulationType: 'particle',
        operation: 'n_body_simulation',
        parameters: {
          particles: [
            { mass: 1e30, position: { x: 0, y: 0, z: 0 }, velocity: { x: 0, y: 0, z: 0 } },
            { mass: 1e24, position: { x: 1e11, y: 0, z: 0 }, velocity: { x: 0, y: 30000, z: 0 } },
          ],
          dt: 86400,
          steps: 10,
        },
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(true);
      expect(result.result).toHaveProperty('trajectory');
      expect(result.result).toHaveProperty('finalState');
      expect(result.result).toHaveProperty('conservationCheck');
    });

    test('should handle velocity >= c gracefully', async () => {
      const request: SimulationRequest = {
        simulationType: 'particle',
        operation: 'relativistic',
        parameters: {
          restMass: 9.1e-31,
          velocity: 3e8, // speed of light
        },
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(false);
      expect(result.error).toContain('speed of light');
    });
  });

  describe('Wave Simulations', () => {
    test('should execute interference simulation', async () => {
      const request: SimulationRequest = {
        simulationType: 'wave',
        operation: 'interference',
        parameters: {
          wavelength: 500e-9,
          sourceSpacing: 1e-3,
          screenDistance: 1,
          screenWidth: 0.1,
        },
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(true);
      expect(result.result).toHaveProperty('resultantAmplitude');
      expect(result.result).toHaveProperty('fringeSpacing');
      expect(result.result).toHaveProperty('constructivePoints');
    });

    test('should execute blackbody simulation', async () => {
      const request: SimulationRequest = {
        simulationType: 'wave',
        operation: 'blackbody',
        parameters: { temperature: 5778 },
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(true);
      expect(result.result).toHaveProperty('peakWavelength');
      expect(result.result).toHaveProperty('totalPower');
      expect(result.result).toHaveProperty('spectralRadiance');
    });

    test('should execute doppler_relativistic simulation', async () => {
      const request: SimulationRequest = {
        simulationType: 'wave',
        operation: 'doppler_relativistic',
        parameters: {
          sourceFrequency: 5e14,
          relativeVelocity: 1e7,
        },
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(true);
      expect(result.result).toHaveProperty('observedFrequency');
      expect(result.result).toHaveProperty('redshift');
    });

    test('should execute refraction simulation', async () => {
      const request: SimulationRequest = {
        simulationType: 'wave',
        operation: 'refraction',
        parameters: {
          incidentAngle: 0.5,
          n1: 1,
          n2: 1.5,
        },
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(true);
      expect(result.result).toHaveProperty('refractedAngle');
      expect(result.result).toHaveProperty('reflectanceS');
    });
  });

  describe('Constants Operations', () => {
    test('should execute get_constant', async () => {
      const request: SimulationRequest = {
        simulationType: 'constants',
        operation: 'get_constant',
        parameters: { name: 'c' },
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(true);
      expect(result.result).toHaveProperty('name', 'c');
      expect(result.result).toHaveProperty('value', 299792458);
    });

    test('should execute get_all_constants', async () => {
      const request: SimulationRequest = {
        simulationType: 'constants',
        operation: 'get_all_constants',
        parameters: {},
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(true);
      expect(result.result).toHaveProperty('c');
      expect(result.result).toHaveProperty('h');
      expect(result.result).toHaveProperty('G');
    });

    test('should handle unknown constant gracefully', async () => {
      const request: SimulationRequest = {
        simulationType: 'constants',
        operation: 'get_constant',
        parameters: { name: 'unknown_constant' },
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Unknown constant');
    });
  });

  describe('Metadata', () => {
    test('should include metadata by default', async () => {
      const request: SimulationRequest = {
        simulationType: 'quantum',
        operation: 'photon_properties',
        parameters: { wavelengthNm: 550 },
      };

      const result = await engine.execute(request);

      expect(result.metadata).toBeDefined();
      expect(result.metadata?.simulationId).toBeDefined();
      expect(result.metadata?.timestamp).toBeDefined();
      expect(result.metadata?.executionTimeMs).toBeDefined();
      expect(result.metadata?.constantsUsed).toBeDefined();
    });

    test('should track constants used', async () => {
      const request: SimulationRequest = {
        simulationType: 'quantum',
        operation: 'photon_properties',
        parameters: { wavelengthNm: 550 },
      };

      const result = await engine.execute(request);

      expect(result.metadata?.constantsUsed).toContain('h');
      expect(result.metadata?.constantsUsed).toContain('c');
    });

    test('should exclude metadata when requested', async () => {
      const request: SimulationRequest = {
        simulationType: 'quantum',
        operation: 'photon_properties',
        parameters: { wavelengthNm: 550 },
        options: { includeMetadata: false },
      };

      const result = await engine.execute(request);

      expect(result.metadata).toBeUndefined();
    });
  });

  describe('Error Handling', () => {
    test('should handle unknown simulation type', async () => {
      const request = {
        simulationType: 'unknown' as any,
        operation: 'test',
        parameters: {},
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Unknown simulation type');
    });

    test('should handle unknown operation', async () => {
      const request: SimulationRequest = {
        simulationType: 'quantum',
        operation: 'unknown_operation' as any,
        parameters: {},
      };

      const result = await engine.execute(request);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Unknown');
    });
  });
});
