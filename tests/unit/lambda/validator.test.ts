/**
 * Request Validator Tests
 */

import { validateRequest, ValidationError } from '../../../src/lambda/validator';

describe('validateRequest', () => {
  describe('Basic Structure Validation', () => {
    test('should reject null body', () => {
      expect(() => validateRequest(null)).toThrow(ValidationError);
    });

    test('should reject non-object body', () => {
      expect(() => validateRequest('string')).toThrow(ValidationError);
      expect(() => validateRequest(123)).toThrow(ValidationError);
    });

    test('should reject missing simulationType', () => {
      expect(() => validateRequest({
        operation: 'test',
        parameters: {},
      })).toThrow(/simulationType/);
    });

    test('should reject invalid simulationType', () => {
      expect(() => validateRequest({
        simulationType: 'invalid',
        operation: 'test',
        parameters: {},
      })).toThrow(/simulationType/);
    });

    test('should reject missing operation', () => {
      expect(() => validateRequest({
        simulationType: 'quantum',
        parameters: {},
      })).toThrow(/operation/);
    });

    test('should reject missing parameters', () => {
      expect(() => validateRequest({
        simulationType: 'quantum',
        operation: 'photon_properties',
      })).toThrow(/parameters/);
    });
  });

  describe('Quantum Operations Validation', () => {
    test('should accept valid photon_properties request', () => {
      const request = validateRequest({
        simulationType: 'quantum',
        operation: 'photon_properties',
        parameters: { wavelengthNm: 550 },
      });

      expect(request.simulationType).toBe('quantum');
      expect(request.operation).toBe('photon_properties');
    });

    test('should reject invalid wavelength', () => {
      expect(() => validateRequest({
        simulationType: 'quantum',
        operation: 'photon_properties',
        parameters: { wavelengthNm: -100 },
      })).toThrow(/wavelengthNm/);
    });

    test('should accept valid hydrogen_energy request', () => {
      const request = validateRequest({
        simulationType: 'quantum',
        operation: 'hydrogen_energy',
        parameters: { n: 2 },
      });

      expect(request.parameters.n).toBe(2);
    });

    test('should reject non-integer quantum number', () => {
      expect(() => validateRequest({
        simulationType: 'quantum',
        operation: 'hydrogen_energy',
        parameters: { n: 2.5 },
      })).toThrow(/n/);
    });

    test('should accept valid tunneling request', () => {
      const request = validateRequest({
        simulationType: 'quantum',
        operation: 'tunneling',
        parameters: {
          particleMass: 9.1e-31,
          particleEnergy: 1e-19,
          barrierHeight: 2e-19,
          barrierWidth: 1e-10,
        },
      });

      expect(request.operation).toBe('tunneling');
    });

    test('should reject invalid operation for quantum', () => {
      expect(() => validateRequest({
        simulationType: 'quantum',
        operation: 'gravitational_force',
        parameters: {},
      })).toThrow(/operation/);
    });
  });

  describe('Particle Operations Validation', () => {
    test('should accept valid gravitational_force request', () => {
      const request = validateRequest({
        simulationType: 'particle',
        operation: 'gravitational_force',
        parameters: {
          m1: 1e24,
          m2: 1e22,
          r: { x: 1e6, y: 0, z: 0 },
        },
      });

      expect(request.operation).toBe('gravitational_force');
    });

    test('should reject invalid vector', () => {
      expect(() => validateRequest({
        simulationType: 'particle',
        operation: 'gravitational_force',
        parameters: {
          m1: 1e24,
          m2: 1e22,
          r: { x: 1, y: 'invalid' },
        },
      })).toThrow(/r.y/);
    });

    test('should accept valid n_body_simulation request', () => {
      const request = validateRequest({
        simulationType: 'particle',
        operation: 'n_body_simulation',
        parameters: {
          particles: [
            { mass: 1e10, position: { x: 0, y: 0, z: 0 }, velocity: { x: 0, y: 0, z: 0 } },
            { mass: 1e10, position: { x: 1e6, y: 0, z: 0 }, velocity: { x: 0, y: 0, z: 0 } },
          ],
          dt: 1,
          steps: 100,
        },
      });

      expect(request.operation).toBe('n_body_simulation');
    });

    test('should reject too many particles', () => {
      const particles = Array(101).fill({
        mass: 1e10,
        position: { x: 0, y: 0, z: 0 },
        velocity: { x: 0, y: 0, z: 0 },
      });

      expect(() => validateRequest({
        simulationType: 'particle',
        operation: 'n_body_simulation',
        parameters: { particles, dt: 1, steps: 100 },
      })).toThrow(/particles/);
    });

    test('should reject too few particles', () => {
      expect(() => validateRequest({
        simulationType: 'particle',
        operation: 'n_body_simulation',
        parameters: {
          particles: [{ mass: 1e10, position: { x: 0, y: 0, z: 0 }, velocity: { x: 0, y: 0, z: 0 } }],
          dt: 1,
          steps: 100,
        },
      })).toThrow(/particles/);
    });

    test('should accept valid relativistic request', () => {
      const request = validateRequest({
        simulationType: 'particle',
        operation: 'relativistic',
        parameters: {
          restMass: 9.1e-31,
          velocity: 1e8,
        },
      });

      expect(request.operation).toBe('relativistic');
    });
  });

  describe('Wave Operations Validation', () => {
    test('should accept valid interference request', () => {
      const request = validateRequest({
        simulationType: 'wave',
        operation: 'interference',
        parameters: {
          wavelength: 500e-9,
          sourceSpacing: 1e-3,
          screenDistance: 1,
          screenWidth: 0.1,
        },
      });

      expect(request.operation).toBe('interference');
    });

    test('should reject negative wavelength', () => {
      expect(() => validateRequest({
        simulationType: 'wave',
        operation: 'interference',
        parameters: {
          wavelength: -500e-9,
          sourceSpacing: 1e-3,
          screenDistance: 1,
          screenWidth: 0.1,
        },
      })).toThrow(/wavelength/);
    });

    test('should accept valid blackbody request', () => {
      const request = validateRequest({
        simulationType: 'wave',
        operation: 'blackbody',
        parameters: { temperature: 6000 },
      });

      expect(request.operation).toBe('blackbody');
    });

    test('should reject temperature too low', () => {
      expect(() => validateRequest({
        simulationType: 'wave',
        operation: 'blackbody',
        parameters: { temperature: 0.1 },
      })).toThrow(/temperature/);
    });

    test('should accept valid refraction request', () => {
      const request = validateRequest({
        simulationType: 'wave',
        operation: 'refraction',
        parameters: {
          incidentAngle: 0.5,
          n1: 1,
          n2: 1.5,
        },
      });

      expect(request.operation).toBe('refraction');
    });

    test('should reject angle out of bounds', () => {
      expect(() => validateRequest({
        simulationType: 'wave',
        operation: 'refraction',
        parameters: {
          incidentAngle: 2, // > π/2
          n1: 1,
          n2: 1.5,
        },
      })).toThrow(/incidentAngle/);
    });
  });

  describe('Constants Operations Validation', () => {
    test('should accept valid get_constant request', () => {
      const request = validateRequest({
        simulationType: 'constants',
        operation: 'get_constant',
        parameters: { name: 'c' },
      });

      expect(request.operation).toBe('get_constant');
    });

    test('should accept get_all_constants with empty parameters', () => {
      const request = validateRequest({
        simulationType: 'constants',
        operation: 'get_all_constants',
        parameters: {},
      });

      expect(request.operation).toBe('get_all_constants');
    });

    test('should reject get_constant without name', () => {
      expect(() => validateRequest({
        simulationType: 'constants',
        operation: 'get_constant',
        parameters: {},
      })).toThrow(/name/);
    });
  });

  describe('Options Validation', () => {
    test('should accept valid options', () => {
      const request = validateRequest({
        simulationType: 'quantum',
        operation: 'photon_properties',
        parameters: { wavelengthNm: 550 },
        options: {
          saveToS3: true,
          includeMetadata: true,
          precision: 'high',
        },
      });

      expect(request.options?.saveToS3).toBe(true);
      expect(request.options?.precision).toBe('high');
    });
  });
});
