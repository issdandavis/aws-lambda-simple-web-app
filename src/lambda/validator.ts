/**
 * Input Validation for Simulation Requests
 */

import {
  SimulationRequest,
  SimulationType,
  QuantumOperation,
  ParticleOperation,
  WaveOperation,
  ConstantsOperation,
  Vector3DParams,
} from './types';

export class ValidationError extends Error {
  constructor(message: string, public field?: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

const SIMULATION_TYPES: SimulationType[] = ['quantum', 'particle', 'wave', 'constants'];

const QUANTUM_OPERATIONS: QuantumOperation[] = [
  'photon_properties',
  'hydrogen_energy',
  'hydrogen_transition',
  'uncertainty',
  'tunneling',
  'harmonic_oscillator',
  'particle_in_box',
  'de_broglie',
  'spin_orbit',
  'hydrogen_wavefunction',
];

const PARTICLE_OPERATIONS: ParticleOperation[] = [
  'gravitational_force',
  'electrostatic_force',
  'lorentz_force',
  'orbital_elements',
  'n_body_simulation',
  'elastic_collision',
  'inelastic_collision',
  'relativistic',
  'escape_velocity',
  'schwarzschild_radius',
  'pendulum',
  'terminal_velocity',
];

const WAVE_OPERATIONS: WaveOperation[] = [
  'wave_parameters',
  'sinusoidal_wave',
  'interference',
  'single_slit_diffraction',
  'diffraction_grating',
  'doppler_sound',
  'doppler_relativistic',
  'standing_waves',
  'blackbody',
  'em_wave_properties',
  'refraction',
];

const CONSTANTS_OPERATIONS: ConstantsOperation[] = [
  'get_constant',
  'get_all_constants',
  'get_constant_with_uncertainty',
];

/**
 * Validate a number is finite and optionally within bounds
 */
function validateNumber(
  value: unknown,
  field: string,
  options: { min?: number; max?: number; positive?: boolean; integer?: boolean } = {}
): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ValidationError(`${field} must be a finite number`, field);
  }

  if (options.positive && value <= 0) {
    throw new ValidationError(`${field} must be positive`, field);
  }

  if (options.min !== undefined && value < options.min) {
    throw new ValidationError(`${field} must be at least ${options.min}`, field);
  }

  if (options.max !== undefined && value > options.max) {
    throw new ValidationError(`${field} must be at most ${options.max}`, field);
  }

  if (options.integer && !Number.isInteger(value)) {
    throw new ValidationError(`${field} must be an integer`, field);
  }

  return value;
}

/**
 * Validate a Vector3D object
 */
function validateVector3D(value: unknown, field: string): Vector3DParams {
  if (!value || typeof value !== 'object') {
    throw new ValidationError(`${field} must be an object with x, y, z properties`, field);
  }

  const vec = value as Record<string, unknown>;

  return {
    x: validateNumber(vec.x, `${field}.x`),
    y: validateNumber(vec.y, `${field}.y`),
    z: validateNumber(vec.z, `${field}.z`),
  };
}

/**
 * Validate the main simulation request structure
 */
export function validateRequest(body: unknown): SimulationRequest {
  if (!body || typeof body !== 'object') {
    throw new ValidationError('Request body must be a JSON object');
  }

  const request = body as Record<string, unknown>;

  // Validate simulationType
  if (!request.simulationType || typeof request.simulationType !== 'string') {
    throw new ValidationError('simulationType is required and must be a string', 'simulationType');
  }

  if (!SIMULATION_TYPES.includes(request.simulationType as SimulationType)) {
    throw new ValidationError(
      `Invalid simulationType. Must be one of: ${SIMULATION_TYPES.join(', ')}`,
      'simulationType'
    );
  }

  // Validate operation
  if (!request.operation || typeof request.operation !== 'string') {
    throw new ValidationError('operation is required and must be a string', 'operation');
  }

  // Validate operation matches simulationType
  validateOperation(request.simulationType as SimulationType, request.operation as string);

  // Validate parameters
  if (!request.parameters || typeof request.parameters !== 'object') {
    throw new ValidationError('parameters is required and must be an object', 'parameters');
  }

  // Validate parameters for specific operation
  validateParameters(
    request.simulationType as SimulationType,
    request.operation as string,
    request.parameters as Record<string, unknown>
  );

  return {
    simulationType: request.simulationType as SimulationType,
    operation: request.operation as string,
    parameters: request.parameters as Record<string, unknown>,
    options: request.options as SimulationRequest['options'],
  };
}

/**
 * Validate operation matches simulation type
 */
function validateOperation(simulationType: SimulationType, operation: string): void {
  let validOperations: string[];

  switch (simulationType) {
    case 'quantum':
      validOperations = QUANTUM_OPERATIONS;
      break;
    case 'particle':
      validOperations = PARTICLE_OPERATIONS;
      break;
    case 'wave':
      validOperations = WAVE_OPERATIONS;
      break;
    case 'constants':
      validOperations = CONSTANTS_OPERATIONS;
      break;
    default:
      throw new ValidationError(`Unknown simulation type: ${simulationType}`, 'simulationType');
  }

  if (!validOperations.includes(operation)) {
    throw new ValidationError(
      `Invalid operation for ${simulationType}. Must be one of: ${validOperations.join(', ')}`,
      'operation'
    );
  }
}

/**
 * Validate parameters for specific operations
 */
function validateParameters(
  simulationType: SimulationType,
  operation: string,
  params: Record<string, unknown>
): void {
  switch (simulationType) {
    case 'quantum':
      validateQuantumParams(operation, params);
      break;
    case 'particle':
      validateParticleParams(operation, params);
      break;
    case 'wave':
      validateWaveParams(operation, params);
      break;
    case 'constants':
      validateConstantsParams(operation, params);
      break;
  }
}

/**
 * Validate quantum mechanics parameters
 */
function validateQuantumParams(operation: string, params: Record<string, unknown>): void {
  switch (operation) {
    case 'photon_properties':
      validateNumber(params.wavelengthNm, 'wavelengthNm', { positive: true, min: 0.001, max: 1e12 });
      break;

    case 'hydrogen_energy':
      validateNumber(params.n, 'n', { positive: true, integer: true, min: 1, max: 1000 });
      break;

    case 'hydrogen_transition':
      validateNumber(params.nInitial, 'nInitial', { positive: true, integer: true, min: 1, max: 1000 });
      validateNumber(params.nFinal, 'nFinal', { positive: true, integer: true, min: 1, max: 1000 });
      break;

    case 'uncertainty':
      validateNumber(params.deltaX, 'deltaX', { positive: true });
      break;

    case 'tunneling':
      validateNumber(params.particleMass, 'particleMass', { positive: true });
      validateNumber(params.particleEnergy, 'particleEnergy', { positive: true });
      validateNumber(params.barrierHeight, 'barrierHeight', { positive: true });
      validateNumber(params.barrierWidth, 'barrierWidth', { positive: true });
      break;

    case 'harmonic_oscillator':
      validateNumber(params.mass, 'mass', { positive: true });
      validateNumber(params.angularFrequency, 'angularFrequency', { positive: true });
      validateNumber(params.n, 'n', { min: 0, integer: true });
      break;

    case 'particle_in_box':
      validateNumber(params.mass, 'mass', { positive: true });
      validateNumber(params.boxLength, 'boxLength', { positive: true });
      validateNumber(params.n, 'n', { positive: true, integer: true, min: 1 });
      break;

    case 'de_broglie':
      validateNumber(params.mass, 'mass', { positive: true });
      validateNumber(params.velocity, 'velocity', { positive: true });
      break;

    case 'spin_orbit':
      validateNumber(params.n, 'n', { positive: true, integer: true, min: 1 });
      validateNumber(params.l, 'l', { min: 0, integer: true });
      validateNumber(params.j, 'j', { min: 0 });
      break;

    case 'hydrogen_wavefunction':
      validateNumber(params.n, 'n', { positive: true, integer: true, min: 1 });
      validateNumber(params.l, 'l', { min: 0, integer: true });
      if (!Array.isArray(params.rValues)) {
        throw new ValidationError('rValues must be an array of numbers', 'rValues');
      }
      break;
  }
}

/**
 * Validate particle dynamics parameters
 */
function validateParticleParams(operation: string, params: Record<string, unknown>): void {
  switch (operation) {
    case 'gravitational_force':
      validateNumber(params.m1, 'm1', { positive: true });
      validateNumber(params.m2, 'm2', { positive: true });
      validateVector3D(params.r, 'r');
      break;

    case 'electrostatic_force':
      validateNumber(params.q1, 'q1');
      validateNumber(params.q2, 'q2');
      validateVector3D(params.r, 'r');
      break;

    case 'lorentz_force':
      validateNumber(params.charge, 'charge');
      validateVector3D(params.velocity, 'velocity');
      validateVector3D(params.electricField, 'electricField');
      validateVector3D(params.magneticField, 'magneticField');
      break;

    case 'orbital_elements':
      validateNumber(params.centralMass, 'centralMass', { positive: true });
      validateNumber(params.orbiterMass, 'orbiterMass', { positive: true });
      validateVector3D(params.position, 'position');
      validateVector3D(params.velocity, 'velocity');
      break;

    case 'n_body_simulation':
      if (!Array.isArray(params.particles) || params.particles.length < 2) {
        throw new ValidationError('particles must be an array with at least 2 particles', 'particles');
      }
      if ((params.particles as unknown[]).length > 100) {
        throw new ValidationError('Maximum 100 particles allowed', 'particles');
      }
      for (let i = 0; i < (params.particles as unknown[]).length; i++) {
        const p = (params.particles as Record<string, unknown>[])[i];
        validateNumber(p.mass, `particles[${i}].mass`, { positive: true });
        validateVector3D(p.position, `particles[${i}].position`);
        validateVector3D(p.velocity, `particles[${i}].velocity`);
      }
      validateNumber(params.dt, 'dt', { positive: true });
      validateNumber(params.steps, 'steps', { positive: true, integer: true, max: 10000 });
      break;

    case 'elastic_collision':
    case 'inelastic_collision':
      validateNumber(params.m1, 'm1', { positive: true });
      validateNumber(params.v1, 'v1');
      validateNumber(params.m2, 'm2', { positive: true });
      validateNumber(params.v2, 'v2');
      if (operation === 'inelastic_collision') {
        validateNumber(params.restitution, 'restitution', { min: 0, max: 1 });
      }
      break;

    case 'relativistic':
      validateNumber(params.restMass, 'restMass', { positive: true });
      validateNumber(params.velocity, 'velocity', { min: 0 });
      break;

    case 'escape_velocity':
      validateNumber(params.centralMass, 'centralMass', { positive: true });
      validateNumber(params.radius, 'radius', { positive: true });
      break;

    case 'schwarzschild_radius':
      validateNumber(params.mass, 'mass', { positive: true });
      break;

    case 'pendulum':
      validateNumber(params.length, 'length', { positive: true });
      if (params.gravity !== undefined) {
        validateNumber(params.gravity, 'gravity', { positive: true });
      }
      break;

    case 'terminal_velocity':
      validateNumber(params.mass, 'mass', { positive: true });
      validateNumber(params.gravity, 'gravity', { positive: true });
      validateNumber(params.fluidDensity, 'fluidDensity', { positive: true });
      validateNumber(params.crossSectionalArea, 'crossSectionalArea', { positive: true });
      validateNumber(params.dragCoefficient, 'dragCoefficient', { positive: true });
      break;
  }
}

/**
 * Validate wave simulation parameters
 */
function validateWaveParams(operation: string, params: Record<string, unknown>): void {
  switch (operation) {
    case 'wave_parameters':
      validateNumber(params.amplitude, 'amplitude', { positive: true });
      validateNumber(params.frequency, 'frequency', { positive: true });
      break;

    case 'sinusoidal_wave':
      if (!params.params || typeof params.params !== 'object') {
        throw new ValidationError('params must be an object with wave parameters', 'params');
      }
      const waveParams = params.params as Record<string, unknown>;
      validateNumber(waveParams.amplitude, 'params.amplitude', { positive: true });
      validateNumber(waveParams.frequency, 'params.frequency', { positive: true });
      validateNumber(params.xMin, 'xMin');
      validateNumber(params.xMax, 'xMax');
      validateNumber(params.time, 'time', { min: 0 });
      break;

    case 'interference':
      validateNumber(params.wavelength, 'wavelength', { positive: true });
      validateNumber(params.sourceSpacing, 'sourceSpacing', { positive: true });
      validateNumber(params.screenDistance, 'screenDistance', { positive: true });
      validateNumber(params.screenWidth, 'screenWidth', { positive: true });
      break;

    case 'single_slit_diffraction':
      validateNumber(params.wavelength, 'wavelength', { positive: true });
      validateNumber(params.slitWidth, 'slitWidth', { positive: true });
      validateNumber(params.screenDistance, 'screenDistance', { positive: true });
      break;

    case 'diffraction_grating':
      validateNumber(params.wavelength, 'wavelength', { positive: true });
      validateNumber(params.gratingSpacing, 'gratingSpacing', { positive: true });
      validateNumber(params.numSlits, 'numSlits', { positive: true, integer: true, min: 2 });
      break;

    case 'doppler_sound':
      validateNumber(params.sourceFrequency, 'sourceFrequency', { positive: true });
      validateNumber(params.soundSpeed, 'soundSpeed', { positive: true });
      validateNumber(params.sourceVelocity, 'sourceVelocity');
      validateNumber(params.observerVelocity, 'observerVelocity');
      break;

    case 'doppler_relativistic':
      validateNumber(params.sourceFrequency, 'sourceFrequency', { positive: true });
      validateNumber(params.relativeVelocity, 'relativeVelocity');
      break;

    case 'standing_waves':
      validateNumber(params.stringLength, 'stringLength', { positive: true });
      validateNumber(params.tension, 'tension', { positive: true });
      validateNumber(params.linearDensity, 'linearDensity', { positive: true });
      break;

    case 'blackbody':
      validateNumber(params.temperature, 'temperature', { positive: true, min: 1, max: 1e9 });
      break;

    case 'em_wave_properties':
      validateNumber(params.electricFieldAmplitude, 'electricFieldAmplitude', { positive: true });
      validateNumber(params.frequency, 'frequency', { positive: true });
      break;

    case 'refraction':
      validateNumber(params.incidentAngle, 'incidentAngle', { min: 0, max: Math.PI / 2 });
      validateNumber(params.n1, 'n1', { positive: true });
      validateNumber(params.n2, 'n2', { positive: true });
      break;
  }
}

/**
 * Validate constants parameters
 */
function validateConstantsParams(operation: string, params: Record<string, unknown>): void {
  switch (operation) {
    case 'get_constant':
    case 'get_constant_with_uncertainty':
      if (!params.name || typeof params.name !== 'string') {
        throw new ValidationError('name is required and must be a string', 'name');
      }
      break;

    case 'get_all_constants':
      // No parameters required
      break;
  }
}

export { QUANTUM_OPERATIONS, PARTICLE_OPERATIONS, WAVE_OPERATIONS, CONSTANTS_OPERATIONS };
