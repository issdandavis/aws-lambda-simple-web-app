/**
 * Simulation Request/Response Types
 */

export type SimulationType = 'quantum' | 'particle' | 'wave' | 'constants';

export interface SimulationRequest {
  simulationType: SimulationType;
  operation: string;
  parameters: Record<string, unknown>;
  options?: SimulationOptions;
}

export interface SimulationOptions {
  saveToS3?: boolean;
  includeMetadata?: boolean;
  precision?: 'standard' | 'high';
}

export interface SimulationResult {
  success: boolean;
  simulationType: SimulationType;
  operation: string;
  result: unknown;
  metadata?: SimulationMetadata;
  error?: string;
}

export interface SimulationMetadata {
  simulationId: string;
  timestamp: string;
  executionTimeMs: number;
  constantsUsed: string[];
  s3Key?: string;
}

// Quantum simulation operations
export type QuantumOperation =
  | 'photon_properties'
  | 'hydrogen_energy'
  | 'hydrogen_transition'
  | 'uncertainty'
  | 'tunneling'
  | 'harmonic_oscillator'
  | 'particle_in_box'
  | 'de_broglie'
  | 'spin_orbit'
  | 'hydrogen_wavefunction';

// Particle dynamics operations
export type ParticleOperation =
  | 'gravitational_force'
  | 'electrostatic_force'
  | 'lorentz_force'
  | 'orbital_elements'
  | 'n_body_simulation'
  | 'elastic_collision'
  | 'inelastic_collision'
  | 'relativistic'
  | 'escape_velocity'
  | 'schwarzschild_radius'
  | 'pendulum'
  | 'terminal_velocity';

// Wave simulation operations
export type WaveOperation =
  | 'wave_parameters'
  | 'sinusoidal_wave'
  | 'interference'
  | 'single_slit_diffraction'
  | 'diffraction_grating'
  | 'doppler_sound'
  | 'doppler_relativistic'
  | 'standing_waves'
  | 'blackbody'
  | 'em_wave_properties'
  | 'refraction';

// Constants operations
export type ConstantsOperation =
  | 'get_constant'
  | 'get_all_constants'
  | 'get_constant_with_uncertainty';

// Parameter interfaces for each operation
export interface PhotonPropertiesParams {
  wavelengthNm: number;
}

export interface HydrogenEnergyParams {
  n: number;
}

export interface HydrogenTransitionParams {
  nInitial: number;
  nFinal: number;
}

export interface UncertaintyParams {
  deltaX: number;
}

export interface TunnelingParams {
  particleMass: number;
  particleEnergy: number;
  barrierHeight: number;
  barrierWidth: number;
}

export interface HarmonicOscillatorParams {
  mass: number;
  angularFrequency: number;
  n: number;
}

export interface ParticleInBoxParams {
  mass: number;
  boxLength: number;
  n: number;
}

export interface DeBroglieParams {
  mass: number;
  velocity: number;
}

export interface Vector3DParams {
  x: number;
  y: number;
  z: number;
}

export interface GravitationalForceParams {
  m1: number;
  m2: number;
  r: Vector3DParams;
}

export interface ElectrostaticForceParams {
  q1: number;
  q2: number;
  r: Vector3DParams;
}

export interface LorentzForceParams {
  charge: number;
  velocity: Vector3DParams;
  electricField: Vector3DParams;
  magneticField: Vector3DParams;
}

export interface OrbitalElementsParams {
  centralMass: number;
  orbiterMass: number;
  position: Vector3DParams;
  velocity: Vector3DParams;
}

export interface NBodyParams {
  particles: Array<{
    mass: number;
    charge?: number;
    position: Vector3DParams;
    velocity: Vector3DParams;
  }>;
  dt: number;
  steps: number;
  includeElectrostatic?: boolean;
}

export interface ElasticCollisionParams {
  m1: number;
  v1: number;
  m2: number;
  v2: number;
}

export interface InelasticCollisionParams {
  m1: number;
  v1: number;
  m2: number;
  v2: number;
  restitution: number;
}

export interface RelativisticParams {
  restMass: number;
  velocity: number;
}

export interface WaveParams {
  amplitude: number;
  frequency: number;
  mediumVelocity?: number;
  phase?: number;
}

export interface SinusoidalWaveParams {
  params: WaveParams;
  xMin: number;
  xMax: number;
  time: number;
  numPoints?: number;
}

export interface InterferenceParams {
  wavelength: number;
  sourceSpacing: number;
  screenDistance: number;
  screenWidth: number;
  numPoints?: number;
}

export interface DiffractionParams {
  wavelength: number;
  slitWidth: number;
  screenDistance: number;
  maxAngle?: number;
  numPoints?: number;
}

export interface DopplerSoundParams {
  sourceFrequency: number;
  soundSpeed: number;
  sourceVelocity: number;
  observerVelocity: number;
}

export interface DopplerRelativisticParams {
  sourceFrequency: number;
  relativeVelocity: number;
}

export interface StandingWavesParams {
  stringLength: number;
  tension: number;
  linearDensity: number;
  numModes?: number;
}

export interface BlackbodyParams {
  temperature: number;
  wavelengthMin?: number;
  wavelengthMax?: number;
  numPoints?: number;
}

export interface RefractionParams {
  incidentAngle: number;
  n1: number;
  n2: number;
}

export interface GetConstantParams {
  name: string;
}
