/**
 * Physics Simulation Engine
 * Main export file for all physics modules
 */

// Core constants
export { PhysicalConstants, ConstantDetails, ConstantName, ConstantWithUncertainty } from './constants';

// Quantum mechanics
export { QuantumMechanics, QuantumState, WavefunctionResult, PhotonProperties, UncertaintyResult, TunnelingResult, HarmonicOscillatorResult } from './quantum';

// Particle dynamics
export { ParticleDynamics, Vector3D, Particle, Force, SimulationState, CollisionResult, OrbitalElements, RelativisticProperties } from './particles';

// Wave physics
export { WaveSimulation, WaveParameters, WaveResult, InterferenceResult, DiffractionResult, DopplerResult, StandingWaveResult, BlackbodyResult } from './waves';

// Atmospheric physics
export { AtmosphericPhysics, AtmosphericState, AerodynamicResult, ReentryResult, BallisticTrajectoryPoint } from './atmosphere';

// Fluid dynamics
export { FluidDynamics, FluidProperties, FlowProperties, BernoulliResult, DragResult, WaveResult as FluidWaveResult, PumpResult } from './fluids';

// Electromagnetism
export { Electromagnetism, ElectricFieldResult, MagneticFieldResult, CircuitResult, CapacitorResult, InductorResult, EMWaveResult, AntennaResult } from './electromagnetism';

// Thermodynamics
export { Thermodynamics, ThermalProperties, HeatTransferResult, ConvectionResult, RadiationResult, ThermodynamicState, CycleResult, HeatExchangerResult } from './thermodynamics';
