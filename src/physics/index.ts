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

// Numerical methods
export { NumericalMethods, ODEResult, RootResult, IntegrationResult, InterpolationResult, LinearSystemResult, SplineCoefficients } from './numerical';

// Optimization
export { ParticleSwarmOptimizer, PhysicsOptimization, Particle as SwarmParticle, PSOConfig, PSOResult, MOPSOResult, ConstrainedResult } from './optimizer';

// Advanced orbital mechanics
export { OrbitalMechanics, CelestialBody, OrbitalElements as AdvancedOrbitalElements, ManeuverResult, LambertSolution, LagrangePoints, NBodyState, SOLAR_SYSTEM } from './orbital';

// Nuclear and particle physics
export { NuclearPhysics, ParticleData, NuclideData, DecayMode, DecayResult, CrossSectionResult, ReactionResult, ComptonResult, PARTICLES, NUCLIDES } from './nuclear';

// Statistical mechanics
export { StatisticalMechanics, EnsembleProperties, DistributionResult, IsingState, PhaseTransitionResult, FluctuationResult } from './statistical';

// Simulation orchestrator
export { SimulationOrchestrator, SimulationConfig, SimulationResult, ScenarioResult } from './orchestrator';
