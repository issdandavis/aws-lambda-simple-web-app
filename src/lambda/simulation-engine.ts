/**
 * Simulation Engine - Executes physics simulations
 */

import { QuantumMechanics, ParticleDynamics, WaveSimulation, PhysicalConstants, ConstantDetails } from '../physics';
import { SimulationRequest, SimulationResult, SimulationMetadata } from './types';
import { v4 as uuidv4 } from 'uuid';

export class SimulationEngine {
  private constantsUsed: Set<string> = new Set();

  /**
   * Execute a simulation based on the request
   */
  async execute(request: SimulationRequest): Promise<SimulationResult> {
    const startTime = Date.now();
    const simulationId = uuidv4();
    this.constantsUsed.clear();

    try {
      let result: unknown;

      switch (request.simulationType) {
        case 'quantum':
          result = this.executeQuantum(request.operation, request.parameters);
          break;
        case 'particle':
          result = this.executeParticle(request.operation, request.parameters);
          break;
        case 'wave':
          result = this.executeWave(request.operation, request.parameters);
          break;
        case 'constants':
          result = this.executeConstants(request.operation, request.parameters);
          break;
        default:
          throw new Error(`Unknown simulation type: ${request.simulationType}`);
      }

      const executionTimeMs = Date.now() - startTime;

      const metadata: SimulationMetadata = {
        simulationId,
        timestamp: new Date().toISOString(),
        executionTimeMs,
        constantsUsed: Array.from(this.constantsUsed),
      };

      return {
        success: true,
        simulationType: request.simulationType,
        operation: request.operation,
        result,
        metadata: request.options?.includeMetadata !== false ? metadata : undefined,
      };
    } catch (error) {
      const executionTimeMs = Date.now() - startTime;

      return {
        success: false,
        simulationType: request.simulationType,
        operation: request.operation,
        result: null,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
        metadata: {
          simulationId,
          timestamp: new Date().toISOString(),
          executionTimeMs,
          constantsUsed: Array.from(this.constantsUsed),
        },
      };
    }
  }

  /**
   * Execute quantum mechanics simulations
   */
  private executeQuantum(operation: string, params: Record<string, unknown>): unknown {
    this.trackConstants(['h', 'hbar', 'c', 'me', 'e', 'epsilon0']);

    switch (operation) {
      case 'photon_properties':
        return QuantumMechanics.calculatePhotonProperties(params.wavelengthNm as number);

      case 'hydrogen_energy':
        return {
          n: params.n,
          energy: QuantumMechanics.hydrogenEnergyLevel(params.n as number),
          energyEV: QuantumMechanics.hydrogenEnergyLevel(params.n as number) / PhysicalConstants.eV_to_J,
        };

      case 'hydrogen_transition':
        return QuantumMechanics.hydrogenTransition(
          params.nInitial as number,
          params.nFinal as number
        );

      case 'uncertainty':
        return QuantumMechanics.calculateUncertainty(params.deltaX as number);

      case 'tunneling':
        return QuantumMechanics.quantumTunneling(
          params.particleMass as number,
          params.particleEnergy as number,
          params.barrierHeight as number,
          params.barrierWidth as number
        );

      case 'harmonic_oscillator':
        return QuantumMechanics.harmonicOscillator(
          params.mass as number,
          params.angularFrequency as number,
          params.n as number
        );

      case 'particle_in_box':
        return QuantumMechanics.particleInBox(
          params.mass as number,
          params.boxLength as number,
          params.n as number
        );

      case 'de_broglie':
        return {
          wavelength: QuantumMechanics.deBroglieWavelength(
            params.mass as number,
            params.velocity as number
          ),
          mass: params.mass,
          velocity: params.velocity,
        };

      case 'spin_orbit':
        return {
          n: params.n,
          l: params.l,
          j: params.j,
          energy: QuantumMechanics.spinOrbitCoupling(
            params.n as number,
            params.l as number,
            params.j as number
          ),
        };

      case 'hydrogen_wavefunction':
        return QuantumMechanics.hydrogenRadialWavefunction(
          params.n as number,
          params.l as number,
          params.rValues as number[]
        );

      default:
        throw new Error(`Unknown quantum operation: ${operation}`);
    }
  }

  /**
   * Execute particle dynamics simulations
   */
  private executeParticle(operation: string, params: Record<string, unknown>): unknown {
    this.trackConstants(['G', 'c', 'ke']);

    switch (operation) {
      case 'gravitational_force': {
        const r = params.r as { x: number; y: number; z: number };
        const force = ParticleDynamics.gravitationalForce(
          params.m1 as number,
          params.m2 as number,
          r
        );
        return {
          force,
          magnitude: ParticleDynamics.vectorMagnitude(force),
        };
      }

      case 'electrostatic_force': {
        const r = params.r as { x: number; y: number; z: number };
        const force = ParticleDynamics.electrostaticForce(
          params.q1 as number,
          params.q2 as number,
          r
        );
        return {
          force,
          magnitude: ParticleDynamics.vectorMagnitude(force),
        };
      }

      case 'lorentz_force': {
        const force = ParticleDynamics.lorentzForce(
          params.charge as number,
          params.velocity as { x: number; y: number; z: number },
          params.electricField as { x: number; y: number; z: number },
          params.magneticField as { x: number; y: number; z: number }
        );
        return {
          force,
          magnitude: ParticleDynamics.vectorMagnitude(force),
        };
      }

      case 'orbital_elements':
        return ParticleDynamics.calculateOrbitalElements(
          params.centralMass as number,
          params.orbiterMass as number,
          params.position as { x: number; y: number; z: number },
          params.velocity as { x: number; y: number; z: number }
        );

      case 'n_body_simulation': {
        const particles = (params.particles as Array<{
          mass: number;
          charge?: number;
          position: { x: number; y: number; z: number };
          velocity: { x: number; y: number; z: number };
        }>).map(p => ({
          mass: p.mass,
          charge: p.charge || 0,
          position: p.position,
          velocity: p.velocity,
        }));

        const dt = params.dt as number;
        const steps = params.steps as number;
        const includeElectrostatic = params.includeElectrostatic as boolean || false;

        const trajectory: Array<{
          step: number;
          time: number;
          particles: typeof particles;
          totalEnergy: number;
          centerOfMass: { x: number; y: number; z: number };
        }> = [];

        let currentParticles = particles;

        for (let step = 0; step <= steps; step++) {
          // Calculate total energy
          let totalKE = 0;
          let totalPE = 0;

          for (const p of currentParticles) {
            totalKE += ParticleDynamics.kineticEnergy(p.mass, p.velocity);
          }

          for (let i = 0; i < currentParticles.length; i++) {
            for (let j = i + 1; j < currentParticles.length; j++) {
              const distance = ParticleDynamics.vectorMagnitude(
                ParticleDynamics.vectorSubtract(
                  currentParticles[i].position,
                  currentParticles[j].position
                )
              );
              totalPE += ParticleDynamics.gravitationalPotentialEnergy(
                currentParticles[i].mass,
                currentParticles[j].mass,
                distance
              );
            }
          }

          // Store state at intervals
          if (step % Math.max(1, Math.floor(steps / 100)) === 0 || step === steps) {
            trajectory.push({
              step,
              time: step * dt,
              particles: JSON.parse(JSON.stringify(currentParticles)),
              totalEnergy: totalKE + totalPE,
              centerOfMass: ParticleDynamics.centerOfMass(currentParticles),
            });
          }

          if (step < steps) {
            currentParticles = ParticleDynamics.nBodyStep(currentParticles, dt, includeElectrostatic);
          }
        }

        return {
          initialState: particles,
          finalState: currentParticles,
          trajectory,
          conservationCheck: {
            initialEnergy: trajectory[0].totalEnergy,
            finalEnergy: trajectory[trajectory.length - 1].totalEnergy,
            energyDrift: Math.abs(trajectory[trajectory.length - 1].totalEnergy - trajectory[0].totalEnergy),
          },
        };
      }

      case 'elastic_collision':
        return ParticleDynamics.elasticCollision(
          params.m1 as number,
          params.v1 as number,
          params.m2 as number,
          params.v2 as number
        );

      case 'inelastic_collision':
        return ParticleDynamics.inelasticCollision(
          params.m1 as number,
          params.v1 as number,
          params.m2 as number,
          params.v2 as number,
          params.restitution as number
        );

      case 'relativistic':
        return ParticleDynamics.calculateRelativistic(
          params.restMass as number,
          params.velocity as number
        );

      case 'escape_velocity':
        return {
          escapeVelocity: ParticleDynamics.escapeVelocity(
            params.centralMass as number,
            params.radius as number
          ),
          centralMass: params.centralMass,
          radius: params.radius,
        };

      case 'schwarzschild_radius':
        return {
          schwarzschildRadius: ParticleDynamics.schwarzschildRadius(params.mass as number),
          mass: params.mass,
        };

      case 'pendulum':
        return {
          period: ParticleDynamics.pendulumPeriod(
            params.length as number,
            params.gravity as number || 9.80665
          ),
          length: params.length,
          gravity: params.gravity || 9.80665,
        };

      case 'terminal_velocity':
        return {
          terminalVelocity: ParticleDynamics.terminalVelocity(
            params.mass as number,
            params.gravity as number,
            params.fluidDensity as number,
            params.crossSectionalArea as number,
            params.dragCoefficient as number
          ),
        };

      default:
        throw new Error(`Unknown particle operation: ${operation}`);
    }
  }

  /**
   * Execute wave simulations
   */
  private executeWave(operation: string, params: Record<string, unknown>): unknown {
    this.trackConstants(['c', 'h', 'kB', 'sigma', 'epsilon0', 'mu0']);

    switch (operation) {
      case 'wave_parameters':
        return WaveSimulation.calculateWaveParameters(
          params.amplitude as number,
          params.frequency as number,
          params.mediumVelocity as number || PhysicalConstants.c,
          params.phase as number || 0
        );

      case 'sinusoidal_wave': {
        const waveParams = params.params as {
          amplitude: number;
          frequency: number;
          wavelength?: number;
          phase?: number;
          velocity?: number;
        };
        const fullParams = WaveSimulation.calculateWaveParameters(
          waveParams.amplitude,
          waveParams.frequency,
          waveParams.velocity || PhysicalConstants.c,
          waveParams.phase || 0
        );
        return WaveSimulation.sinusoidalWave(
          fullParams,
          params.xMin as number,
          params.xMax as number,
          params.time as number,
          params.numPoints as number || 100
        );
      }

      case 'interference':
        return WaveSimulation.twoSourceInterference(
          params.wavelength as number,
          params.sourceSpacing as number,
          params.screenDistance as number,
          params.screenWidth as number,
          params.numPoints as number || 200
        );

      case 'single_slit_diffraction':
        return WaveSimulation.singleSlitDiffraction(
          params.wavelength as number,
          params.slitWidth as number,
          params.screenDistance as number,
          params.maxAngle as number || Math.PI / 6,
          params.numPoints as number || 200
        );

      case 'diffraction_grating':
        return WaveSimulation.diffractionGrating(
          params.wavelength as number,
          params.gratingSpacing as number,
          params.numSlits as number,
          params.maxAngle as number || Math.PI / 3,
          params.numPoints as number || 500
        );

      case 'doppler_sound':
        return WaveSimulation.dopplerSound(
          params.sourceFrequency as number,
          params.soundSpeed as number,
          params.sourceVelocity as number,
          params.observerVelocity as number
        );

      case 'doppler_relativistic':
        return WaveSimulation.dopplerRelativistic(
          params.sourceFrequency as number,
          params.relativeVelocity as number
        );

      case 'standing_waves':
        return WaveSimulation.standingWaves(
          params.stringLength as number,
          params.tension as number,
          params.linearDensity as number,
          params.numModes as number || 5
        );

      case 'blackbody':
        return WaveSimulation.blackbodyRadiation(
          params.temperature as number,
          params.wavelengthMin as number,
          params.wavelengthMax as number,
          params.numPoints as number || 200
        );

      case 'em_wave_properties':
        return WaveSimulation.emWaveProperties(
          params.electricFieldAmplitude as number,
          params.frequency as number
        );

      case 'refraction':
        return WaveSimulation.refraction(
          params.incidentAngle as number,
          params.n1 as number,
          params.n2 as number
        );

      default:
        throw new Error(`Unknown wave operation: ${operation}`);
    }
  }

  /**
   * Execute constants operations
   */
  private executeConstants(operation: string, params: Record<string, unknown>): unknown {
    switch (operation) {
      case 'get_constant': {
        const name = params.name as string;
        const value = (PhysicalConstants as Record<string, number>)[name];
        if (value === undefined) {
          throw new Error(`Unknown constant: ${name}. Available: ${Object.keys(PhysicalConstants).join(', ')}`);
        }
        this.trackConstants([name]);
        return { name, value };
      }

      case 'get_all_constants':
        this.trackConstants(Object.keys(PhysicalConstants));
        return PhysicalConstants;

      case 'get_constant_with_uncertainty': {
        const name = params.name as string;
        const details = ConstantDetails[name];
        if (!details) {
          throw new Error(`No uncertainty data for constant: ${name}. Available: ${Object.keys(ConstantDetails).join(', ')}`);
        }
        this.trackConstants([name]);
        return { name, ...details };
      }

      default:
        throw new Error(`Unknown constants operation: ${operation}`);
    }
  }

  /**
   * Track which constants were used in the simulation
   */
  private trackConstants(constants: string[]): void {
    for (const c of constants) {
      this.constantsUsed.add(c);
    }
  }
}
