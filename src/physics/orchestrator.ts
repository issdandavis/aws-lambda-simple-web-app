/**
 * Unified Simulation Orchestrator
 *
 * Coordinates multi-physics simulations by combining:
 * - Particle dynamics with EM fields
 * - Orbital mechanics with atmospheric drag
 * - Thermodynamics with statistical mechanics
 * - Nuclear physics with radiation transport
 * - Optimization for design problems
 *
 * Provides a high-level API for complex simulations
 */

import { PhysicalConstants } from './constants';
import { QuantumMechanics } from './quantum';
import { ParticleDynamics, Vector3D as PDVector3D, Particle } from './particles';
import { WaveSimulation } from './waves';
import { AtmosphericPhysics } from './atmosphere';
import { FluidDynamics } from './fluids';
import { Electromagnetism } from './electromagnetism';
import { Thermodynamics } from './thermodynamics';
import { NumericalMethods } from './numerical';
import { ParticleSwarmOptimizer, PhysicsOptimization } from './optimizer';
import { OrbitalMechanics, CelestialBody, Vector3D } from './orbital';
import { NuclearPhysics } from './nuclear';
import { StatisticalMechanics } from './statistical';

export interface SimulationConfig {
  name: string;
  description: string;
  modules: string[];
  parameters: Record<string, any>;
  outputInterval?: number;
  maxDuration?: number;
  convergenceTolerance?: number;
}

export interface SimulationResult {
  name: string;
  success: boolean;
  duration: number;
  iterations?: number;
  data: Record<string, any>;
  timeSeries?: { time: number; state: Record<string, any> }[];
  summary: Record<string, number>;
  warnings?: string[];
  errors?: string[];
}

export interface ScenarioResult {
  scenario: string;
  results: SimulationResult[];
  totalDuration: number;
  summary: Record<string, any>;
}

// Pre-built simulation scenarios
type ScenarioName =
  | 'rocket_launch'
  | 'satellite_orbit'
  | 'particle_accelerator'
  | 'nuclear_reactor'
  | 'heat_exchanger'
  | 'laser_cavity'
  | 'plasma_confinement'
  | 'stellar_evolution'
  | 'atmospheric_reentry';

export class SimulationOrchestrator {
  private results: Map<string, SimulationResult> = new Map();

  /**
   * Run a pre-built scenario
   */
  async runScenario(
    scenario: ScenarioName,
    parameters: Record<string, any> = {}
  ): Promise<ScenarioResult> {
    const startTime = Date.now();

    switch (scenario) {
      case 'rocket_launch':
        return this.runRocketLaunchScenario(parameters);
      case 'satellite_orbit':
        return this.runSatelliteOrbitScenario(parameters);
      case 'particle_accelerator':
        return this.runParticleAcceleratorScenario(parameters);
      case 'nuclear_reactor':
        return this.runNuclearReactorScenario(parameters);
      case 'heat_exchanger':
        return this.runHeatExchangerScenario(parameters);
      case 'laser_cavity':
        return this.runLaserCavityScenario(parameters);
      case 'plasma_confinement':
        return this.runPlasmaConfinementScenario(parameters);
      case 'atmospheric_reentry':
        return this.runAtmosphericReentryScenario(parameters);
      default:
        return {
          scenario,
          results: [],
          totalDuration: Date.now() - startTime,
          summary: { error: 'Unknown scenario' }
        };
    }
  }

  /**
   * Rocket launch simulation combining atmospheric physics and orbital mechanics
   */
  private async runRocketLaunchScenario(params: Record<string, any>): Promise<ScenarioResult> {
    const startTime = Date.now();
    const results: SimulationResult[] = [];

    const {
      initialMass = 500000,      // kg (like Falcon 9)
      propellantMass = 400000,   // kg
      thrust = 7600000,          // N
      exhaustVelocity = 3000,    // m/s
      dragCoefficient = 0.3,
      crossSection = 10,         // m²
      targetAltitude = 400000,   // m (400 km)
      dt = 0.1,                  // s
      maxTime = 600              // s
    } = params;

    const g0 = 9.81;
    const Re = 6.371e6;  // Earth radius

    // Simulation state
    let altitude = 0;
    let velocity = 0;
    let mass = initialMass;
    let time = 0;
    let propellantRemaining = propellantMass;

    const timeSeries: { time: number; state: Record<string, any> }[] = [];

    // Launch simulation
    while (time < maxTime && altitude >= 0) {
      // Atmospheric conditions
      const atm = AtmosphericPhysics.getAtmosphericState(altitude);

      // Gravity (decreases with altitude)
      const r = Re + altitude;
      const g = g0 * (Re / r) ** 2;

      // Drag force
      const dynamicPressure = 0.5 * atm.density * velocity ** 2;
      const drag = dynamicPressure * crossSection * dragCoefficient;

      // Thrust (only while propellant available)
      let currentThrust = 0;
      let massFlowRate = 0;
      if (propellantRemaining > 0) {
        currentThrust = thrust;
        massFlowRate = thrust / exhaustVelocity;
      }

      // Acceleration
      const acceleration = (currentThrust - drag) / mass - g;

      // Update state
      velocity += acceleration * dt;
      altitude += velocity * dt;
      mass -= massFlowRate * dt;
      propellantRemaining -= massFlowRate * dt;
      time += dt;

      // Record state
      if (Math.floor(time) % 10 === 0 || time < 10) {
        timeSeries.push({
          time,
          state: {
            altitude,
            velocity,
            mass,
            acceleration,
            dynamicPressure,
            machNumber: velocity / atm.speedOfSound,
            drag,
            thrust: currentThrust,
            gravity: g
          }
        });
      }

      // Check if orbit achieved
      if (altitude >= targetAltitude && velocity >= Math.sqrt(g0 * Re ** 2 / r)) {
        break;
      }
    }

    // Calculate orbital parameters if orbit achieved
    const finalR = Re + altitude;
    const orbitalVelocity = Math.sqrt(g0 * Re ** 2 / finalR);
    const orbitAchieved = velocity >= orbitalVelocity * 0.95;

    results.push({
      name: 'rocket_launch',
      success: orbitAchieved,
      duration: time,
      data: {
        finalAltitude: altitude,
        finalVelocity: velocity,
        finalMass: mass,
        propellantUsed: propellantMass - propellantRemaining,
        orbitalVelocityRequired: orbitalVelocity,
        deltaV: exhaustVelocity * Math.log(initialMass / mass)
      },
      timeSeries,
      summary: {
        maxAltitude: Math.max(...timeSeries.map(t => t.state.altitude)),
        maxVelocity: Math.max(...timeSeries.map(t => t.state.velocity)),
        maxQ: Math.max(...timeSeries.map(t => t.state.dynamicPressure)),
        maxAcceleration: Math.max(...timeSeries.map(t => t.state.acceleration)),
        burnTime: propellantMass / (thrust / exhaustVelocity)
      }
    });

    return {
      scenario: 'rocket_launch',
      results,
      totalDuration: Date.now() - startTime,
      summary: {
        orbitAchieved,
        finalAltitude: altitude,
        finalVelocity: velocity
      }
    };
  }

  /**
   * Satellite orbit simulation with perturbations
   */
  private async runSatelliteOrbitScenario(params: Record<string, any>): Promise<ScenarioResult> {
    const startTime = Date.now();
    const results: SimulationResult[] = [];

    const {
      semiMajorAxis = 7000000,   // m (about 630 km altitude)
      eccentricity = 0.001,
      inclination = 0.9,         // rad (~51.6° like ISS)
      duration = 86400,          // s (1 day)
      dt = 10,                   // s
      includeJ2 = true,
      includeDrag = true,
      satelliteMass = 1000,      // kg
      crossSection = 10          // m²
    } = params;

    const elements = {
      semiMajorAxis,
      eccentricity,
      inclination,
      longitudeOfAscendingNode: 0,
      argumentOfPeriapsis: 0,
      trueAnomaly: 0
    };

    const initialState = OrbitalMechanics.elementsToState(elements);

    const Earth: CelestialBody = {
      name: 'Earth',
      mass: 5.972e24,
      radius: 6.371e6,
      position: { x: 0, y: 0, z: 0 },
      velocity: { x: 0, y: 0, z: 0 },
      mu: 3.986004418e14,
      J2: 1.08263e-3,
      atmosphereHeight: 8500
    };

    // Propagate orbit with perturbations
    const trajectory = OrbitalMechanics.propagateOrbitWithPerturbations(
      initialState.position,
      initialState.velocity,
      Earth,
      dt,
      duration,
      {
        includeJ2,
        includeDrag,
        dragCoefficient: 2.2,
        crossSection,
        mass: satelliteMass
      }
    );

    // Analyze results
    const altitudes = trajectory.map(t => {
      const r = Math.sqrt(t.position.x ** 2 + t.position.y ** 2 + t.position.z ** 2);
      return r - Earth.radius;
    });

    const velocities = trajectory.map(t =>
      Math.sqrt(t.velocity.x ** 2 + t.velocity.y ** 2 + t.velocity.z ** 2)
    );

    // Calculate ground track
    const groundTrack = OrbitalMechanics.groundTrack(
      elements,
      { radius: Earth.radius, rotationPeriod: 86164 },
      duration,
      60
    );

    results.push({
      name: 'satellite_orbit',
      success: true,
      duration: Date.now() - startTime,
      iterations: trajectory.length,
      data: {
        trajectory: trajectory.filter((_, i) => i % 10 === 0),
        groundTrack: groundTrack.filter((_, i) => i % 5 === 0),
        orbitalElements: elements
      },
      timeSeries: trajectory.filter((_, i) => i % 100 === 0).map(t => ({
        time: t.time,
        state: {
          position: t.position,
          velocity: t.velocity,
          altitude: Math.sqrt(t.position.x ** 2 + t.position.y ** 2 + t.position.z ** 2) - Earth.radius
        }
      })),
      summary: {
        minAltitude: Math.min(...altitudes),
        maxAltitude: Math.max(...altitudes),
        meanAltitude: altitudes.reduce((a, b) => a + b, 0) / altitudes.length,
        orbitalPeriod: 2 * Math.PI * Math.sqrt(semiMajorAxis ** 3 / Earth.mu!),
        orbitsCompleted: duration / (2 * Math.PI * Math.sqrt(semiMajorAxis ** 3 / Earth.mu!))
      }
    });

    return {
      scenario: 'satellite_orbit',
      results,
      totalDuration: Date.now() - startTime,
      summary: {
        decayRate: (altitudes[0] - altitudes[altitudes.length - 1]) / duration * 86400  // m/day
      }
    };
  }

  /**
   * Particle accelerator simulation combining EM and relativistic dynamics
   */
  private async runParticleAcceleratorScenario(params: Record<string, any>): Promise<ScenarioResult> {
    const startTime = Date.now();
    const results: SimulationResult[] = [];

    const {
      particleType = 'proton',
      initialEnergy = 1e9,       // eV
      targetEnergy = 1e12,       // eV (1 TeV)
      magneticFieldStrength = 8, // T
      acceleratorRadius = 4300,  // m (like LHC)
      accelerationPerTurn = 0.5e6, // eV
      dt = 1e-9                  // s
    } = params;

    const c = PhysicalConstants.get('speed_of_light').value;
    const eV = 1.602e-19;
    const mp = 1.673e-27;  // proton mass

    let energy = initialEnergy * eV;  // Convert to Joules
    const restEnergy = mp * c * c;

    const timeSeries: { time: number; state: Record<string, any> }[] = [];
    let time = 0;
    let turns = 0;

    while (energy < targetEnergy * eV) {
      // Relativistic gamma
      const gamma = energy / restEnergy;
      const beta = Math.sqrt(1 - 1 / (gamma * gamma));
      const velocity = beta * c;

      // Momentum
      const momentum = gamma * mp * velocity;

      // Cyclotron frequency (relativistic)
      const omega = velocity / acceleratorRadius;
      const period = 2 * Math.PI / omega;

      // Synchrotron radiation power
      const synchrotronPower = (c * eV * eV / (6 * Math.PI * 8.85e-12)) *
                                Math.pow(gamma, 4) / Math.pow(acceleratorRadius, 2);

      // Energy after one turn
      energy += accelerationPerTurn * eV - synchrotronPower * period;
      turns++;
      time += period;

      if (turns % 10000 === 0) {
        timeSeries.push({
          time,
          state: {
            energy: energy / eV,
            gamma,
            beta,
            velocity,
            momentum,
            synchrotronPower,
            period
          }
        });
      }

      // Safety limit
      if (turns > 1e8) break;
    }

    results.push({
      name: 'particle_accelerator',
      success: energy >= targetEnergy * eV,
      duration: time,
      iterations: turns,
      data: {
        finalEnergy: energy / eV,
        totalTurns: turns,
        totalDistance: turns * 2 * Math.PI * acceleratorRadius
      },
      timeSeries,
      summary: {
        accelerationTime: time,
        finalGamma: energy / restEnergy,
        finalBeta: Math.sqrt(1 - Math.pow(restEnergy / energy, 2)),
        turnsRequired: turns,
        energyGainPerTurn: accelerationPerTurn
      }
    });

    return {
      scenario: 'particle_accelerator',
      results,
      totalDuration: Date.now() - startTime,
      summary: { energyAchieved: energy / eV }
    };
  }

  /**
   * Nuclear reactor simulation
   */
  private async runNuclearReactorScenario(params: Record<string, any>): Promise<ScenarioResult> {
    const startTime = Date.now();
    const results: SimulationResult[] = [];

    const {
      fuelMass = 100000,        // kg UO2
      enrichment = 0.035,       // 3.5% U-235
      thermalPower = 3000e6,    // W (3 GW thermal)
      coolantInletTemp = 290 + 273.15,  // K
      coolantFlowRate = 15000,  // kg/s
      simulationTime = 86400 * 30,  // 30 days
      dt = 3600                 // 1 hour timesteps
    } = params;

    // Calculate initial quantities
    const U235mass = fuelMass * 0.88 * enrichment;  // 88% U in UO2
    const NA = 6.022e23;
    let N_U235 = U235mass * NA / 0.235;

    // Fission parameters
    const fissionEnergy = NuclearPhysics.fissionEnergy('U-235');
    const energyPerFission = fissionEnergy.totalEnergy * 1.602e-13;  // J

    // Calculate fission rate for desired power
    const fissionRate = thermalPower / energyPerFission;  // fissions/s

    const timeSeries: { time: number; state: Record<string, any> }[] = [];
    let time = 0;
    let coolantTemp = coolantInletTemp;
    let fuelTemp = coolantInletTemp + 500;  // Initial fuel temp

    while (time < simulationTime) {
      // Burnup calculation
      const fissionsThisStep = fissionRate * dt;
      N_U235 -= fissionsThisStep;

      // Reactivity feedback (simplified)
      const reactivity = (N_U235 / (U235mass * NA / 0.235) - 0.5) * 0.01;

      // Heat transfer to coolant
      const specificHeat = 5000;  // J/(kg·K) for water
      const coolantOutletTemp = coolantInletTemp + thermalPower / (coolantFlowRate * specificHeat);

      // Fuel temperature (simplified)
      const heatTransferCoeff = 10000;  // W/(m²·K)
      const fuelSurfaceArea = 50000;    // m²
      fuelTemp = coolantOutletTemp + thermalPower / (heatTransferCoeff * fuelSurfaceArea);

      time += dt;

      if (Math.floor(time / 86400) !== Math.floor((time - dt) / 86400)) {
        timeSeries.push({
          time,
          state: {
            U235remaining: N_U235 * 0.235 / NA,
            burnup: 1 - N_U235 / (U235mass * NA / 0.235),
            fuelTemp,
            coolantOutletTemp,
            reactivity,
            power: thermalPower
          }
        });
      }
    }

    const burnup = 1 - N_U235 / (U235mass * NA / 0.235);

    results.push({
      name: 'nuclear_reactor',
      success: true,
      duration: simulationTime,
      iterations: Math.floor(simulationTime / dt),
      data: {
        initialU235: U235mass,
        finalU235: N_U235 * 0.235 / NA,
        burnup,
        totalEnergy: thermalPower * simulationTime
      },
      timeSeries,
      summary: {
        burnupPercentage: burnup * 100,
        averageFuelTemp: fuelTemp,
        thermalEfficiency: 0.33,  // Typical for PWR
        electricOutput: thermalPower * 0.33
      }
    });

    return {
      scenario: 'nuclear_reactor',
      results,
      totalDuration: Date.now() - startTime,
      summary: { burnup: burnup * 100 }
    };
  }

  /**
   * Heat exchanger design and simulation
   */
  private async runHeatExchangerScenario(params: Record<string, any>): Promise<ScenarioResult> {
    const startTime = Date.now();
    const results: SimulationResult[] = [];

    const {
      heatDuty = 1e6,           // W
      hotInletTemp = 400,       // K
      coldInletTemp = 300,      // K
      optimize = true
    } = params;

    if (optimize) {
      // Use PSO to optimize design
      const optimResult = PhysicsOptimization.optimizeHeatExchanger({
        heatDuty,
        hotInletTemp,
        coldInletTemp,
        maxPressureDrop: 50000
      });

      results.push({
        name: 'heat_exchanger_optimization',
        success: optimResult.feasible,
        duration: Date.now() - startTime,
        iterations: optimResult.iterations,
        data: {
          optimalDiameter: optimResult.bestPosition[0],
          optimalLength: optimResult.bestPosition[1],
          optimalTubes: Math.round(optimResult.bestPosition[2]),
          cost: optimResult.bestFitness,
          convergence: optimResult.convergenceHistory
        },
        timeSeries: [],
        summary: {
          tubeDiameter: optimResult.bestPosition[0] * 1000,  // mm
          tubeLength: optimResult.bestPosition[1],
          numberOfTubes: Math.round(optimResult.bestPosition[2]),
          constraintViolation: optimResult.constraintViolation
        }
      });
    }

    // Run thermal simulation with design
    const design = {
      tubeDiameter: 0.025,
      tubeLength: 5,
      numberOfTubes: 200
    };

    const U = 500;  // Overall heat transfer coefficient
    const A = Math.PI * design.tubeDiameter * design.tubeLength * design.numberOfTubes;
    const LMTD = ((hotInletTemp - coldInletTemp) - (hotInletTemp - 50 - coldInletTemp - 50)) /
                  Math.log((hotInletTemp - coldInletTemp) / (hotInletTemp - 50 - coldInletTemp - 50));
    const actualHeatDuty = U * A * Math.abs(LMTD);

    results.push({
      name: 'heat_exchanger_simulation',
      success: actualHeatDuty >= heatDuty * 0.9,
      duration: Date.now() - startTime,
      data: {
        design,
        surfaceArea: A,
        LMTD,
        actualHeatDuty,
        effectiveness: actualHeatDuty / heatDuty
      },
      timeSeries: [],
      summary: {
        surfaceArea: A,
        heatDutyAchieved: actualHeatDuty,
        effectiveness: actualHeatDuty / heatDuty
      }
    });

    return {
      scenario: 'heat_exchanger',
      results,
      totalDuration: Date.now() - startTime,
      summary: { effectiveness: actualHeatDuty / heatDuty }
    };
  }

  /**
   * Laser cavity simulation
   */
  private async runLaserCavityScenario(params: Record<string, any>): Promise<ScenarioResult> {
    const startTime = Date.now();
    const results: SimulationResult[] = [];

    const {
      wavelength = 632.8e-9,    // He-Ne laser
      cavityLength = 0.3,       // m
      mirrorReflectivity = 0.99,
      gainMediumLength = 0.1,   // m
      smallSignalGain = 0.1,    // per pass
      pumpPower = 10            // W
    } = params;

    const c = PhysicalConstants.get('speed_of_light').value;

    // Cavity parameters
    const roundTripTime = 2 * cavityLength / c;
    const roundTripLoss = 1 - mirrorReflectivity * mirrorReflectivity;

    // Threshold condition
    const thresholdGain = roundTripLoss / (2 * gainMediumLength);

    // Steady-state analysis
    const isSingleMode = cavityLength < c / (2 * 1e9);  // < 1 GHz bandwidth
    const modeSpacing = c / (2 * cavityLength);
    const numberOfModes = Math.floor(1e9 / modeSpacing);  // Assuming 1 GHz bandwidth

    // Output power calculation (simplified)
    const saturationIntensity = 1000;  // W/m²
    const modeArea = 1e-6;  // m²
    const outputCoupling = 1 - mirrorReflectivity;
    const outputPower = pumpPower * 0.1 * outputCoupling;  // 10% efficiency estimate

    // Beam quality
    const M2 = 1.0 + 0.1 * numberOfModes;  // Degrades with more modes
    const divergence = M2 * wavelength / (Math.PI * Math.sqrt(modeArea));

    results.push({
      name: 'laser_cavity',
      success: smallSignalGain > thresholdGain,
      duration: Date.now() - startTime,
      data: {
        wavelength,
        cavityLength,
        roundTripTime,
        roundTripLoss,
        thresholdGain,
        modeSpacing,
        numberOfModes,
        outputPower,
        M2,
        divergence
      },
      timeSeries: [],
      summary: {
        thresholdGain,
        isAboveThreshold: smallSignalGain > thresholdGain,
        outputPower,
        beamQualityM2: M2,
        divergenceRad: divergence
      }
    });

    return {
      scenario: 'laser_cavity',
      results,
      totalDuration: Date.now() - startTime,
      summary: { outputPower, beamQuality: M2 }
    };
  }

  /**
   * Plasma confinement simulation (tokamak)
   */
  private async runPlasmaConfinementScenario(params: Record<string, any>): Promise<ScenarioResult> {
    const startTime = Date.now();
    const results: SimulationResult[] = [];

    const {
      majorRadius = 6.2,        // m (ITER-like)
      minorRadius = 2.0,        // m
      toroidalField = 5.3,      // T
      plasmaCurrent = 15e6,     // A
      temperature = 15,         // keV
      density = 1e20            // particles/m³
    } = params;

    const kB = 1.38e-23;
    const mu0 = 4 * Math.PI * 1e-7;

    // Plasma parameters
    const plasmaVolume = 2 * Math.PI * Math.PI * majorRadius * minorRadius * minorRadius;
    const betaPoloidal = 2 * mu0 * density * temperature * 1000 * 1.6e-19 /
                         Math.pow(mu0 * plasmaCurrent / (2 * Math.PI * minorRadius), 2);

    // Confinement time (ITER98 scaling)
    const IP = plasmaCurrent / 1e6;  // MA
    const BT = toroidalField;
    const n19 = density / 1e19;
    const P = density * plasmaVolume * temperature * 1000 * 1.6e-19 / 1e6;  // MW estimate
    const tauE = 0.0562 * Math.pow(IP, 0.93) * Math.pow(BT, 0.15) *
                 Math.pow(n19, 0.41) * Math.pow(P, -0.69) *
                 Math.pow(majorRadius, 1.97) * Math.pow(minorRadius / majorRadius, 0.58);

    // Triple product
    const tripleProduct = density * temperature * tauE;
    const ignitionThreshold = 3e21;  // m⁻³ keV s

    // Fusion power
    const fusionReaction = NuclearPhysics.fusionReactionRate('DT', temperature, density);

    results.push({
      name: 'plasma_confinement',
      success: tripleProduct > ignitionThreshold * 0.1,  // Q > 10 level
      duration: Date.now() - startTime,
      data: {
        geometry: { majorRadius, minorRadius, volume: plasmaVolume },
        fields: { toroidalField, plasmaCurrent },
        plasma: { temperature, density },
        performance: {
          confinementTime: tauE,
          tripleProduct,
          ignitionFraction: tripleProduct / ignitionThreshold,
          betaPoloidal
        },
        fusion: fusionReaction
      },
      timeSeries: [],
      summary: {
        confinementTime: tauE,
        tripleProduct,
        Q: tripleProduct / ignitionThreshold * 10,
        fusionPowerDensity: fusionReaction.powerDensity
      }
    });

    return {
      scenario: 'plasma_confinement',
      results,
      totalDuration: Date.now() - startTime,
      summary: { Q: tripleProduct / ignitionThreshold * 10 }
    };
  }

  /**
   * Atmospheric reentry simulation
   */
  private async runAtmosphericReentryScenario(params: Record<string, any>): Promise<ScenarioResult> {
    const startTime = Date.now();
    const results: SimulationResult[] = [];

    const {
      entryVelocity = 7800,     // m/s (orbital velocity)
      entryAltitude = 120000,   // m
      entryAngle = -1.5,        // degrees
      vehicleMass = 5000,       // kg
      noseRadius = 1.0,         // m
      liftToDrag = 0.3,
      dragCoefficient = 1.5,
      referenceArea = 10,       // m²
      dt = 0.1                  // s
    } = params;

    const g0 = 9.81;
    const Re = 6.371e6;

    let altitude = entryAltitude;
    let velocity = entryVelocity;
    let flightPathAngle = entryAngle * Math.PI / 180;
    let range = 0;
    let time = 0;

    const timeSeries: { time: number; state: Record<string, any> }[] = [];

    while (altitude > 0 && altitude < 200000) {
      // Atmospheric conditions
      const atm = AtmosphericPhysics.getAtmosphericState(altitude);

      // Gravity
      const r = Re + altitude;
      const g = g0 * (Re / r) ** 2;

      // Aerodynamic forces
      const aero = AtmosphericPhysics.calculateAerodynamics(
        velocity, altitude, referenceArea, dragCoefficient
      );

      const drag = aero.drag / vehicleMass;
      const lift = drag * liftToDrag;

      // Heating
      const heating = AtmosphericPhysics.calculateReentryHeating(
        velocity, altitude, noseRadius, vehicleMass
      );

      // Equations of motion (planar)
      const dVdt = -drag - g * Math.sin(flightPathAngle);
      const dGammaVdt = lift - (g - velocity * velocity / r) * Math.cos(flightPathAngle);
      const dGammadt = dGammaVdt / velocity;
      const dAltdt = velocity * Math.sin(flightPathAngle);
      const dRangedt = velocity * Math.cos(flightPathAngle) * Re / r;

      // Update state
      velocity += dVdt * dt;
      flightPathAngle += dGammadt * dt;
      altitude += dAltdt * dt;
      range += dRangedt * dt;
      time += dt;

      // Record state
      if (Math.floor(time * 10) % 10 === 0) {
        timeSeries.push({
          time,
          state: {
            altitude,
            velocity,
            flightPathAngle: flightPathAngle * 180 / Math.PI,
            range,
            deceleration: drag / g0,
            heatFlux: heating.heatFlux,
            dynamicPressure: aero.dynamicPressure,
            machNumber: aero.machNumber
          }
        });
      }

      // Safety check
      if (time > 2000) break;
    }

    // Find peak values
    const peakDeceleration = Math.max(...timeSeries.map(t => t.state.deceleration));
    const peakHeatFlux = Math.max(...timeSeries.map(t => t.state.heatFlux));
    const peakDynamicPressure = Math.max(...timeSeries.map(t => t.state.dynamicPressure));

    results.push({
      name: 'atmospheric_reentry',
      success: altitude <= 0 && velocity < 200,
      duration: time,
      iterations: timeSeries.length,
      data: {
        finalAltitude: altitude,
        finalVelocity: velocity,
        totalRange: range,
        landingLocation: range * 180 / (Math.PI * Re)  // degrees
      },
      timeSeries,
      summary: {
        flightTime: time,
        peakDeceleration,
        peakHeatFlux,
        peakDynamicPressure,
        totalRange: range / 1000  // km
      }
    });

    return {
      scenario: 'atmospheric_reentry',
      results,
      totalDuration: Date.now() - startTime,
      summary: {
        survivalProbability: peakDeceleration < 15 && peakHeatFlux < 5e6 ? 'HIGH' : 'LOW'
      }
    };
  }

  /**
   * Run a custom simulation configuration
   */
  async runCustomSimulation(config: SimulationConfig): Promise<SimulationResult> {
    const startTime = Date.now();

    try {
      const data: Record<string, any> = {};
      const warnings: string[] = [];

      // Execute requested modules
      for (const module of config.modules) {
        switch (module) {
          case 'quantum':
            if (config.parameters.photonWavelength) {
              data.photon = QuantumMechanics.photonProperties(config.parameters.photonWavelength);
            }
            if (config.parameters.hydrogenN) {
              data.hydrogen = QuantumMechanics.hydrogenAtom(config.parameters.hydrogenN);
            }
            break;

          case 'particles':
            if (config.parameters.particles) {
              data.nBody = ParticleDynamics.nBodySimulation(
                config.parameters.particles,
                config.parameters.dt || 0.001,
                config.parameters.duration || 10
              );
            }
            break;

          case 'waves':
            if (config.parameters.frequency) {
              data.standing = WaveSimulation.standingWave(
                config.parameters.amplitude || 1,
                config.parameters.wavelength || 1,
                config.parameters.length || 2,
                0
              );
            }
            break;

          case 'atmosphere':
            if (config.parameters.altitude !== undefined) {
              data.atmosphere = AtmosphericPhysics.getAtmosphericState(config.parameters.altitude);
            }
            break;

          case 'fluids':
            if (config.parameters.fluidVelocity) {
              data.flow = FluidDynamics.pipeFlow(
                config.parameters.fluidVelocity,
                config.parameters.pipeDiameter || 0.1,
                { density: 1000, viscosity: 0.001, specificHeat: 4186, thermalConductivity: 0.6 }
              );
            }
            break;

          case 'electromagnetism':
            if (config.parameters.charge) {
              data.field = Electromagnetism.electricField(
                config.parameters.charge,
                config.parameters.position || { x: 1, y: 0, z: 0 }
              );
            }
            break;

          case 'thermodynamics':
            if (config.parameters.temperatures) {
              data.conduction = Thermodynamics.conductionSteadyState(
                config.parameters.temperatures,
                config.parameters.thermalConductivity || 1,
                config.parameters.area || 1,
                config.parameters.thickness || 0.1
              );
            }
            break;

          case 'orbital':
            if (config.parameters.orbitalElements) {
              data.orbit = OrbitalMechanics.elementsToState(config.parameters.orbitalElements);
            }
            break;

          case 'nuclear':
            if (config.parameters.halfLife) {
              data.decay = NuclearPhysics.decay(
                config.parameters.initialNuclei || 1e6,
                config.parameters.halfLife,
                config.parameters.time || config.parameters.halfLife
              );
            }
            break;

          case 'statistical':
            if (config.parameters.temperature && config.parameters.mass) {
              data.maxwell = StatisticalMechanics.maxwellBoltzmannVelocity(
                config.parameters.temperature,
                config.parameters.mass
              );
            }
            break;

          default:
            warnings.push(`Unknown module: ${module}`);
        }
      }

      return {
        name: config.name,
        success: true,
        duration: Date.now() - startTime,
        data,
        summary: {},
        warnings: warnings.length > 0 ? warnings : undefined
      };

    } catch (error) {
      return {
        name: config.name,
        success: false,
        duration: Date.now() - startTime,
        data: {},
        summary: {},
        errors: [(error as Error).message]
      };
    }
  }

  /**
   * Get list of available scenarios
   */
  getAvailableScenarios(): { name: string; description: string; requiredParams: string[] }[] {
    return [
      {
        name: 'rocket_launch',
        description: 'Simulate rocket launch from ground to orbit',
        requiredParams: ['initialMass', 'thrust', 'targetAltitude']
      },
      {
        name: 'satellite_orbit',
        description: 'Satellite orbit propagation with perturbations',
        requiredParams: ['semiMajorAxis', 'eccentricity', 'inclination']
      },
      {
        name: 'particle_accelerator',
        description: 'Particle acceleration in a synchrotron',
        requiredParams: ['initialEnergy', 'targetEnergy', 'magneticFieldStrength']
      },
      {
        name: 'nuclear_reactor',
        description: 'Nuclear reactor power generation simulation',
        requiredParams: ['fuelMass', 'enrichment', 'thermalPower']
      },
      {
        name: 'heat_exchanger',
        description: 'Heat exchanger design and optimization',
        requiredParams: ['heatDuty', 'hotInletTemp', 'coldInletTemp']
      },
      {
        name: 'laser_cavity',
        description: 'Laser cavity mode analysis',
        requiredParams: ['wavelength', 'cavityLength', 'mirrorReflectivity']
      },
      {
        name: 'plasma_confinement',
        description: 'Tokamak plasma confinement analysis',
        requiredParams: ['majorRadius', 'toroidalField', 'plasmaCurrent']
      },
      {
        name: 'atmospheric_reentry',
        description: 'Spacecraft atmospheric reentry simulation',
        requiredParams: ['entryVelocity', 'entryAltitude', 'vehicleMass']
      }
    ];
  }

  /**
   * Get available physics modules
   */
  getAvailableModules(): string[] {
    return [
      'quantum',
      'particles',
      'waves',
      'atmosphere',
      'fluids',
      'electromagnetism',
      'thermodynamics',
      'orbital',
      'nuclear',
      'statistical'
    ];
  }
}

export default SimulationOrchestrator;
