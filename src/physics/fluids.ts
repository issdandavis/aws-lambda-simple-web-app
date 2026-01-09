/**
 * Fluid Dynamics Module
 * Implements fluid mechanics, pipe flow, and basic hydrodynamics
 */

import { PhysicalConstants as PC } from './constants';

// Common fluid properties at 20°C
export const FluidProperties = {
  water: {
    density: 998.2,          // kg/m³
    dynamicViscosity: 1.002e-3,  // Pa·s
    kinematicViscosity: 1.004e-6, // m²/s
    bulkModulus: 2.2e9,      // Pa
    surfaceTension: 0.0728,  // N/m
  },
  air: {
    density: 1.204,
    dynamicViscosity: 1.825e-5,
    kinematicViscosity: 1.516e-5,
    bulkModulus: 1.42e5,
    surfaceTension: 0,
  },
  seawater: {
    density: 1025,
    dynamicViscosity: 1.08e-3,
    kinematicViscosity: 1.05e-6,
    bulkModulus: 2.34e9,
    surfaceTension: 0.0736,
  },
  oil: {
    density: 900,
    dynamicViscosity: 0.03,
    kinematicViscosity: 3.33e-5,
    bulkModulus: 1.5e9,
    surfaceTension: 0.032,
  },
} as const;

export interface FlowProperties {
  velocity: number;
  reynoldsNumber: number;
  flowRegime: 'laminar' | 'transitional' | 'turbulent';
  frictionFactor: number;
  pressureDrop: number;
  volumetricFlowRate: number;
  massFlowRate: number;
}

export interface BernoulliResult {
  velocity1: number;
  velocity2: number;
  pressure1: number;
  pressure2: number;
  height1: number;
  height2: number;
  totalHead: number;
  dynamicPressure1: number;
  dynamicPressure2: number;
}

export interface DragResult {
  dragForce: number;
  dragCoefficient: number;
  reynoldsNumber: number;
  terminalVelocity: number;
}

export interface WaveResult {
  wavelength: number;
  period: number;
  celerity: number;     // Wave phase speed
  groupVelocity: number;
  amplitude: number;
  energy: number;       // J/m² (energy per unit surface area)
}

export interface PumpResult {
  head: number;          // m
  power: number;         // W
  efficiency: number;
  npshRequired: number;  // Net Positive Suction Head
  specificSpeed: number;
}

/**
 * Fluid Dynamics Calculator
 */
export class FluidDynamics {

  /**
   * Calculate Reynolds number
   * Re = ρvL/μ = vL/ν
   */
  static reynoldsNumber(
    velocity: number,
    characteristicLength: number,
    kinematicViscosity: number
  ): number {
    return velocity * characteristicLength / kinematicViscosity;
  }

  /**
   * Determine flow regime from Reynolds number
   */
  static flowRegime(reynoldsNumber: number): 'laminar' | 'transitional' | 'turbulent' {
    if (reynoldsNumber < 2300) return 'laminar';
    if (reynoldsNumber < 4000) return 'transitional';
    return 'turbulent';
  }

  /**
   * Calculate Darcy friction factor for pipe flow
   * Uses Churchill equation (valid for all regimes)
   */
  static darcyFrictionFactor(
    reynoldsNumber: number,
    relativeroughness: number = 0  // ε/D
  ): number {
    if (reynoldsNumber < 2300) {
      // Laminar flow: f = 64/Re
      return 64 / reynoldsNumber;
    }

    // Churchill equation for turbulent flow
    const A = Math.pow(2.457 * Math.log(1 / (Math.pow(7 / reynoldsNumber, 0.9) + 0.27 * relativeroughness)), 16);
    const B = Math.pow(37530 / reynoldsNumber, 16);

    return 8 * Math.pow(Math.pow(8 / reynoldsNumber, 12) + 1 / Math.pow(A + B, 1.5), 1 / 12);
  }

  /**
   * Calculate pipe flow properties
   * Darcy-Weisbach equation: ΔP = f * (L/D) * (ρv²/2)
   */
  static pipeFlow(
    diameter: number,         // m
    length: number,           // m
    flowRate: number,         // m³/s
    density: number,          // kg/m³
    kinematicViscosity: number, // m²/s
    roughness: number = 0     // m (pipe surface roughness)
  ): FlowProperties {
    const area = Math.PI * diameter * diameter / 4;
    const velocity = flowRate / area;

    const reynoldsNumber = this.reynoldsNumber(velocity, diameter, kinematicViscosity);
    const flowRegime = this.flowRegime(reynoldsNumber);

    const relativeRoughness = roughness / diameter;
    const frictionFactor = this.darcyFrictionFactor(reynoldsNumber, relativeRoughness);

    // Pressure drop: ΔP = f * (L/D) * (ρv²/2)
    const pressureDrop = frictionFactor * (length / diameter) * (0.5 * density * velocity * velocity);

    const massFlowRate = density * flowRate;

    return {
      velocity,
      reynoldsNumber,
      flowRegime,
      frictionFactor,
      pressureDrop,
      volumetricFlowRate: flowRate,
      massFlowRate,
    };
  }

  /**
   * Bernoulli equation solver
   * P₁ + ½ρv₁² + ρgh₁ = P₂ + ½ρv₂² + ρgh₂
   */
  static bernoulli(
    density: number,
    velocity1: number,
    pressure1: number,
    height1: number,
    velocity2?: number,
    pressure2?: number,
    height2?: number
  ): BernoulliResult {
    const g = 9.80665;

    // Total head (constant along streamline for ideal flow)
    const dynamicPressure1 = 0.5 * density * velocity1 * velocity1;
    const totalHead = pressure1 + dynamicPressure1 + density * g * height1;

    // Solve for unknowns
    let v2 = velocity2;
    let p2 = pressure2;
    let h2 = height2;

    if (v2 === undefined && p2 !== undefined && h2 !== undefined) {
      // Solve for v2
      const dynamicPressure2 = totalHead - p2 - density * g * h2;
      v2 = Math.sqrt(2 * dynamicPressure2 / density);
    } else if (p2 === undefined && v2 !== undefined && h2 !== undefined) {
      // Solve for p2
      const dynamicPressure2 = 0.5 * density * v2 * v2;
      p2 = totalHead - dynamicPressure2 - density * g * h2;
    } else if (h2 === undefined && v2 !== undefined && p2 !== undefined) {
      // Solve for h2
      const dynamicPressure2 = 0.5 * density * v2 * v2;
      h2 = (totalHead - p2 - dynamicPressure2) / (density * g);
    } else {
      // All specified or not enough info
      v2 = v2 ?? velocity1;
      p2 = p2 ?? pressure1;
      h2 = h2 ?? height1;
    }

    const dynamicPressure2 = 0.5 * density * v2 * v2;

    return {
      velocity1,
      velocity2: v2,
      pressure1,
      pressure2: p2,
      height1,
      height2: h2,
      totalHead,
      dynamicPressure1,
      dynamicPressure2,
    };
  }

  /**
   * Calculate drag force on a sphere
   */
  static sphereDrag(
    diameter: number,
    velocity: number,
    fluidDensity: number,
    kinematicViscosity: number
  ): DragResult {
    const reynoldsNumber = this.reynoldsNumber(velocity, diameter, kinematicViscosity);

    // Drag coefficient for sphere (empirical correlations)
    let dragCoefficient: number;

    if (reynoldsNumber < 1) {
      // Stokes regime: Cd = 24/Re
      dragCoefficient = 24 / reynoldsNumber;
    } else if (reynoldsNumber < 1000) {
      // Intermediate regime
      dragCoefficient = 24 / reynoldsNumber * (1 + 0.15 * Math.pow(reynoldsNumber, 0.687));
    } else if (reynoldsNumber < 2e5) {
      // Newton's regime
      dragCoefficient = 0.44;
    } else {
      // Supercritical (drag crisis)
      dragCoefficient = 0.1;
    }

    const area = Math.PI * diameter * diameter / 4;
    const dragForce = 0.5 * fluidDensity * velocity * velocity * dragCoefficient * area;

    // Terminal velocity (when drag = weight)
    // Assuming a solid sphere with density 2500 kg/m³ (typical for particles)
    const sphereDensity = 2500;
    const mass = sphereDensity * (4 / 3) * Math.PI * Math.pow(diameter / 2, 3);
    const weight = mass * 9.80665;
    const terminalVelocity = Math.sqrt(2 * weight / (fluidDensity * dragCoefficient * area));

    return {
      dragForce,
      dragCoefficient,
      reynoldsNumber,
      terminalVelocity,
    };
  }

  /**
   * Orifice flow calculation
   * Q = Cd * A * sqrt(2 * ΔP / ρ)
   */
  static orificeFlow(
    orificeDiameter: number,      // m
    upstreamDiameter: number,     // m
    pressureDrop: number,         // Pa
    density: number,              // kg/m³
    dischargeCoefficient: number = 0.61  // Typical for sharp-edged orifice
  ): { flowRate: number; velocity: number; beta: number } {
    const beta = orificeDiameter / upstreamDiameter;
    const area = Math.PI * orificeDiameter * orificeDiameter / 4;

    // Account for velocity of approach
    const E = 1 / Math.sqrt(1 - Math.pow(beta, 4));

    const velocity = E * dischargeCoefficient * Math.sqrt(2 * pressureDrop / density);
    const flowRate = area * velocity;

    return { flowRate, velocity, beta };
  }

  /**
   * Venturi meter flow calculation
   */
  static venturiFlow(
    inletDiameter: number,        // m
    throatDiameter: number,       // m
    pressureDrop: number,         // Pa
    density: number               // kg/m³
  ): { flowRate: number; inletVelocity: number; throatVelocity: number } {
    const Cd = 0.98;  // Venturi has higher Cd than orifice

    const A1 = Math.PI * inletDiameter * inletDiameter / 4;
    const A2 = Math.PI * throatDiameter * throatDiameter / 4;

    // From Bernoulli: v2 = sqrt(2*ΔP / (ρ*(1 - (A2/A1)²)))
    const areaRatio = A2 / A1;
    const throatVelocity = Cd * Math.sqrt(2 * pressureDrop / (density * (1 - areaRatio * areaRatio)));
    const flowRate = A2 * throatVelocity;
    const inletVelocity = flowRate / A1;

    return { flowRate, inletVelocity, throatVelocity };
  }

  /**
   * Open channel flow - Manning's equation
   * v = (1/n) * R^(2/3) * S^(1/2)
   */
  static manningFlow(
    hydraulicRadius: number,      // m (A/P where P is wetted perimeter)
    slope: number,                // m/m (channel slope)
    manningN: number = 0.013      // Manning's roughness (0.013 for concrete)
  ): { velocity: number; shearStress: number } {
    const velocity = (1 / manningN) * Math.pow(hydraulicRadius, 2 / 3) * Math.sqrt(slope);

    // Wall shear stress: τ = ρgRS
    const shearStress = 998.2 * 9.80665 * hydraulicRadius * slope;

    return { velocity, shearStress };
  }

  /**
   * Water wave properties (deep water waves)
   * Dispersion relation: ω² = gk (deep water)
   */
  static waterWave(
    wavelength: number,
    amplitude: number,
    waterDepth: number = Infinity
  ): WaveResult {
    const g = 9.80665;
    const k = 2 * Math.PI / wavelength;  // Wave number

    let celerity: number;
    let groupVelocity: number;

    if (waterDepth === Infinity || waterDepth > wavelength / 2) {
      // Deep water approximation
      celerity = Math.sqrt(g / k);
      groupVelocity = celerity / 2;
    } else if (waterDepth < wavelength / 20) {
      // Shallow water approximation
      celerity = Math.sqrt(g * waterDepth);
      groupVelocity = celerity;  // No dispersion in shallow water
    } else {
      // Intermediate depth - full dispersion relation
      celerity = Math.sqrt(g / k * Math.tanh(k * waterDepth));
      const n = 0.5 * (1 + 2 * k * waterDepth / Math.sinh(2 * k * waterDepth));
      groupVelocity = celerity * n;
    }

    const period = wavelength / celerity;

    // Wave energy per unit surface area: E = 0.5 * ρ * g * a²
    const density = 1025;  // Seawater
    const energy = 0.5 * density * g * amplitude * amplitude;

    return {
      wavelength,
      period,
      celerity,
      groupVelocity,
      amplitude,
      energy,
    };
  }

  /**
   * Pump head and power calculation
   */
  static pumpCalculation(
    flowRate: number,             // m³/s
    headRise: number,             // m
    efficiency: number = 0.75,    // Pump efficiency
    fluidDensity: number = 998.2  // kg/m³
  ): PumpResult {
    const g = 9.80665;

    // Hydraulic power: P_h = ρ * g * Q * H
    const hydraulicPower = fluidDensity * g * flowRate * headRise;

    // Shaft power required
    const power = hydraulicPower / efficiency;

    // Specific speed (dimensionless, for pump selection)
    // Ns = N * sqrt(Q) / H^0.75, assuming 1750 RPM
    const N = 1750;  // Typical pump speed (RPM)
    const specificSpeed = N * Math.sqrt(flowRate) / Math.pow(headRise, 0.75);

    // NPSH required (rough estimate)
    const npshRequired = 0.3 * Math.pow(specificSpeed, 4 / 3);

    return {
      head: headRise,
      power,
      efficiency,
      npshRequired,
      specificSpeed,
    };
  }

  /**
   * Buoyancy force calculation
   * F_b = ρ_fluid * V_displaced * g
   */
  static buoyancy(
    displacedVolume: number,      // m³
    fluidDensity: number          // kg/m³
  ): { buoyancyForce: number; massEquivalent: number } {
    const g = 9.80665;
    const buoyancyForce = fluidDensity * displacedVolume * g;
    const massEquivalent = fluidDensity * displacedVolume;

    return { buoyancyForce, massEquivalent };
  }

  /**
   * Hydrostatic pressure at depth
   * P = P_atm + ρgh
   */
  static hydrostaticPressure(
    depth: number,                // m
    fluidDensity: number = 998.2, // kg/m³
    atmosphericPressure: number = 101325  // Pa
  ): { pressure: number; gauge: number } {
    const g = 9.80665;
    const gauge = fluidDensity * g * depth;
    const pressure = atmosphericPressure + gauge;

    return { pressure, gauge };
  }

  /**
   * Capillary rise in a tube
   * h = 2σcosθ / (ρgr)
   */
  static capillaryRise(
    tubeRadius: number,           // m
    surfaceTension: number,       // N/m
    contactAngle: number,         // radians (0 for perfect wetting)
    fluidDensity: number          // kg/m³
  ): number {
    const g = 9.80665;
    return 2 * surfaceTension * Math.cos(contactAngle) / (fluidDensity * g * tubeRadius);
  }

  /**
   * Froude number (ratio of inertial to gravitational forces)
   * Fr = v / sqrt(gL)
   */
  static froudeNumber(velocity: number, characteristicLength: number): number {
    const g = 9.80665;
    return velocity / Math.sqrt(g * characteristicLength);
  }

  /**
   * Weber number (ratio of inertial to surface tension forces)
   * We = ρv²L / σ
   */
  static weberNumber(
    density: number,
    velocity: number,
    characteristicLength: number,
    surfaceTension: number
  ): number {
    return density * velocity * velocity * characteristicLength / surfaceTension;
  }

  /**
   * Cavitation number
   * σ = (P - P_v) / (0.5ρv²)
   */
  static cavitationNumber(
    pressure: number,             // Pa (local pressure)
    vaporPressure: number,        // Pa
    density: number,              // kg/m³
    velocity: number              // m/s
  ): { sigma: number; willCavitate: boolean } {
    const sigma = (pressure - vaporPressure) / (0.5 * density * velocity * velocity);
    const willCavitate = sigma < 0.3;  // Typical threshold

    return { sigma, willCavitate };
  }
}
