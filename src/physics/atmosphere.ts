/**
 * Atmospheric Physics Module
 * Implements International Standard Atmosphere (ISA), aerodynamics, and reentry physics
 */

import { PhysicalConstants as PC } from './constants';

// Atmospheric layer boundaries (meters)
const TROPOSPHERE_TOP = 11000;
const STRATOSPHERE_TOP = 25000;
const MESOSPHERE_TOP = 47000;
const THERMOSPHERE_TOP = 86000;

// ISA sea-level reference values
const ISA_T0 = 288.15;      // Temperature (K)
const ISA_P0 = 101325;      // Pressure (Pa)
const ISA_RHO0 = 1.225;     // Density (kg/m³)
const ISA_g0 = 9.80665;     // Gravity (m/s²)
const ISA_M = 0.0289644;    // Molar mass of dry air (kg/mol)
const ISA_R = 8.31447;      // Universal gas constant (J/(mol·K))
const ISA_GAMMA = 1.4;      // Heat capacity ratio for air

export interface AtmosphericState {
  altitude: number;         // m
  temperature: number;      // K
  pressure: number;         // Pa
  density: number;          // kg/m³
  speedOfSound: number;     // m/s
  dynamicViscosity: number; // Pa·s
  layer: string;
}

export interface AerodynamicResult {
  drag: number;             // N
  lift: number;             // N
  dynamicPressure: number;  // Pa
  machNumber: number;
  reynoldsNumber: number;
  dragCoefficient: number;
  liftCoefficient: number;
}

export interface ReentryResult {
  heatFlux: number;         // W/m²
  stagnationTemperature: number;  // K
  deceleration: number;     // m/s² (g-force = deceleration / 9.8)
  altitudeRate: number;     // m/s
  velocityRate: number;     // m/s²
}

export interface BallisticTrajectoryPoint {
  time: number;
  altitude: number;
  downrange: number;
  velocity: number;
  flightPathAngle: number;
  machNumber: number;
  dynamicPressure: number;
}

/**
 * Atmospheric Physics Calculator
 * Based on International Standard Atmosphere (ISA) 1976
 */
export class AtmosphericPhysics {

  /**
   * Calculate atmospheric properties at a given altitude
   * Implements the US Standard Atmosphere 1976 model
   */
  static getAtmosphericState(altitude: number): AtmosphericState {
    let temperature: number;
    let pressure: number;
    let layer: string;

    if (altitude < 0) {
      // Below sea level - extrapolate
      const L = -0.0065; // Lapse rate
      temperature = ISA_T0 - L * altitude;
      pressure = ISA_P0 * Math.pow(temperature / ISA_T0, -ISA_g0 * ISA_M / (ISA_R * L));
      layer = 'Below Sea Level';
    } else if (altitude < TROPOSPHERE_TOP) {
      // Troposphere: temperature decreases linearly
      const L = 0.0065; // Lapse rate (K/m)
      temperature = ISA_T0 - L * altitude;
      pressure = ISA_P0 * Math.pow(temperature / ISA_T0, ISA_g0 * ISA_M / (ISA_R * L));
      layer = 'Troposphere';
    } else if (altitude < STRATOSPHERE_TOP) {
      // Lower Stratosphere: isothermal
      const T11 = ISA_T0 - 0.0065 * TROPOSPHERE_TOP; // 216.65 K
      const P11 = ISA_P0 * Math.pow(T11 / ISA_T0, ISA_g0 * ISA_M / (ISA_R * 0.0065));
      temperature = T11;
      pressure = P11 * Math.exp(-ISA_g0 * ISA_M * (altitude - TROPOSPHERE_TOP) / (ISA_R * T11));
      layer = 'Stratosphere (Lower)';
    } else if (altitude < MESOSPHERE_TOP) {
      // Upper Stratosphere: temperature increases
      const T11 = 216.65;
      const P11 = ISA_P0 * Math.pow(T11 / ISA_T0, ISA_g0 * ISA_M / (ISA_R * 0.0065));
      const P25 = P11 * Math.exp(-ISA_g0 * ISA_M * (STRATOSPHERE_TOP - TROPOSPHERE_TOP) / (ISA_R * T11));

      const L = -0.0028; // Negative lapse rate (temperature increases)
      temperature = T11 - L * (altitude - STRATOSPHERE_TOP);
      pressure = P25 * Math.pow(temperature / T11, ISA_g0 * ISA_M / (ISA_R * L));
      layer = 'Stratosphere (Upper)';
    } else if (altitude < THERMOSPHERE_TOP) {
      // Mesosphere: temperature decreases again
      const T47 = 216.65 + 0.0028 * (MESOSPHERE_TOP - STRATOSPHERE_TOP);
      temperature = T47 - 0.0028 * (altitude - MESOSPHERE_TOP);
      // Simplified pressure calculation for mesosphere
      pressure = 0.1 * Math.exp(-(altitude - MESOSPHERE_TOP) / 10000);
      layer = 'Mesosphere';
    } else {
      // Thermosphere: temperature increases significantly
      temperature = 186.87 + 2.0 * Math.sqrt(altitude - THERMOSPHERE_TOP);
      // Very low pressure
      pressure = 0.0001 * Math.exp(-(altitude - THERMOSPHERE_TOP) / 50000);
      layer = 'Thermosphere';
    }

    // Calculate density from ideal gas law: ρ = PM/(RT)
    const density = pressure * ISA_M / (ISA_R * temperature);

    // Speed of sound: a = sqrt(γRT/M)
    const speedOfSound = Math.sqrt(ISA_GAMMA * ISA_R * temperature / ISA_M);

    // Dynamic viscosity using Sutherland's formula
    const dynamicViscosity = this.sutherlandViscosity(temperature);

    return {
      altitude,
      temperature,
      pressure,
      density,
      speedOfSound,
      dynamicViscosity,
      layer,
    };
  }

  /**
   * Sutherland's formula for dynamic viscosity of air
   */
  private static sutherlandViscosity(temperature: number): number {
    const T0 = 291.15;       // Reference temperature (K)
    const mu0 = 1.827e-5;    // Reference viscosity (Pa·s)
    const S = 120;           // Sutherland constant (K)

    return mu0 * Math.pow(temperature / T0, 1.5) * (T0 + S) / (temperature + S);
  }

  /**
   * Calculate aerodynamic forces on a body
   */
  static calculateAerodynamics(
    velocity: number,           // m/s
    altitude: number,           // m
    referenceArea: number,      // m²
    dragCoefficient: number,    // Cd
    liftCoefficient: number = 0, // Cl
    characteristicLength: number = 1 // m (for Reynolds number)
  ): AerodynamicResult {
    const atm = this.getAtmosphericState(altitude);

    // Dynamic pressure: q = 0.5 * ρ * v²
    const dynamicPressure = 0.5 * atm.density * velocity * velocity;

    // Mach number: M = v / a
    const machNumber = velocity / atm.speedOfSound;

    // Reynolds number: Re = ρvL/μ
    const reynoldsNumber = atm.density * velocity * characteristicLength / atm.dynamicViscosity;

    // Drag force: D = q * S * Cd
    const drag = dynamicPressure * referenceArea * dragCoefficient;

    // Lift force: L = q * S * Cl
    const lift = dynamicPressure * referenceArea * liftCoefficient;

    return {
      drag,
      lift,
      dynamicPressure,
      machNumber,
      reynoldsNumber,
      dragCoefficient,
      liftCoefficient,
    };
  }

  /**
   * Calculate drag coefficient variation with Mach number
   * Simplified model for subsonic/transonic/supersonic regimes
   */
  static machDragCoefficient(baseCd: number, machNumber: number): number {
    if (machNumber < 0.8) {
      // Subsonic: slight increase due to compressibility (Prandtl-Glauert)
      return baseCd / Math.sqrt(1 - machNumber * machNumber);
    } else if (machNumber < 1.2) {
      // Transonic: peak drag (wave drag onset)
      const transitionFactor = 1 + 2 * Math.pow((machNumber - 0.8) / 0.4, 2);
      return baseCd * transitionFactor;
    } else {
      // Supersonic: decreases with Mach
      return baseCd * (1.2 + 0.3 / machNumber);
    }
  }

  /**
   * Atmospheric reentry heating calculation
   * Uses Sutton-Graves correlation for stagnation point heating
   */
  static calculateReentryHeating(
    velocity: number,       // m/s
    altitude: number,       // m
    noseRadius: number,     // m
    mass: number,           // kg
    ballisticCoefficient: number = 100  // kg/m² (m / (Cd * A))
  ): ReentryResult {
    const atm = this.getAtmosphericState(altitude);

    // Sutton-Graves stagnation point heat flux (W/m²)
    // q = k * sqrt(ρ/r_n) * v³
    const k = 1.83e-4;  // Constant for Earth atmosphere
    const heatFlux = k * Math.sqrt(atm.density / noseRadius) * Math.pow(velocity, 3);

    // Stagnation temperature (assuming complete stagnation)
    // T_stag = T + v² / (2 * Cp)
    const Cp = 1005;  // Specific heat of air (J/(kg·K))
    const stagnationTemperature = atm.temperature + (velocity * velocity) / (2 * Cp);

    // Deceleration from drag
    // a = D/m = 0.5 * ρ * v² * Cd * A / m = 0.5 * ρ * v² / β
    const deceleration = 0.5 * atm.density * velocity * velocity / ballisticCoefficient;

    // Altitude rate (assuming vertical descent for simplification)
    const altitudeRate = -velocity * 0.1;  // Simplified

    // Velocity rate (deceleration)
    const velocityRate = -deceleration;

    return {
      heatFlux,
      stagnationTemperature,
      deceleration,
      altitudeRate,
      velocityRate,
    };
  }

  /**
   * Simulate ballistic trajectory through atmosphere
   * Uses simple Euler integration
   */
  static simulateBallisticTrajectory(
    initialAltitude: number,    // m
    initialVelocity: number,    // m/s
    initialAngle: number,       // radians (negative = descending)
    mass: number,               // kg
    referenceArea: number,      // m²
    dragCoefficient: number,    // Cd
    timeStep: number = 0.1,     // s
    maxTime: number = 1000      // s
  ): BallisticTrajectoryPoint[] {
    const trajectory: BallisticTrajectoryPoint[] = [];

    let altitude = initialAltitude;
    let velocity = initialVelocity;
    let angle = initialAngle;
    let downrange = 0;
    let time = 0;

    while (altitude > 0 && time < maxTime) {
      const atm = this.getAtmosphericState(altitude);
      const dynamicPressure = 0.5 * atm.density * velocity * velocity;
      const machNumber = velocity / atm.speedOfSound;

      // Adjust Cd for Mach number
      const Cd = this.machDragCoefficient(dragCoefficient, machNumber);

      // Drag force
      const drag = dynamicPressure * referenceArea * Cd;

      // Gravity (varies with altitude)
      const g = ISA_g0 * Math.pow(6371000 / (6371000 + altitude), 2);

      // Accelerations
      const dragAccel = drag / mass;
      const tangentialAccel = -dragAccel - g * Math.sin(angle);
      const normalAccel = -g * Math.cos(angle);

      // Record state
      trajectory.push({
        time,
        altitude,
        downrange,
        velocity,
        flightPathAngle: angle * 180 / Math.PI,
        machNumber,
        dynamicPressure,
      });

      // Update state (Euler integration)
      const vx = velocity * Math.cos(angle);
      const vy = velocity * Math.sin(angle);

      velocity += tangentialAccel * timeStep;
      if (velocity > 0) {
        angle += (normalAccel / velocity) * timeStep;
      }

      downrange += vx * timeStep;
      altitude += vy * timeStep;
      time += timeStep;
    }

    return trajectory;
  }

  /**
   * Calculate terminal velocity for a falling object
   */
  static terminalVelocity(
    mass: number,
    dragCoefficient: number,
    referenceArea: number,
    altitude: number = 0
  ): number {
    const atm = this.getAtmosphericState(altitude);
    const g = ISA_g0 * Math.pow(6371000 / (6371000 + altitude), 2);

    // v_terminal = sqrt(2mg / (ρ * Cd * A))
    return Math.sqrt(2 * mass * g / (atm.density * dragCoefficient * referenceArea));
  }

  /**
   * Calculate pressure altitude (altitude in ISA where pressure matches)
   */
  static pressureAltitude(pressure: number): number {
    // Inverse of tropospheric pressure formula
    if (pressure > 22632) {  // In troposphere
      const exponent = ISA_R * 0.0065 / (ISA_g0 * ISA_M);
      const T_ratio = Math.pow(pressure / ISA_P0, exponent);
      return (ISA_T0 - ISA_T0 * T_ratio) / 0.0065;
    } else {
      // In stratosphere (simplified)
      const P11 = 22632;
      const T11 = 216.65;
      return TROPOSPHERE_TOP + (ISA_R * T11 / (ISA_g0 * ISA_M)) * Math.log(P11 / pressure);
    }
  }

  /**
   * Calculate density altitude
   * Altitude in ISA where density matches current conditions
   */
  static densityAltitude(temperature: number, pressure: number): number {
    const density = pressure * ISA_M / (ISA_R * temperature);

    // Iterative search for matching altitude
    let low = -1000;
    let high = 100000;

    while (high - low > 1) {
      const mid = (low + high) / 2;
      const testDensity = this.getAtmosphericState(mid).density;

      if (testDensity > density) {
        low = mid;
      } else {
        high = mid;
      }
    }

    return (low + high) / 2;
  }

  /**
   * Wind gradient calculation (simplified logarithmic profile)
   */
  static windGradient(
    surfaceWindSpeed: number,  // m/s at reference height
    altitude: number,           // m
    surfaceRoughness: number = 0.03  // m (0.03 for open terrain)
  ): number {
    const referenceHeight = 10;  // Standard meteorological reference height

    if (altitude < surfaceRoughness) {
      return 0;
    }

    // Logarithmic wind profile
    return surfaceWindSpeed * Math.log(altitude / surfaceRoughness) /
           Math.log(referenceHeight / surfaceRoughness);
  }

  /**
   * Calculate air density for non-standard conditions
   */
  static airDensity(temperature: number, pressure: number, relativeHumidity: number = 0): number {
    // Saturation vapor pressure (Tetens formula)
    const tempC = temperature - 273.15;
    const es = 610.78 * Math.exp(17.27 * tempC / (tempC + 237.3));

    // Actual vapor pressure
    const e = relativeHumidity * es;

    // Virtual temperature (accounts for moisture)
    const Tv = temperature / (1 - (e / pressure) * (1 - 0.622));

    // Density from ideal gas law with virtual temperature
    return pressure * ISA_M / (ISA_R * Tv);
  }
}
