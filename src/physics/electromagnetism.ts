/**
 * Electromagnetism Module
 * Implements electric/magnetic fields, circuits, and EM waves
 */

import { PhysicalConstants as PC } from './constants';
import { Vector3D } from './particles';

export interface ElectricFieldResult {
  field: Vector3D;          // V/m
  magnitude: number;        // V/m
  potential: number;        // V
  forceOnTestCharge: Vector3D;  // N (for 1 C test charge)
}

export interface MagneticFieldResult {
  field: Vector3D;          // T
  magnitude: number;        // T
  fluxDensity: number;      // Wb/m²
}

export interface CircuitResult {
  current: number;          // A
  voltage: number;          // V
  power: number;            // W
  impedance: number;        // Ω
  phaseAngle: number;       // radians
  powerFactor: number;
  reactivepower: number;    // VAR
  apparentPower: number;    // VA
}

export interface CapacitorResult {
  capacitance: number;      // F
  charge: number;           // C
  energy: number;           // J
  voltage: number;          // V
  electricField: number;    // V/m
}

export interface InductorResult {
  inductance: number;       // H
  current: number;          // A
  energy: number;           // J
  magneticFlux: number;     // Wb
  inducedVoltage: number;   // V
}

export interface EMWaveResult {
  frequency: number;        // Hz
  wavelength: number;       // m
  energy: number;           // J (per photon)
  momentum: number;         // kg·m/s
  electricAmplitude: number; // V/m
  magneticAmplitude: number; // T
  intensity: number;        // W/m²
  radiationPressure: number; // Pa
}

export interface AntennaResult {
  gain: number;             // dBi
  directivity: number;
  effectiveArea: number;    // m²
  radiationResistance: number; // Ω
  beamwidth: number;        // degrees
}

/**
 * Electromagnetism Calculator
 */
export class Electromagnetism {

  // Vector utilities
  private static vectorAdd(a: Vector3D, b: Vector3D): Vector3D {
    return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
  }

  private static vectorScale(v: Vector3D, s: number): Vector3D {
    return { x: v.x * s, y: v.y * s, z: v.z * s };
  }

  private static vectorMagnitude(v: Vector3D): number {
    return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  }

  private static vectorCross(a: Vector3D, b: Vector3D): Vector3D {
    return {
      x: a.y * b.z - a.z * b.y,
      y: a.z * b.x - a.x * b.z,
      z: a.x * b.y - a.y * b.x,
    };
  }

  /**
   * Electric field from a point charge
   * E = kq/r² * r̂
   */
  static pointChargeField(
    charge: number,           // C
    fieldPoint: Vector3D,     // m
    chargePosition: Vector3D = { x: 0, y: 0, z: 0 }
  ): ElectricFieldResult {
    const r = {
      x: fieldPoint.x - chargePosition.x,
      y: fieldPoint.y - chargePosition.y,
      z: fieldPoint.z - chargePosition.z,
    };

    const distance = this.vectorMagnitude(r);

    if (distance < 1e-15) {
      throw new Error('Field point cannot coincide with charge position');
    }

    const magnitude = PC.ke * Math.abs(charge) / (distance * distance);
    const direction = charge > 0 ? 1 : -1;

    const field = this.vectorScale(r, direction * PC.ke * charge / Math.pow(distance, 3));
    const potential = PC.ke * charge / distance;
    const forceOnTestCharge = field;  // Force on 1 C test charge = E field

    return {
      field,
      magnitude,
      potential,
      forceOnTestCharge,
    };
  }

  /**
   * Electric field from multiple point charges (superposition)
   */
  static multipleChargeField(
    charges: Array<{ charge: number; position: Vector3D }>,
    fieldPoint: Vector3D
  ): ElectricFieldResult {
    let totalField: Vector3D = { x: 0, y: 0, z: 0 };
    let totalPotential = 0;

    for (const { charge, position } of charges) {
      const result = this.pointChargeField(charge, fieldPoint, position);
      totalField = this.vectorAdd(totalField, result.field);
      totalPotential += result.potential;
    }

    return {
      field: totalField,
      magnitude: this.vectorMagnitude(totalField),
      potential: totalPotential,
      forceOnTestCharge: totalField,
    };
  }

  /**
   * Electric field of infinite line charge
   * E = λ / (2πε₀r)
   */
  static lineChargeField(
    linearChargeDensity: number,  // C/m
    perpendicularDistance: number  // m
  ): number {
    return linearChargeDensity / (2 * Math.PI * PC.epsilon0 * perpendicularDistance);
  }

  /**
   * Electric field of infinite plane of charge
   * E = σ / (2ε₀)
   */
  static planeChargeField(surfaceChargeDensity: number): number {
    return surfaceChargeDensity / (2 * PC.epsilon0);
  }

  /**
   * Parallel plate capacitor
   */
  static parallelPlateCapacitor(
    area: number,             // m²
    separation: number,       // m
    voltage: number,          // V
    dielectricConstant: number = 1
  ): CapacitorResult {
    const capacitance = dielectricConstant * PC.epsilon0 * area / separation;
    const charge = capacitance * voltage;
    const energy = 0.5 * capacitance * voltage * voltage;
    const electricField = voltage / separation;

    return {
      capacitance,
      charge,
      energy,
      voltage,
      electricField,
    };
  }

  /**
   * Cylindrical capacitor
   */
  static cylindricalCapacitor(
    innerRadius: number,      // m
    outerRadius: number,      // m
    length: number,           // m
    voltage: number,          // V
    dielectricConstant: number = 1
  ): CapacitorResult {
    const capacitance = 2 * Math.PI * dielectricConstant * PC.epsilon0 * length /
                        Math.log(outerRadius / innerRadius);
    const charge = capacitance * voltage;
    const energy = 0.5 * capacitance * voltage * voltage;
    const electricField = voltage / (innerRadius * Math.log(outerRadius / innerRadius));

    return {
      capacitance,
      charge,
      energy,
      voltage,
      electricField,
    };
  }

  /**
   * Magnetic field from a long straight wire (Biot-Savart)
   * B = μ₀I / (2πr)
   */
  static straightWireField(
    current: number,          // A
    perpendicularDistance: number  // m
  ): number {
    return PC.mu0 * current / (2 * Math.PI * perpendicularDistance);
  }

  /**
   * Magnetic field at center of circular loop
   * B = μ₀I / (2r)
   */
  static circularLoopField(
    current: number,          // A
    radius: number            // m
  ): number {
    return PC.mu0 * current / (2 * radius);
  }

  /**
   * Magnetic field inside a solenoid
   * B = μ₀nI
   */
  static solenoidField(
    current: number,          // A
    turnsPerLength: number,   // turns/m
    relativePermeability: number = 1
  ): number {
    return relativePermeability * PC.mu0 * turnsPerLength * current;
  }

  /**
   * Inductance of a solenoid
   * L = μ₀n²V = μ₀n²Al
   */
  static solenoidInductance(
    turns: number,
    crossSectionArea: number, // m²
    length: number,           // m
    relativePermeability: number = 1
  ): InductorResult {
    const turnsPerLength = turns / length;
    const inductance = relativePermeability * PC.mu0 * turnsPerLength * turnsPerLength *
                       crossSectionArea * length;

    return {
      inductance,
      current: 0,
      energy: 0,
      magneticFlux: 0,
      inducedVoltage: 0,
    };
  }

  /**
   * Inductor energy and flux
   * E = 0.5LI², Φ = LI
   */
  static inductorEnergy(
    inductance: number,       // H
    current: number,          // A
    dIdt: number = 0          // A/s (rate of change of current)
  ): InductorResult {
    const energy = 0.5 * inductance * current * current;
    const magneticFlux = inductance * current;
    const inducedVoltage = -inductance * dIdt;  // Lenz's law (negative sign)

    return {
      inductance,
      current,
      energy,
      magneticFlux,
      inducedVoltage,
    };
  }

  /**
   * Lorentz force on a moving charge
   * F = q(E + v × B)
   */
  static lorentzForce(
    charge: number,
    velocity: Vector3D,
    electricField: Vector3D,
    magneticField: Vector3D
  ): { force: Vector3D; magnitude: number } {
    const vCrossB = this.vectorCross(velocity, magneticField);
    const totalField = this.vectorAdd(electricField, vCrossB);
    const force = this.vectorScale(totalField, charge);

    return {
      force,
      magnitude: this.vectorMagnitude(force),
    };
  }

  /**
   * Cyclotron motion of charged particle in magnetic field
   * r = mv/(qB), T = 2πm/(qB)
   */
  static cyclotronMotion(
    mass: number,             // kg
    charge: number,           // C
    velocity: number,         // m/s (perpendicular to B)
    magneticField: number     // T
  ): { radius: number; period: number; frequency: number; angularFrequency: number } {
    const radius = mass * velocity / (Math.abs(charge) * magneticField);
    const period = 2 * Math.PI * mass / (Math.abs(charge) * magneticField);
    const frequency = 1 / period;
    const angularFrequency = 2 * Math.PI * frequency;

    return { radius, period, frequency, angularFrequency };
  }

  /**
   * AC circuit analysis (RLC series)
   */
  static rlcCircuit(
    resistance: number,       // Ω
    inductance: number,       // H
    capacitance: number,      // F
    voltage: number,          // V (RMS or amplitude)
    frequency: number         // Hz
  ): CircuitResult {
    const omega = 2 * Math.PI * frequency;

    // Reactances
    const XL = omega * inductance;            // Inductive reactance
    const XC = 1 / (omega * capacitance);     // Capacitive reactance
    const X = XL - XC;                        // Net reactance

    // Impedance
    const impedance = Math.sqrt(resistance * resistance + X * X);

    // Phase angle (positive = inductive, negative = capacitive)
    const phaseAngle = Math.atan2(X, resistance);

    // Current
    const current = voltage / impedance;

    // Power calculations
    const powerFactor = Math.cos(phaseAngle);
    const power = voltage * current * powerFactor;          // Real power (W)
    const reactivepower = voltage * current * Math.sin(phaseAngle);  // Reactive power (VAR)
    const apparentPower = voltage * current;                 // Apparent power (VA)

    return {
      current,
      voltage,
      power,
      impedance,
      phaseAngle,
      powerFactor,
      reactivepower,
      apparentPower,
    };
  }

  /**
   * Resonance frequency of LC circuit
   * f₀ = 1/(2π√LC)
   */
  static resonanceFrequency(inductance: number, capacitance: number): {
    frequency: number;
    angularFrequency: number;
    wavelength: number;
  } {
    const angularFrequency = 1 / Math.sqrt(inductance * capacitance);
    const frequency = angularFrequency / (2 * Math.PI);
    const wavelength = PC.c / frequency;

    return { frequency, angularFrequency, wavelength };
  }

  /**
   * RC time constant and transient response
   */
  static rcTransient(
    resistance: number,       // Ω
    capacitance: number,      // F
    voltage: number,          // V (source voltage)
    time: number              // s
  ): { timeConstant: number; voltage: number; current: number; charge: number } {
    const timeConstant = resistance * capacitance;
    const chargeVoltage = voltage * (1 - Math.exp(-time / timeConstant));
    const current = (voltage / resistance) * Math.exp(-time / timeConstant);
    const charge = capacitance * chargeVoltage;

    return {
      timeConstant,
      voltage: chargeVoltage,
      current,
      charge,
    };
  }

  /**
   * RL time constant and transient response
   */
  static rlTransient(
    resistance: number,       // Ω
    inductance: number,       // H
    voltage: number,          // V
    time: number              // s
  ): { timeConstant: number; current: number; inducedVoltage: number } {
    const timeConstant = inductance / resistance;
    const steadyCurrent = voltage / resistance;
    const current = steadyCurrent * (1 - Math.exp(-time / timeConstant));
    const inducedVoltage = voltage * Math.exp(-time / timeConstant);

    return { timeConstant, current, inducedVoltage };
  }

  /**
   * Electromagnetic wave properties
   */
  static emWave(
    frequency: number,        // Hz
    electricAmplitude: number // V/m
  ): EMWaveResult {
    const wavelength = PC.c / frequency;
    const energy = PC.h * frequency;  // Photon energy
    const momentum = energy / PC.c;   // Photon momentum

    // B₀ = E₀/c
    const magneticAmplitude = electricAmplitude / PC.c;

    // Intensity: I = 0.5 * ε₀ * c * E₀²
    const intensity = 0.5 * PC.epsilon0 * PC.c * electricAmplitude * electricAmplitude;

    // Radiation pressure (for perfect absorption): P = I/c
    const radiationPressure = intensity / PC.c;

    return {
      frequency,
      wavelength,
      energy,
      momentum,
      electricAmplitude,
      magneticAmplitude,
      intensity,
      radiationPressure,
    };
  }

  /**
   * Skin depth in conductor
   * δ = sqrt(2ρ/(ωμ))
   */
  static skinDepth(
    frequency: number,        // Hz
    resistivity: number,      // Ω·m
    relativePermeability: number = 1
  ): number {
    const omega = 2 * Math.PI * frequency;
    return Math.sqrt(2 * resistivity / (omega * relativePermeability * PC.mu0));
  }

  /**
   * Dipole antenna radiation pattern
   */
  static dipoleAntenna(
    frequency: number,        // Hz
    length: number            // m (total dipole length)
  ): AntennaResult {
    const wavelength = PC.c / frequency;
    const k = 2 * Math.PI / wavelength;
    const L = length;

    // Half-wave dipole approximation
    const isHalfWave = Math.abs(length - wavelength / 2) < wavelength * 0.1;

    // Directivity for half-wave dipole ≈ 1.64 (2.15 dBi)
    const directivity = isHalfWave ? 1.64 : 1.5;
    const gain = 10 * Math.log10(directivity);

    // Effective area: A_e = λ²D/(4π)
    const effectiveArea = wavelength * wavelength * directivity / (4 * Math.PI);

    // Radiation resistance for half-wave dipole ≈ 73 Ω
    const radiationResistance = isHalfWave ? 73 : 20 * Math.pow(Math.PI * length / wavelength, 2);

    // Half-power beamwidth for half-wave dipole ≈ 78°
    const beamwidth = isHalfWave ? 78 : 90;

    return {
      gain,
      directivity,
      effectiveArea,
      radiationResistance,
      beamwidth,
    };
  }

  /**
   * Transformer calculations
   */
  static transformer(
    primaryTurns: number,
    secondaryTurns: number,
    primaryVoltage: number,   // V
    efficiency: number = 1    // 0 to 1
  ): {
    turnsRatio: number;
    secondaryVoltage: number;
    currentRatio: number;
    impedanceRatio: number;
  } {
    const turnsRatio = secondaryTurns / primaryTurns;
    const secondaryVoltage = primaryVoltage * turnsRatio * efficiency;
    const currentRatio = 1 / turnsRatio;  // Current transforms inversely
    const impedanceRatio = turnsRatio * turnsRatio;

    return {
      turnsRatio,
      secondaryVoltage,
      currentRatio,
      impedanceRatio,
    };
  }

  /**
   * Faraday's law of induction
   * EMF = -N * dΦ/dt
   */
  static faradayInduction(
    turns: number,
    fluxChange: number,       // Wb
    timeInterval: number      // s
  ): { emf: number; averageVoltage: number } {
    const emf = -turns * fluxChange / timeInterval;
    return { emf: Math.abs(emf), averageVoltage: Math.abs(emf) };
  }

  /**
   * Motional EMF
   * EMF = BLv
   */
  static motionalEMF(
    magneticField: number,    // T
    length: number,           // m (conductor length perpendicular to motion)
    velocity: number          // m/s
  ): number {
    return magneticField * length * velocity;
  }
}
