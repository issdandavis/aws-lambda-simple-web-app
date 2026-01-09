/**
 * Thermodynamics Module
 * Implements heat transfer, thermodynamic cycles, and statistical mechanics
 */

import { PhysicalConstants as PC } from './constants';

// Common material thermal properties
export const ThermalProperties = {
  copper: {
    thermalConductivity: 401,     // W/(m·K)
    specificHeat: 385,            // J/(kg·K)
    density: 8960,                // kg/m³
    thermalDiffusivity: 1.11e-4,  // m²/s
    emissivity: 0.03,
  },
  aluminum: {
    thermalConductivity: 237,
    specificHeat: 897,
    density: 2700,
    thermalDiffusivity: 9.7e-5,
    emissivity: 0.04,
  },
  steel: {
    thermalConductivity: 50,
    specificHeat: 500,
    density: 7850,
    thermalDiffusivity: 1.27e-5,
    emissivity: 0.3,
  },
  water: {
    thermalConductivity: 0.606,
    specificHeat: 4186,
    density: 998,
    thermalDiffusivity: 1.43e-7,
    emissivity: 0.96,
  },
  air: {
    thermalConductivity: 0.026,
    specificHeat: 1005,
    density: 1.2,
    thermalDiffusivity: 2.2e-5,
    emissivity: 0,
  },
  glass: {
    thermalConductivity: 1.0,
    specificHeat: 840,
    density: 2500,
    thermalDiffusivity: 4.76e-7,
    emissivity: 0.92,
  },
} as const;

export interface HeatTransferResult {
  heatRate: number;           // W
  heatFlux: number;           // W/m²
  thermalResistance: number;  // K/W
  temperatureGradient: number; // K/m
}

export interface ConvectionResult {
  heatTransferCoefficient: number;  // W/(m²·K)
  nusseltNumber: number;
  heatRate: number;           // W
  thermalBoundaryLayer: number; // m
}

export interface RadiationResult {
  emittedPower: number;       // W
  netHeatTransfer: number;    // W
  viewFactor: number;
  radiosity: number;          // W/m²
}

export interface ThermodynamicState {
  temperature: number;        // K
  pressure: number;           // Pa
  volume: number;             // m³
  internalEnergy: number;     // J
  enthalpy: number;           // J
  entropy: number;            // J/K
}

export interface CycleResult {
  efficiency: number;
  work: number;               // J
  heatIn: number;             // J
  heatOut: number;            // J
  cop?: number;               // Coefficient of performance (for refrigeration)
}

export interface HeatExchangerResult {
  heatTransferRate: number;   // W
  lmtd: number;               // Log mean temperature difference
  effectiveness: number;
  ntu: number;                // Number of transfer units
  outletTemperatureHot: number;
  outletTemperatureCold: number;
}

/**
 * Thermodynamics Calculator
 */
export class Thermodynamics {

  /**
   * Conduction heat transfer (Fourier's law)
   * Q = -kA(dT/dx)
   */
  static conduction(
    thermalConductivity: number,  // W/(m·K)
    area: number,                 // m²
    thickness: number,            // m
    temperatureDifference: number // K
  ): HeatTransferResult {
    const thermalResistance = thickness / (thermalConductivity * area);
    const heatRate = temperatureDifference / thermalResistance;
    const heatFlux = heatRate / area;
    const temperatureGradient = temperatureDifference / thickness;

    return {
      heatRate,
      heatFlux,
      thermalResistance,
      temperatureGradient,
    };
  }

  /**
   * Multi-layer conduction (composite wall)
   */
  static compositeWallConduction(
    layers: Array<{ conductivity: number; thickness: number }>,
    area: number,
    temperatureDifference: number
  ): { heatRate: number; totalResistance: number; interfaceTemperatures: number[] } {
    // Total thermal resistance
    let totalResistance = 0;
    for (const layer of layers) {
      totalResistance += layer.thickness / (layer.conductivity * area);
    }

    const heatRate = temperatureDifference / totalResistance;

    // Calculate interface temperatures
    const interfaceTemperatures: number[] = [];
    let T = temperatureDifference;  // Start from hot side (assuming ΔT = T_hot - T_cold)

    for (const layer of layers) {
      const layerResistance = layer.thickness / (layer.conductivity * area);
      const tempDrop = heatRate * layerResistance;
      T -= tempDrop;
      interfaceTemperatures.push(T);
    }

    return { heatRate, totalResistance, interfaceTemperatures };
  }

  /**
   * Convection heat transfer (Newton's law of cooling)
   * Q = hA(T_s - T_∞)
   */
  static convection(
    heatTransferCoefficient: number,  // W/(m²·K)
    area: number,                     // m²
    surfaceTemperature: number,       // K
    fluidTemperature: number          // K
  ): { heatRate: number; thermalResistance: number } {
    const heatRate = heatTransferCoefficient * area * (surfaceTemperature - fluidTemperature);
    const thermalResistance = 1 / (heatTransferCoefficient * area);

    return { heatRate, thermalResistance };
  }

  /**
   * Natural convection from vertical plate
   * Uses empirical correlations for Nusselt number
   */
  static naturalConvectionVerticalPlate(
    height: number,               // m
    surfaceTemperature: number,   // K
    fluidTemperature: number,     // K
    fluidProperties: {
      thermalConductivity: number;
      kinematicViscosity: number;
      thermalDiffusivity: number;
      beta: number;  // Thermal expansion coefficient
    }
  ): ConvectionResult {
    const g = 9.80665;
    const deltaT = Math.abs(surfaceTemperature - fluidTemperature);
    const Tf = (surfaceTemperature + fluidTemperature) / 2;

    // Grashof number: Gr = gβΔTL³/ν²
    const Gr = g * fluidProperties.beta * deltaT * Math.pow(height, 3) /
               Math.pow(fluidProperties.kinematicViscosity, 2);

    // Prandtl number: Pr = ν/α
    const Pr = fluidProperties.kinematicViscosity / fluidProperties.thermalDiffusivity;

    // Rayleigh number: Ra = Gr × Pr
    const Ra = Gr * Pr;

    // Nusselt number correlation (Churchill-Chu)
    let Nu: number;
    if (Ra < 1e9) {
      // Laminar
      Nu = 0.68 + 0.67 * Math.pow(Ra, 0.25) / Math.pow(1 + Math.pow(0.492 / Pr, 9 / 16), 4 / 9);
    } else {
      // Turbulent
      const temp = 1 + Math.pow(0.492 / Pr, 9 / 16);
      Nu = Math.pow(0.825 + 0.387 * Math.pow(Ra, 1 / 6) / Math.pow(temp, 8 / 27), 2);
    }

    const heatTransferCoefficient = Nu * fluidProperties.thermalConductivity / height;
    const heatRate = heatTransferCoefficient * deltaT;  // Per unit area
    const thermalBoundaryLayer = height / Math.pow(Gr, 0.25);

    return {
      heatTransferCoefficient,
      nusseltNumber: Nu,
      heatRate,
      thermalBoundaryLayer,
    };
  }

  /**
   * Forced convection in pipe (Dittus-Boelter correlation)
   * Nu = 0.023 × Re^0.8 × Pr^n
   */
  static forcedConvectionPipe(
    diameter: number,             // m
    velocity: number,             // m/s
    fluidProperties: {
      density: number;
      thermalConductivity: number;
      dynamicViscosity: number;
      specificHeat: number;
    },
    heating: boolean = true       // true for heating fluid, false for cooling
  ): ConvectionResult {
    // Reynolds number
    const Re = fluidProperties.density * velocity * diameter / fluidProperties.dynamicViscosity;

    // Prandtl number
    const Pr = fluidProperties.dynamicViscosity * fluidProperties.specificHeat /
               fluidProperties.thermalConductivity;

    // Dittus-Boelter exponent
    const n = heating ? 0.4 : 0.3;

    // Nusselt number (valid for Re > 10000, 0.7 < Pr < 160)
    let Nu: number;
    if (Re > 10000) {
      Nu = 0.023 * Math.pow(Re, 0.8) * Math.pow(Pr, n);
    } else if (Re > 2300) {
      // Transitional (use Gnielinski)
      const f = Math.pow(0.79 * Math.log(Re) - 1.64, -2);
      Nu = (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * Math.sqrt(f / 8) * (Math.pow(Pr, 2 / 3) - 1));
    } else {
      // Laminar (constant wall temperature)
      Nu = 3.66;
    }

    const heatTransferCoefficient = Nu * fluidProperties.thermalConductivity / diameter;
    const thermalBoundaryLayer = diameter / Nu;

    return {
      heatTransferCoefficient,
      nusseltNumber: Nu,
      heatRate: 0,  // Need surface/fluid temps to calculate
      thermalBoundaryLayer,
    };
  }

  /**
   * Radiation heat transfer (Stefan-Boltzmann law)
   * Q = εσA(T⁴ - T_surr⁴)
   */
  static radiation(
    emissivity: number,           // 0 to 1
    area: number,                 // m²
    surfaceTemperature: number,   // K
    surroundingTemperature: number // K
  ): RadiationResult {
    const emittedPower = emissivity * PC.sigma * area * Math.pow(surfaceTemperature, 4);
    const absorbedPower = emissivity * PC.sigma * area * Math.pow(surroundingTemperature, 4);
    const netHeatTransfer = emittedPower - absorbedPower;
    const radiosity = emissivity * PC.sigma * Math.pow(surfaceTemperature, 4);

    return {
      emittedPower,
      netHeatTransfer,
      viewFactor: 1,  // Assuming complete enclosure
      radiosity,
    };
  }

  /**
   * Radiation between two surfaces
   */
  static radiationExchange(
    emissivity1: number,
    emissivity2: number,
    area1: number,                // m²
    temperature1: number,         // K
    temperature2: number,         // K
    viewFactor: number = 1        // F_12
  ): number {
    // Radiosity method for gray surfaces
    const Eb1 = PC.sigma * Math.pow(temperature1, 4);
    const Eb2 = PC.sigma * Math.pow(temperature2, 4);

    // Net radiation exchange
    const numerator = Eb1 - Eb2;
    const denominator = (1 - emissivity1) / (emissivity1 * area1) +
                        1 / (area1 * viewFactor) +
                        (1 - emissivity2) / (emissivity2 * area1 * viewFactor);

    return numerator / denominator;
  }

  /**
   * Carnot cycle efficiency
   * η = 1 - T_cold/T_hot
   */
  static carnotEfficiency(hotTemperature: number, coldTemperature: number): number {
    return 1 - coldTemperature / hotTemperature;
  }

  /**
   * Ideal gas state change
   */
  static idealGasState(
    moles: number,
    temperature: number,          // K
    pressure?: number,            // Pa
    volume?: number               // m³
  ): ThermodynamicState {
    // PV = nRT
    let P = pressure;
    let V = volume;

    if (P !== undefined && V === undefined) {
      V = moles * PC.R * temperature / P;
    } else if (V !== undefined && P === undefined) {
      P = moles * PC.R * temperature / V;
    } else if (P === undefined && V === undefined) {
      throw new Error('Either pressure or volume must be specified');
    }

    // For ideal gas: U = (f/2)nRT where f = degrees of freedom
    // Assuming diatomic gas (f = 5)
    const f = 5;
    const internalEnergy = (f / 2) * moles * PC.R * temperature;

    // Enthalpy: H = U + PV = U + nRT
    const enthalpy = internalEnergy + moles * PC.R * temperature;

    // Entropy (relative to some reference state)
    // S = nCv*ln(T) + nR*ln(V) + S0
    // Simplified calculation
    const entropy = moles * PC.R * (Math.log(temperature) + Math.log(V!));

    return {
      temperature,
      pressure: P!,
      volume: V!,
      internalEnergy,
      enthalpy,
      entropy,
    };
  }

  /**
   * Isentropic process for ideal gas
   * T₂/T₁ = (P₂/P₁)^((γ-1)/γ) = (V₁/V₂)^(γ-1)
   */
  static isentropicProcess(
    initialTemp: number,
    initialPressure: number,
    finalPressure: number,
    gamma: number = 1.4           // Heat capacity ratio (1.4 for air)
  ): { finalTemperature: number; compressionRatio: number } {
    const pressureRatio = finalPressure / initialPressure;
    const exponent = (gamma - 1) / gamma;
    const finalTemperature = initialTemp * Math.pow(pressureRatio, exponent);
    const compressionRatio = Math.pow(pressureRatio, 1 / gamma);

    return { finalTemperature, compressionRatio };
  }

  /**
   * Otto cycle analysis (gasoline engine)
   */
  static ottoCycle(
    compressionRatio: number,     // V1/V2
    heatInput: number,            // J
    gamma: number = 1.4
  ): CycleResult {
    // Efficiency: η = 1 - 1/r^(γ-1)
    const efficiency = 1 - 1 / Math.pow(compressionRatio, gamma - 1);
    const work = efficiency * heatInput;
    const heatOut = heatInput - work;

    return {
      efficiency,
      work,
      heatIn: heatInput,
      heatOut,
    };
  }

  /**
   * Diesel cycle analysis
   */
  static dieselCycle(
    compressionRatio: number,     // V1/V2
    cutoffRatio: number,          // V3/V2 (volume at end of heat addition / volume at end of compression)
    heatInput: number,
    gamma: number = 1.4
  ): CycleResult {
    // Efficiency: η = 1 - (1/r^(γ-1)) × (ρ^γ - 1)/(γ(ρ - 1))
    const rhoTerm = (Math.pow(cutoffRatio, gamma) - 1) / (gamma * (cutoffRatio - 1));
    const efficiency = 1 - (1 / Math.pow(compressionRatio, gamma - 1)) * rhoTerm;
    const work = efficiency * heatInput;
    const heatOut = heatInput - work;

    return {
      efficiency,
      work,
      heatIn: heatInput,
      heatOut,
    };
  }

  /**
   * Rankine cycle analysis (steam power plant)
   */
  static rankineCycle(
    boilerPressure: number,       // Pa
    condenserPressure: number,    // Pa
    boilerTemperature: number,    // K
    pumpEfficiency: number = 0.85,
    turbineEfficiency: number = 0.9
  ): CycleResult {
    // Simplified analysis using approximations
    // In practice, would use steam tables

    const Tboiler = boilerTemperature;
    const Tcondenser = 273.15 + 40;  // Approximate condenser temp

    // Ideal Carnot-like efficiency
    const idealEfficiency = 1 - Tcondenser / Tboiler;

    // Account for irreversibilities
    const efficiency = idealEfficiency * 0.7;  // Typical Rankine is ~70% of Carnot

    // Placeholder values (would need steam tables for accurate calculation)
    const heatIn = 1000;  // Normalized
    const work = efficiency * heatIn;
    const heatOut = heatIn - work;

    return {
      efficiency,
      work,
      heatIn,
      heatOut,
    };
  }

  /**
   * Refrigeration/heat pump cycle
   */
  static refrigerationCycle(
    hotTemperature: number,       // K (condenser)
    coldTemperature: number,      // K (evaporator)
    refrigerationCapacity: number, // W
    compressorEfficiency: number = 0.8
  ): CycleResult {
    // Carnot COP for refrigerator: COP_ref = T_cold / (T_hot - T_cold)
    const carnotCOP = coldTemperature / (hotTemperature - coldTemperature);
    const actualCOP = carnotCOP * compressorEfficiency;

    const workInput = refrigerationCapacity / actualCOP;
    const heatRejected = refrigerationCapacity + workInput;

    return {
      efficiency: actualCOP,  // COP is analogous to efficiency for refrigeration
      work: workInput,
      heatIn: refrigerationCapacity,  // Heat absorbed from cold reservoir
      heatOut: heatRejected,          // Heat rejected to hot reservoir
      cop: actualCOP,
    };
  }

  /**
   * Counter-flow heat exchanger (LMTD method)
   */
  static heatExchangerLMTD(
    hotInlet: number,             // K
    hotOutlet: number,            // K
    coldInlet: number,            // K
    coldOutlet: number,           // K
    overallHeatTransferCoeff: number, // W/(m²·K)
    area: number                  // m²
  ): HeatExchangerResult {
    // Temperature differences at each end
    const deltaT1 = hotInlet - coldOutlet;   // Hot end
    const deltaT2 = hotOutlet - coldInlet;   // Cold end

    // Log mean temperature difference
    let lmtd: number;
    if (Math.abs(deltaT1 - deltaT2) < 0.001) {
      lmtd = deltaT1;  // Avoid division by zero
    } else {
      lmtd = (deltaT1 - deltaT2) / Math.log(deltaT1 / deltaT2);
    }

    // Heat transfer rate: Q = UA × LMTD
    const heatTransferRate = overallHeatTransferCoeff * area * lmtd;

    // NTU and effectiveness (assuming balanced flow)
    const ntu = overallHeatTransferCoeff * area / (heatTransferRate / lmtd * 0.5);
    const effectiveness = (hotInlet - hotOutlet) / (hotInlet - coldInlet);

    return {
      heatTransferRate,
      lmtd,
      effectiveness,
      ntu,
      outletTemperatureHot: hotOutlet,
      outletTemperatureCold: coldOutlet,
    };
  }

  /**
   * Transient heat conduction (lumped capacitance method)
   * Valid when Bi < 0.1
   */
  static lumpedCapacitance(
    mass: number,                 // kg
    specificHeat: number,         // J/(kg·K)
    heatTransferCoeff: number,    // W/(m²·K)
    surfaceArea: number,          // m²
    initialTemperature: number,   // K
    ambientTemperature: number,   // K
    time: number                  // s
  ): { temperature: number; timeConstant: number; biotNumber: number } {
    // Time constant: τ = ρVc/(hA) = mc/(hA)
    const timeConstant = mass * specificHeat / (heatTransferCoeff * surfaceArea);

    // Temperature at time t: T(t) = T_∞ + (T_i - T_∞)e^(-t/τ)
    const temperature = ambientTemperature +
                        (initialTemperature - ambientTemperature) * Math.exp(-time / timeConstant);

    // Biot number for validation
    const characteristicLength = mass / (surfaceArea * 1000);  // Approximate (assuming density ~1000)
    const thermalConductivity = 50;  // Assumed
    const biotNumber = heatTransferCoeff * characteristicLength / thermalConductivity;

    return { temperature, timeConstant, biotNumber };
  }

  /**
   * Fin heat transfer
   */
  static finHeatTransfer(
    finLength: number,            // m
    finThickness: number,         // m
    thermalConductivity: number,  // W/(m·K)
    heatTransferCoeff: number,    // W/(m²·K)
    baseTemperature: number,      // K
    ambientTemperature: number,   // K
    finWidth: number = 1          // m (for rectangular fin)
  ): { heatRate: number; efficiency: number; effectiveness: number } {
    const perimeter = 2 * (finWidth + finThickness);
    const crossSection = finWidth * finThickness;

    // m parameter: m = sqrt(hP/(kA))
    const m = Math.sqrt(heatTransferCoeff * perimeter / (thermalConductivity * crossSection));

    // Heat rate (assuming adiabatic tip)
    const thetaB = baseTemperature - ambientTemperature;
    const heatRate = Math.sqrt(heatTransferCoeff * perimeter * thermalConductivity * crossSection) *
                     thetaB * Math.tanh(m * finLength);

    // Fin efficiency
    const efficiency = Math.tanh(m * finLength) / (m * finLength);

    // Fin effectiveness (ratio of fin heat transfer to heat transfer without fin)
    const bareHeatRate = heatTransferCoeff * crossSection * thetaB;
    const effectiveness = heatRate / bareHeatRate;

    return { heatRate, efficiency, effectiveness };
  }

  /**
   * Thermal resistance network
   */
  static thermalResistanceNetwork(
    resistances: number[],        // K/W
    configuration: 'series' | 'parallel',
    temperatureDifference: number // K
  ): { totalResistance: number; heatRate: number } {
    let totalResistance: number;

    if (configuration === 'series') {
      totalResistance = resistances.reduce((sum, r) => sum + r, 0);
    } else {
      const inverseSum = resistances.reduce((sum, r) => sum + 1 / r, 0);
      totalResistance = 1 / inverseSum;
    }

    const heatRate = temperatureDifference / totalResistance;

    return { totalResistance, heatRate };
  }
}
