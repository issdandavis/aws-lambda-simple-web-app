/**
 * Advanced Orbital Mechanics Module
 *
 * Implements high-fidelity orbital mechanics:
 * - N-body gravitational simulation with symplectic integrators
 * - Orbital maneuver planning (Hohmann, bi-elliptic, optimal)
 * - Perturbation effects (J2, atmospheric drag, third-body, SRP)
 * - Lagrange point calculation
 * - Lambert's problem solver for trajectory design
 * - Patched conic approximation for interplanetary missions
 */

import { PhysicalConstants } from './constants';

export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

export interface OrbitalElements {
  semiMajorAxis: number;      // a (m)
  eccentricity: number;       // e
  inclination: number;        // i (rad)
  longitudeOfAscendingNode: number;  // Ω (rad)
  argumentOfPeriapsis: number;       // ω (rad)
  trueAnomaly: number;        // ν (rad)
  meanAnomaly?: number;       // M (rad)
  period?: number;            // T (s)
  specificEnergy?: number;    // ε (J/kg)
  specificAngularMomentum?: number;  // h (m²/s)
}

export interface CelestialBody {
  name: string;
  mass: number;               // kg
  radius: number;             // m
  position: Vector3D;         // m
  velocity: Vector3D;         // m/s
  mu?: number;                // GM (m³/s²)
  J2?: number;                // Oblateness coefficient
  atmosphereHeight?: number;  // m (scale height)
}

export interface ManeuverResult {
  deltaV: number;             // Total Δv (m/s)
  deltaVComponents: Vector3D[];  // Each burn
  transferTime: number;       // Transfer duration (s)
  burns: {
    position: Vector3D;
    velocity: Vector3D;
    deltaV: Vector3D;
    time: number;
  }[];
}

export interface LambertSolution {
  v1: Vector3D;               // Departure velocity
  v2: Vector3D;               // Arrival velocity
  transferTime: number;
  deltaV1: number;
  deltaV2: number;
  totalDeltaV: number;
  shortWay: boolean;
}

export interface LagrangePoints {
  L1: Vector3D;
  L2: Vector3D;
  L3: Vector3D;
  L4: Vector3D;
  L5: Vector3D;
}

export interface NBodyState {
  time: number;
  bodies: {
    name: string;
    position: Vector3D;
    velocity: Vector3D;
    mass: number;
  }[];
  totalEnergy: number;
  angularMomentum: Vector3D;
}

// Standard gravitational parameters (GM) in m³/s²
const SOLAR_SYSTEM = {
  Sun: { mass: 1.989e30, mu: 1.32712440018e20, radius: 6.957e8 },
  Mercury: { mass: 3.301e23, mu: 2.2032e13, radius: 2.4397e6, sma: 5.791e10 },
  Venus: { mass: 4.867e24, mu: 3.24859e14, radius: 6.0518e6, sma: 1.082e11 },
  Earth: { mass: 5.972e24, mu: 3.986004418e14, radius: 6.371e6, sma: 1.496e11, J2: 1.08263e-3 },
  Moon: { mass: 7.342e22, mu: 4.9048695e12, radius: 1.7374e6, sma: 3.844e8 },
  Mars: { mass: 6.417e23, mu: 4.282837e13, radius: 3.3895e6, sma: 2.279e11 },
  Jupiter: { mass: 1.898e27, mu: 1.26686534e17, radius: 6.9911e7, sma: 7.786e11 },
  Saturn: { mass: 5.683e26, mu: 3.7931187e16, radius: 5.8232e7, sma: 1.433e12 },
  Uranus: { mass: 8.681e25, mu: 5.793939e15, radius: 2.5362e7, sma: 2.872e12 },
  Neptune: { mass: 1.024e26, mu: 6.836529e15, radius: 2.4622e7, sma: 4.495e12 }
};

export class OrbitalMechanics {
  /**
   * Convert state vectors to orbital elements
   */
  static stateToElements(
    position: Vector3D,
    velocity: Vector3D,
    mu: number = SOLAR_SYSTEM.Earth.mu
  ): OrbitalElements {
    const r = this.magnitude(position);
    const v = this.magnitude(velocity);

    // Specific angular momentum
    const h = this.cross(position, velocity);
    const hMag = this.magnitude(h);

    // Node vector
    const k: Vector3D = { x: 0, y: 0, z: 1 };
    const n = this.cross(k, h);
    const nMag = this.magnitude(n);

    // Eccentricity vector
    const vCrossH = this.cross(velocity, h);
    const eVec: Vector3D = {
      x: vCrossH.x / mu - position.x / r,
      y: vCrossH.y / mu - position.y / r,
      z: vCrossH.z / mu - position.z / r
    };
    const e = this.magnitude(eVec);

    // Specific orbital energy
    const energy = v * v / 2 - mu / r;

    // Semi-major axis
    let a: number;
    if (Math.abs(e - 1) < 1e-10) {
      // Parabolic
      a = Infinity;
    } else {
      a = -mu / (2 * energy);
    }

    // Inclination
    const i = Math.acos(h.z / hMag);

    // Longitude of ascending node
    let Omega = 0;
    if (nMag > 1e-10) {
      Omega = Math.acos(n.x / nMag);
      if (n.y < 0) Omega = 2 * Math.PI - Omega;
    }

    // Argument of periapsis
    let omega = 0;
    if (nMag > 1e-10 && e > 1e-10) {
      omega = Math.acos(this.dot(n, eVec) / (nMag * e));
      if (eVec.z < 0) omega = 2 * Math.PI - omega;
    }

    // True anomaly
    let nu = 0;
    if (e > 1e-10) {
      nu = Math.acos(this.dot(eVec, position) / (e * r));
      if (this.dot(position, velocity) < 0) nu = 2 * Math.PI - nu;
    }

    // Period (for elliptical orbits)
    const period = e < 1 ? 2 * Math.PI * Math.sqrt(a * a * a / mu) : Infinity;

    // Mean anomaly
    let M = 0;
    if (e < 1) {
      const E = 2 * Math.atan(Math.sqrt((1 - e) / (1 + e)) * Math.tan(nu / 2));
      M = E - e * Math.sin(E);
      if (M < 0) M += 2 * Math.PI;
    }

    return {
      semiMajorAxis: a,
      eccentricity: e,
      inclination: i,
      longitudeOfAscendingNode: Omega,
      argumentOfPeriapsis: omega,
      trueAnomaly: nu,
      meanAnomaly: M,
      period,
      specificEnergy: energy,
      specificAngularMomentum: hMag
    };
  }

  /**
   * Convert orbital elements to state vectors
   */
  static elementsToState(
    elements: OrbitalElements,
    mu: number = SOLAR_SYSTEM.Earth.mu
  ): { position: Vector3D; velocity: Vector3D } {
    const { semiMajorAxis: a, eccentricity: e, inclination: i,
            longitudeOfAscendingNode: Omega, argumentOfPeriapsis: omega,
            trueAnomaly: nu } = elements;

    // Position in orbital plane
    const p = a * (1 - e * e);  // Semi-latus rectum
    const r = p / (1 + e * Math.cos(nu));

    const rPQW: Vector3D = {
      x: r * Math.cos(nu),
      y: r * Math.sin(nu),
      z: 0
    };

    // Velocity in orbital plane
    const sqrtMuP = Math.sqrt(mu / p);
    const vPQW: Vector3D = {
      x: -sqrtMuP * Math.sin(nu),
      y: sqrtMuP * (e + Math.cos(nu)),
      z: 0
    };

    // Rotation matrices
    const cosO = Math.cos(Omega);
    const sinO = Math.sin(Omega);
    const cosi = Math.cos(i);
    const sini = Math.sin(i);
    const cosw = Math.cos(omega);
    const sinw = Math.sin(omega);

    // Combined rotation matrix elements
    const R11 = cosO * cosw - sinO * sinw * cosi;
    const R12 = -cosO * sinw - sinO * cosw * cosi;
    const R21 = sinO * cosw + cosO * sinw * cosi;
    const R22 = -sinO * sinw + cosO * cosw * cosi;
    const R31 = sinw * sini;
    const R32 = cosw * sini;

    const position: Vector3D = {
      x: R11 * rPQW.x + R12 * rPQW.y,
      y: R21 * rPQW.x + R22 * rPQW.y,
      z: R31 * rPQW.x + R32 * rPQW.y
    };

    const velocity: Vector3D = {
      x: R11 * vPQW.x + R12 * vPQW.y,
      y: R21 * vPQW.x + R22 * vPQW.y,
      z: R31 * vPQW.x + R32 * vPQW.y
    };

    return { position, velocity };
  }

  /**
   * Hohmann transfer between circular coplanar orbits
   */
  static hohmannTransfer(
    r1: number,           // Initial orbit radius (m)
    r2: number,           // Final orbit radius (m)
    mu: number = SOLAR_SYSTEM.Earth.mu
  ): ManeuverResult {
    const a_transfer = (r1 + r2) / 2;

    // Circular velocities
    const v1_circ = Math.sqrt(mu / r1);
    const v2_circ = Math.sqrt(mu / r2);

    // Transfer orbit velocities
    const v1_transfer = Math.sqrt(mu * (2 / r1 - 1 / a_transfer));
    const v2_transfer = Math.sqrt(mu * (2 / r2 - 1 / a_transfer));

    // Delta-V
    const dv1 = Math.abs(v1_transfer - v1_circ);
    const dv2 = Math.abs(v2_circ - v2_transfer);

    // Transfer time (half the transfer orbit period)
    const transferTime = Math.PI * Math.sqrt(a_transfer ** 3 / mu);

    return {
      deltaV: dv1 + dv2,
      deltaVComponents: [
        { x: dv1, y: 0, z: 0 },
        { x: dv2, y: 0, z: 0 }
      ],
      transferTime,
      burns: [
        {
          position: { x: r1, y: 0, z: 0 },
          velocity: { x: 0, y: v1_circ, z: 0 },
          deltaV: { x: 0, y: dv1 * Math.sign(r2 - r1), z: 0 },
          time: 0
        },
        {
          position: { x: r2, y: 0, z: 0 },
          velocity: { x: 0, y: v2_transfer, z: 0 },
          deltaV: { x: 0, y: dv2 * Math.sign(r2 - r1), z: 0 },
          time: transferTime
        }
      ]
    };
  }

  /**
   * Bi-elliptic transfer (can be more efficient for large radius ratios)
   */
  static biEllipticTransfer(
    r1: number,
    r2: number,
    rb: number,           // Intermediate apoapsis radius
    mu: number = SOLAR_SYSTEM.Earth.mu
  ): ManeuverResult {
    // First ellipse: r1 to rb
    const a1 = (r1 + rb) / 2;
    const v1_circ = Math.sqrt(mu / r1);
    const v1_e1 = Math.sqrt(mu * (2 / r1 - 1 / a1));
    const vb_e1 = Math.sqrt(mu * (2 / rb - 1 / a1));

    // Second ellipse: rb to r2
    const a2 = (rb + r2) / 2;
    const vb_e2 = Math.sqrt(mu * (2 / rb - 1 / a2));
    const v2_e2 = Math.sqrt(mu * (2 / r2 - 1 / a2));
    const v2_circ = Math.sqrt(mu / r2);

    const dv1 = Math.abs(v1_e1 - v1_circ);
    const dv2 = Math.abs(vb_e2 - vb_e1);
    const dv3 = Math.abs(v2_circ - v2_e2);

    const t1 = Math.PI * Math.sqrt(a1 ** 3 / mu);
    const t2 = Math.PI * Math.sqrt(a2 ** 3 / mu);

    return {
      deltaV: dv1 + dv2 + dv3,
      deltaVComponents: [
        { x: dv1, y: 0, z: 0 },
        { x: dv2, y: 0, z: 0 },
        { x: dv3, y: 0, z: 0 }
      ],
      transferTime: t1 + t2,
      burns: [
        {
          position: { x: r1, y: 0, z: 0 },
          velocity: { x: 0, y: v1_circ, z: 0 },
          deltaV: { x: 0, y: dv1, z: 0 },
          time: 0
        },
        {
          position: { x: -rb, y: 0, z: 0 },
          velocity: { x: 0, y: -vb_e1, z: 0 },
          deltaV: { x: 0, y: -dv2, z: 0 },
          time: t1
        },
        {
          position: { x: r2, y: 0, z: 0 },
          velocity: { x: 0, y: v2_e2, z: 0 },
          deltaV: { x: 0, y: dv3, z: 0 },
          time: t1 + t2
        }
      ]
    };
  }

  /**
   * Plane change maneuver
   */
  static planeChange(
    r: number,            // Orbit radius
    deltaI: number,       // Inclination change (rad)
    mu: number = SOLAR_SYSTEM.Earth.mu
  ): number {
    const v = Math.sqrt(mu / r);
    return 2 * v * Math.sin(deltaI / 2);
  }

  /**
   * Lambert's problem solver
   * Finds the orbit connecting two positions in a given time
   */
  static solveLambert(
    r1: Vector3D,
    r2: Vector3D,
    tof: number,          // Time of flight (s)
    mu: number = SOLAR_SYSTEM.Earth.mu,
    shortWay: boolean = true,
    maxIterations: number = 100,
    tolerance: number = 1e-10
  ): LambertSolution | null {
    const r1Mag = this.magnitude(r1);
    const r2Mag = this.magnitude(r2);

    // Chord
    const cosNu = this.dot(r1, r2) / (r1Mag * r2Mag);
    let sinNu = Math.sqrt(1 - cosNu * cosNu);
    if (!shortWay) sinNu = -sinNu;

    // Transfer angle
    const A = Math.sqrt(r1Mag * r2Mag * (1 + cosNu));
    if (Math.abs(A) < tolerance) return null;

    // Initial guess for z (universal variable)
    let z = 0;
    let zLow = -4 * Math.PI * Math.PI;
    let zHigh = 4 * Math.PI * Math.PI;

    // Stumpff functions
    const C = (z: number): number => {
      if (z > tolerance) {
        const sqrtZ = Math.sqrt(z);
        return (1 - Math.cos(sqrtZ)) / z;
      } else if (z < -tolerance) {
        const sqrtNegZ = Math.sqrt(-z);
        return (1 - Math.cosh(sqrtNegZ)) / z;
      }
      return 1/2 - z/24 + z*z/720;
    };

    const S = (z: number): number => {
      if (z > tolerance) {
        const sqrtZ = Math.sqrt(z);
        return (sqrtZ - Math.sin(sqrtZ)) / Math.pow(sqrtZ, 3);
      } else if (z < -tolerance) {
        const sqrtNegZ = Math.sqrt(-z);
        return (Math.sinh(sqrtNegZ) - sqrtNegZ) / Math.pow(sqrtNegZ, 3);
      }
      return 1/6 - z/120 + z*z/5040;
    };

    // Newton-Raphson iteration
    for (let iter = 0; iter < maxIterations; iter++) {
      const Cz = C(z);
      const Sz = S(z);

      const y = r1Mag + r2Mag + A * (z * Sz - 1) / Math.sqrt(Cz);

      if (y < 0) {
        zLow = z;
        z = (z + zHigh) / 2;
        continue;
      }

      const x = Math.sqrt(y / Cz);
      const tofCalc = (x * x * x * Sz + A * Math.sqrt(y)) / Math.sqrt(mu);

      if (Math.abs(tofCalc - tof) < tolerance) {
        // Converged - calculate velocities
        const f = 1 - y / r1Mag;
        const g = A * Math.sqrt(y / mu);
        const gDot = 1 - y / r2Mag;

        const v1: Vector3D = {
          x: (r2.x - f * r1.x) / g,
          y: (r2.y - f * r1.y) / g,
          z: (r2.z - f * r1.z) / g
        };

        const v2: Vector3D = {
          x: (gDot * r2.x - r1.x) / g,
          y: (gDot * r2.y - r1.y) / g,
          z: (gDot * r2.z - r1.z) / g
        };

        return {
          v1,
          v2,
          transferTime: tof,
          deltaV1: this.magnitude(v1),
          deltaV2: this.magnitude(v2),
          totalDeltaV: this.magnitude(v1) + this.magnitude(v2),
          shortWay
        };
      }

      // Update z using bisection/Newton hybrid
      const dTdz = x * x * x * ((Cz - 3 * Sz / (2 * Cz)) / (2 * z) - 3 * Sz * Sz / Cz / (4 * Cz)) +
                   A * (3 * Sz * Math.sqrt(y) / Cz + A / (8 * Math.sqrt(y * Cz * Cz * Cz)));

      if (Math.abs(dTdz) > tolerance) {
        const zNew = z - (tofCalc - tof) / dTdz;
        if (zNew > zLow && zNew < zHigh) {
          z = zNew;
        } else {
          // Bisection fallback
          if (tofCalc < tof) {
            zHigh = z;
          } else {
            zLow = z;
          }
          z = (zLow + zHigh) / 2;
        }
      } else {
        z = (zLow + zHigh) / 2;
      }
    }

    return null;  // Did not converge
  }

  /**
   * Calculate Lagrange points for a two-body system
   */
  static lagrangePoints(
    m1: number,           // Primary mass (kg)
    m2: number,           // Secondary mass (kg)
    d: number             // Distance between bodies (m)
  ): LagrangePoints {
    const mu = m2 / (m1 + m2);  // Mass ratio
    const oneMinusMu = 1 - mu;

    // L1: Between the bodies
    // Solve: (1-mu)/(x+mu)² - mu/(x-1+mu)² = x
    let x1 = d * (1 - Math.cbrt(mu / 3));
    for (let i = 0; i < 50; i++) {
      const xi = x1 / d;
      const f = xi - oneMinusMu / Math.pow(xi + mu, 2) + mu / Math.pow(xi - oneMinusMu, 2);
      const df = 1 + 2 * oneMinusMu / Math.pow(xi + mu, 3) + 2 * mu / Math.pow(xi - oneMinusMu, 3);
      x1 = d * (xi - f / df);
    }

    // L2: Beyond the secondary
    let x2 = d * (1 + Math.cbrt(mu / 3));
    for (let i = 0; i < 50; i++) {
      const xi = x2 / d;
      const f = xi - oneMinusMu / Math.pow(xi + mu, 2) - mu / Math.pow(xi - oneMinusMu, 2);
      const df = 1 + 2 * oneMinusMu / Math.pow(xi + mu, 3) - 2 * mu / Math.pow(xi - oneMinusMu, 3);
      x2 = d * (xi - f / df);
    }

    // L3: Beyond the primary
    let x3 = -d * (1 + 5 * mu / 12);
    for (let i = 0; i < 50; i++) {
      const xi = x3 / d;
      const f = xi + oneMinusMu / Math.pow(xi + mu, 2) + mu / Math.pow(xi - oneMinusMu, 2);
      const df = 1 - 2 * oneMinusMu / Math.pow(xi + mu, 3) - 2 * mu / Math.pow(xi - oneMinusMu, 3);
      x3 = d * (xi - f / df);
    }

    // L4 and L5: Equilateral triangle points
    const x45 = d * (0.5 - mu);
    const y45 = d * Math.sqrt(3) / 2;

    return {
      L1: { x: x1 - mu * d, y: 0, z: 0 },
      L2: { x: x2 - mu * d, y: 0, z: 0 },
      L3: { x: x3 - mu * d, y: 0, z: 0 },
      L4: { x: x45 - mu * d, y: y45, z: 0 },
      L5: { x: x45 - mu * d, y: -y45, z: 0 }
    };
  }

  /**
   * N-body simulation using Velocity Verlet (symplectic integrator)
   */
  static simulateNBody(
    bodies: CelestialBody[],
    dt: number,
    duration: number,
    outputInterval: number = dt
  ): NBodyState[] {
    const G = PhysicalConstants.get('gravitational_constant').value;
    const n = bodies.length;
    const states: NBodyState[] = [];

    // Initialize positions, velocities, and masses
    const pos: Vector3D[] = bodies.map(b => ({ ...b.position }));
    const vel: Vector3D[] = bodies.map(b => ({ ...b.velocity }));
    const mass: number[] = bodies.map(b => b.mass);
    const names: string[] = bodies.map(b => b.name);

    // Calculate initial accelerations
    const acc: Vector3D[] = this.calculateAccelerations(pos, mass, G);

    let time = 0;
    let outputTime = 0;

    // Record initial state
    states.push(this.createNBodyState(time, names, pos, vel, mass, G));

    // Main integration loop (Velocity Verlet)
    while (time < duration) {
      // Update positions
      for (let i = 0; i < n; i++) {
        pos[i].x += vel[i].x * dt + 0.5 * acc[i].x * dt * dt;
        pos[i].y += vel[i].y * dt + 0.5 * acc[i].y * dt * dt;
        pos[i].z += vel[i].z * dt + 0.5 * acc[i].z * dt * dt;
      }

      // Calculate new accelerations
      const accNew = this.calculateAccelerations(pos, mass, G);

      // Update velocities
      for (let i = 0; i < n; i++) {
        vel[i].x += 0.5 * (acc[i].x + accNew[i].x) * dt;
        vel[i].y += 0.5 * (acc[i].y + accNew[i].y) * dt;
        vel[i].z += 0.5 * (acc[i].z + accNew[i].z) * dt;

        acc[i] = accNew[i];
      }

      time += dt;
      outputTime += dt;

      // Record state at intervals
      if (outputTime >= outputInterval) {
        states.push(this.createNBodyState(time, names, pos, vel, mass, G));
        outputTime = 0;
      }
    }

    return states;
  }

  /**
   * High-fidelity orbit propagation with perturbations
   */
  static propagateOrbitWithPerturbations(
    initialPosition: Vector3D,
    initialVelocity: Vector3D,
    centralBody: CelestialBody,
    dt: number,
    duration: number,
    options: {
      includeJ2?: boolean;
      includeDrag?: boolean;
      dragCoefficient?: number;
      crossSection?: number;
      mass?: number;
      includeSolarRadiation?: boolean;
      reflectivity?: number;
      includeThirdBody?: CelestialBody;
    } = {}
  ): { position: Vector3D; velocity: Vector3D; time: number }[] {
    const {
      includeJ2 = true,
      includeDrag = false,
      dragCoefficient = 2.2,
      crossSection = 10,
      mass = 1000,
      includeSolarRadiation = false,
      reflectivity = 1.4,
      includeThirdBody
    } = options;

    const mu = centralBody.mu || PhysicalConstants.get('gravitational_constant').value * centralBody.mass;
    const Re = centralBody.radius;
    const J2 = centralBody.J2 || 0;

    const trajectory: { position: Vector3D; velocity: Vector3D; time: number }[] = [];

    let pos = { ...initialPosition };
    let vel = { ...initialVelocity };
    let time = 0;

    trajectory.push({ position: { ...pos }, velocity: { ...vel }, time: 0 });

    // RK4 integration
    while (time < duration) {
      const acceleration = (p: Vector3D, v: Vector3D): Vector3D => {
        const r = this.magnitude(p);
        const r2 = r * r;
        const r3 = r2 * r;

        // Two-body acceleration
        let ax = -mu * p.x / r3;
        let ay = -mu * p.y / r3;
        let az = -mu * p.z / r3;

        // J2 perturbation
        if (includeJ2 && J2 !== 0) {
          const z2_r2 = (p.z * p.z) / r2;
          const factor = 1.5 * J2 * mu * Re * Re / (r2 * r3);

          ax += factor * p.x * (5 * z2_r2 - 1);
          ay += factor * p.y * (5 * z2_r2 - 1);
          az += factor * p.z * (5 * z2_r2 - 3);
        }

        // Atmospheric drag
        if (includeDrag && centralBody.atmosphereHeight) {
          const altitude = r - Re;
          if (altitude < centralBody.atmosphereHeight * 10) {
            // Exponential atmosphere model
            const rho0 = 1.225;  // Sea level density kg/m³
            const H = centralBody.atmosphereHeight;
            const rho = rho0 * Math.exp(-altitude / H);

            const vMag = this.magnitude(v);
            const dragAccel = -0.5 * rho * vMag * dragCoefficient * crossSection / mass;

            ax += dragAccel * v.x;
            ay += dragAccel * v.y;
            az += dragAccel * v.z;
          }
        }

        // Solar radiation pressure
        if (includeSolarRadiation) {
          const AU = 1.496e11;
          const solarFlux = 1361;  // W/m² at 1 AU
          const c = 299792458;
          const pressure = solarFlux / c;

          // Assume Sun is in +x direction at 1 AU
          const srpAccel = -pressure * reflectivity * crossSection / mass;
          ax += srpAccel;
        }

        // Third body perturbation
        if (includeThirdBody) {
          const mu3 = includeThirdBody.mu ||
                      PhysicalConstants.get('gravitational_constant').value * includeThirdBody.mass;

          // Vector from spacecraft to third body
          const r3b: Vector3D = {
            x: includeThirdBody.position.x - p.x,
            y: includeThirdBody.position.y - p.y,
            z: includeThirdBody.position.z - p.z
          };
          const d3 = this.magnitude(r3b);

          // Vector from central body to third body
          const d3c = this.magnitude(includeThirdBody.position);

          ax += mu3 * (r3b.x / (d3 * d3 * d3) - includeThirdBody.position.x / (d3c * d3c * d3c));
          ay += mu3 * (r3b.y / (d3 * d3 * d3) - includeThirdBody.position.y / (d3c * d3c * d3c));
          az += mu3 * (r3b.z / (d3 * d3 * d3) - includeThirdBody.position.z / (d3c * d3c * d3c));
        }

        return { x: ax, y: ay, z: az };
      };

      // RK4 step
      const k1v = acceleration(pos, vel);
      const k1r = vel;

      const p2: Vector3D = {
        x: pos.x + 0.5 * dt * k1r.x,
        y: pos.y + 0.5 * dt * k1r.y,
        z: pos.z + 0.5 * dt * k1r.z
      };
      const v2: Vector3D = {
        x: vel.x + 0.5 * dt * k1v.x,
        y: vel.y + 0.5 * dt * k1v.y,
        z: vel.z + 0.5 * dt * k1v.z
      };
      const k2v = acceleration(p2, v2);
      const k2r = v2;

      const p3: Vector3D = {
        x: pos.x + 0.5 * dt * k2r.x,
        y: pos.y + 0.5 * dt * k2r.y,
        z: pos.z + 0.5 * dt * k2r.z
      };
      const v3: Vector3D = {
        x: vel.x + 0.5 * dt * k2v.x,
        y: vel.y + 0.5 * dt * k2v.y,
        z: vel.z + 0.5 * dt * k2v.z
      };
      const k3v = acceleration(p3, v3);
      const k3r = v3;

      const p4: Vector3D = {
        x: pos.x + dt * k3r.x,
        y: pos.y + dt * k3r.y,
        z: pos.z + dt * k3r.z
      };
      const v4: Vector3D = {
        x: vel.x + dt * k3v.x,
        y: vel.y + dt * k3v.y,
        z: vel.z + dt * k3v.z
      };
      const k4v = acceleration(p4, v4);
      const k4r = v4;

      pos = {
        x: pos.x + (dt / 6) * (k1r.x + 2 * k2r.x + 2 * k3r.x + k4r.x),
        y: pos.y + (dt / 6) * (k1r.y + 2 * k2r.y + 2 * k3r.y + k4r.y),
        z: pos.z + (dt / 6) * (k1r.z + 2 * k2r.z + 2 * k3r.z + k4r.z)
      };

      vel = {
        x: vel.x + (dt / 6) * (k1v.x + 2 * k2v.x + 2 * k3v.x + k4v.x),
        y: vel.y + (dt / 6) * (k1v.y + 2 * k2v.y + 2 * k3v.y + k4v.y),
        z: vel.z + (dt / 6) * (k1v.z + 2 * k2v.z + 2 * k3v.z + k4v.z)
      };

      time += dt;
      trajectory.push({ position: { ...pos }, velocity: { ...vel }, time });
    }

    return trajectory;
  }

  /**
   * Interplanetary trajectory using patched conics
   */
  static interplanetaryTrajectory(
    departurePlanet: keyof typeof SOLAR_SYSTEM,
    arrivalPlanet: keyof typeof SOLAR_SYSTEM,
    departureDate: number,        // Days from epoch
    transferTime: number          // Days
  ): {
    departureInfinity: Vector3D;
    arrivalInfinity: Vector3D;
    c3: number;
    totalDeltaV: number;
    porkchopValue: number;
  } | null {
    const muSun = SOLAR_SYSTEM.Sun.mu;

    // Get planetary positions (simplified circular orbits)
    const dep = SOLAR_SYSTEM[departurePlanet] as { sma: number; mu: number };
    const arr = SOLAR_SYSTEM[arrivalPlanet] as { sma: number; mu: number };

    if (!dep.sma || !arr.sma) return null;

    // Angular velocities
    const nDep = Math.sqrt(muSun / (dep.sma ** 3));
    const nArr = Math.sqrt(muSun / (arr.sma ** 3));

    // Positions at departure and arrival
    const thetaDep = nDep * departureDate * 86400;
    const thetaArr = nArr * (departureDate + transferTime) * 86400;

    const r1: Vector3D = {
      x: dep.sma * Math.cos(thetaDep),
      y: dep.sma * Math.sin(thetaDep),
      z: 0
    };

    const r2: Vector3D = {
      x: arr.sma * Math.cos(thetaArr),
      y: arr.sma * Math.sin(thetaArr),
      z: 0
    };

    // Solve Lambert's problem
    const lambert = this.solveLambert(r1, r2, transferTime * 86400, muSun);
    if (!lambert) return null;

    // Planet velocities (circular)
    const vDep: Vector3D = {
      x: -Math.sqrt(muSun / dep.sma) * Math.sin(thetaDep),
      y: Math.sqrt(muSun / dep.sma) * Math.cos(thetaDep),
      z: 0
    };

    const vArr: Vector3D = {
      x: -Math.sqrt(muSun / arr.sma) * Math.sin(thetaArr),
      y: Math.sqrt(muSun / arr.sma) * Math.cos(thetaArr),
      z: 0
    };

    // Hyperbolic excess velocities
    const vInfDep: Vector3D = {
      x: lambert.v1.x - vDep.x,
      y: lambert.v1.y - vDep.y,
      z: lambert.v1.z - vDep.z
    };

    const vInfArr: Vector3D = {
      x: lambert.v2.x - vArr.x,
      y: lambert.v2.y - vArr.y,
      z: lambert.v2.z - vArr.z
    };

    // C3 (characteristic energy)
    const c3 = this.magnitude(vInfDep) ** 2;

    // Delta-V for escape from parking orbit (300 km altitude)
    const parkingAlt = 300000;
    const rPark = (SOLAR_SYSTEM[departurePlanet] as { radius: number }).radius + parkingAlt;
    const vPark = Math.sqrt(dep.mu / rPark);
    const vEscape = Math.sqrt(2 * dep.mu / rPark + c3);
    const dvDeparture = vEscape - vPark;

    // Delta-V for capture at arrival
    const rCapture = (SOLAR_SYSTEM[arrivalPlanet] as { radius: number }).radius + parkingAlt;
    const vInfArrMag = this.magnitude(vInfArr);
    const vCapture = Math.sqrt(2 * arr.mu / rCapture + vInfArrMag ** 2);
    const vCircular = Math.sqrt(arr.mu / rCapture);
    const dvArrival = vCapture - vCircular;

    return {
      departureInfinity: vInfDep,
      arrivalInfinity: vInfArr,
      c3,
      totalDeltaV: dvDeparture + dvArrival,
      porkchopValue: c3 + vInfArrMag ** 2  // Proxy for total energy
    };
  }

  /**
   * Ground track calculation for orbiting satellite
   */
  static groundTrack(
    elements: OrbitalElements,
    centralBody: { radius: number; rotationPeriod: number },
    duration: number,
    dt: number = 60
  ): { latitude: number; longitude: number; altitude: number; time: number }[] {
    const mu = SOLAR_SYSTEM.Earth.mu;
    const track: { latitude: number; longitude: number; altitude: number; time: number }[] = [];

    // Earth rotation rate
    const omegaE = 2 * Math.PI / centralBody.rotationPeriod;

    // Initial state
    let { position, velocity } = this.elementsToState(elements, mu);
    let time = 0;

    while (time < duration) {
      const r = this.magnitude(position);
      const altitude = r - centralBody.radius;

      // Latitude (declination)
      const latitude = Math.asin(position.z / r) * 180 / Math.PI;

      // Longitude (accounting for Earth rotation)
      let longitude = Math.atan2(position.y, position.x) - omegaE * time;
      longitude = longitude * 180 / Math.PI;
      longitude = ((longitude + 180) % 360 + 360) % 360 - 180;

      track.push({ latitude, longitude, altitude, time });

      // Simple Keplerian propagation
      const a = elements.semiMajorAxis;
      const n = Math.sqrt(mu / (a * a * a));
      const dM = n * dt;

      // Update mean anomaly
      elements.meanAnomaly = ((elements.meanAnomaly || 0) + dM) % (2 * Math.PI);

      // Convert to true anomaly (iterative solution of Kepler's equation)
      let E = elements.meanAnomaly!;
      for (let i = 0; i < 10; i++) {
        E = elements.meanAnomaly! + elements.eccentricity * Math.sin(E);
      }
      elements.trueAnomaly = 2 * Math.atan(
        Math.sqrt((1 + elements.eccentricity) / (1 - elements.eccentricity)) * Math.tan(E / 2)
      );

      const state = this.elementsToState(elements, mu);
      position = state.position;
      velocity = state.velocity;

      time += dt;
    }

    return track;
  }

  // Helper functions
  private static magnitude(v: Vector3D): number {
    return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  }

  private static dot(a: Vector3D, b: Vector3D): number {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  private static cross(a: Vector3D, b: Vector3D): Vector3D {
    return {
      x: a.y * b.z - a.z * b.y,
      y: a.z * b.x - a.x * b.z,
      z: a.x * b.y - a.y * b.x
    };
  }

  private static calculateAccelerations(
    positions: Vector3D[],
    masses: number[],
    G: number
  ): Vector3D[] {
    const n = positions.length;
    const acc: Vector3D[] = positions.map(() => ({ x: 0, y: 0, z: 0 }));

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = positions[j].x - positions[i].x;
        const dy = positions[j].y - positions[i].y;
        const dz = positions[j].z - positions[i].z;

        const r2 = dx * dx + dy * dy + dz * dz;
        const r = Math.sqrt(r2);
        const r3 = r2 * r;

        const factor = G / r3;

        acc[i].x += factor * masses[j] * dx;
        acc[i].y += factor * masses[j] * dy;
        acc[i].z += factor * masses[j] * dz;

        acc[j].x -= factor * masses[i] * dx;
        acc[j].y -= factor * masses[i] * dy;
        acc[j].z -= factor * masses[i] * dz;
      }
    }

    return acc;
  }

  private static createNBodyState(
    time: number,
    names: string[],
    positions: Vector3D[],
    velocities: Vector3D[],
    masses: number[],
    G: number
  ): NBodyState {
    const n = positions.length;

    // Calculate total energy
    let kineticEnergy = 0;
    let potentialEnergy = 0;

    for (let i = 0; i < n; i++) {
      const v2 = velocities[i].x ** 2 + velocities[i].y ** 2 + velocities[i].z ** 2;
      kineticEnergy += 0.5 * masses[i] * v2;

      for (let j = i + 1; j < n; j++) {
        const dx = positions[j].x - positions[i].x;
        const dy = positions[j].y - positions[i].y;
        const dz = positions[j].z - positions[i].z;
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        potentialEnergy -= G * masses[i] * masses[j] / r;
      }
    }

    // Calculate total angular momentum
    const L: Vector3D = { x: 0, y: 0, z: 0 };
    for (let i = 0; i < n; i++) {
      const li = this.cross(positions[i], {
        x: masses[i] * velocities[i].x,
        y: masses[i] * velocities[i].y,
        z: masses[i] * velocities[i].z
      });
      L.x += li.x;
      L.y += li.y;
      L.z += li.z;
    }

    return {
      time,
      bodies: names.map((name, i) => ({
        name,
        position: { ...positions[i] },
        velocity: { ...velocities[i] },
        mass: masses[i]
      })),
      totalEnergy: kineticEnergy + potentialEnergy,
      angularMomentum: L
    };
  }
}

// Export solar system data for external use
export { SOLAR_SYSTEM };
