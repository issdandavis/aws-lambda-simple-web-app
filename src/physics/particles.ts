/**
 * Particle Dynamics Simulation Module
 * Implements classical and relativistic particle mechanics
 */

import { PhysicalConstants as PC } from './constants';

export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

export interface Particle {
  mass: number;           // kg
  charge: number;         // Coulombs
  position: Vector3D;     // m
  velocity: Vector3D;     // m/s
  spin?: number;          // quantum spin
}

export interface Force {
  type: 'gravitational' | 'electromagnetic' | 'spring' | 'drag' | 'custom';
  magnitude: number;
  direction: Vector3D;
}

export interface SimulationState {
  time: number;
  particles: Particle[];
  totalEnergy: number;
  totalMomentum: Vector3D;
  centerOfMass: Vector3D;
}

export interface CollisionResult {
  particle1Final: { velocity: Vector3D; kineticEnergy: number };
  particle2Final: { velocity: Vector3D; kineticEnergy: number };
  energyLoss: number;
  impactParameter: number;
}

export interface OrbitalElements {
  semiMajorAxis: number;      // a (m)
  eccentricity: number;       // e
  periapsis: number;          // closest approach (m)
  apoapsis: number;           // farthest distance (m)
  orbitalPeriod: number;      // T (s)
  orbitalVelocity: number;    // current v (m/s)
  specificEnergy: number;     // E/m (J/kg)
  angularMomentum: number;    // L (kg·m²/s)
}

export interface RelativisticProperties {
  restMass: number;           // m₀ (kg)
  relativisticMass: number;   // γm₀ (kg)
  lorentzFactor: number;      // γ
  kineticEnergy: number;      // (γ-1)m₀c² (J)
  totalEnergy: number;        // γm₀c² (J)
  momentum: number;           // γm₀v (kg·m/s)
  properTime: number;         // τ relative to coordinate time
}

/**
 * Particle Dynamics Calculator
 */
export class ParticleDynamics {

  /**
   * Vector operations
   */
  static vectorAdd(a: Vector3D, b: Vector3D): Vector3D {
    return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
  }

  static vectorSubtract(a: Vector3D, b: Vector3D): Vector3D {
    return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  }

  static vectorScale(v: Vector3D, scalar: number): Vector3D {
    return { x: v.x * scalar, y: v.y * scalar, z: v.z * scalar };
  }

  static vectorDot(a: Vector3D, b: Vector3D): number {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  static vectorCross(a: Vector3D, b: Vector3D): Vector3D {
    return {
      x: a.y * b.z - a.z * b.y,
      y: a.z * b.x - a.x * b.z,
      z: a.x * b.y - a.y * b.x,
    };
  }

  static vectorMagnitude(v: Vector3D): number {
    return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  }

  static vectorNormalize(v: Vector3D): Vector3D {
    const mag = this.vectorMagnitude(v);
    if (mag === 0) return { x: 0, y: 0, z: 0 };
    return this.vectorScale(v, 1 / mag);
  }

  /**
   * Calculate gravitational force between two masses
   * F = -G * m1 * m2 / r² * r̂
   */
  static gravitationalForce(m1: number, m2: number, r: Vector3D): Vector3D {
    const distance = this.vectorMagnitude(r);
    if (distance === 0) return { x: 0, y: 0, z: 0 };

    const forceMagnitude = -PC.G * m1 * m2 / (distance * distance);
    const direction = this.vectorNormalize(r);

    return this.vectorScale(direction, forceMagnitude);
  }

  /**
   * Calculate electrostatic (Coulomb) force between two charges
   * F = k * q1 * q2 / r² * r̂
   */
  static electrostaticForce(q1: number, q2: number, r: Vector3D): Vector3D {
    const distance = this.vectorMagnitude(r);
    if (distance === 0) return { x: 0, y: 0, z: 0 };

    const forceMagnitude = PC.ke * q1 * q2 / (distance * distance);
    const direction = this.vectorNormalize(r);

    return this.vectorScale(direction, forceMagnitude);
  }

  /**
   * Calculate Lorentz force on a charged particle
   * F = q(E + v × B)
   */
  static lorentzForce(
    charge: number,
    velocity: Vector3D,
    electricField: Vector3D,
    magneticField: Vector3D
  ): Vector3D {
    const vCrossB = this.vectorCross(velocity, magneticField);
    const totalField = this.vectorAdd(electricField, vCrossB);
    return this.vectorScale(totalField, charge);
  }

  /**
   * Calculate orbital elements from position and velocity
   */
  static calculateOrbitalElements(
    centralMass: number,
    orbiterMass: number,
    position: Vector3D,
    velocity: Vector3D
  ): OrbitalElements {
    const mu = PC.G * (centralMass + orbiterMass); // Gravitational parameter
    const r = this.vectorMagnitude(position);
    const v = this.vectorMagnitude(velocity);

    // Specific orbital energy
    const specificEnergy = 0.5 * v * v - mu / r;

    // Specific angular momentum vector
    const hVector = this.vectorCross(position, velocity);
    const h = this.vectorMagnitude(hVector);

    // Semi-major axis
    const semiMajorAxis = specificEnergy < 0 ? -mu / (2 * specificEnergy) : Infinity;

    // Eccentricity vector
    const vCrossH = this.vectorCross(velocity, hVector);
    const rNorm = this.vectorNormalize(position);
    const eVector = this.vectorSubtract(
      this.vectorScale(vCrossH, 1 / mu),
      rNorm
    );
    const eccentricity = this.vectorMagnitude(eVector);

    // Periapsis and apoapsis
    const periapsis = semiMajorAxis * (1 - eccentricity);
    const apoapsis = eccentricity < 1 ? semiMajorAxis * (1 + eccentricity) : Infinity;

    // Orbital period (for elliptical orbits)
    const orbitalPeriod = eccentricity < 1
      ? 2 * Math.PI * Math.sqrt(Math.pow(semiMajorAxis, 3) / mu)
      : Infinity;

    return {
      semiMajorAxis,
      eccentricity,
      periapsis,
      apoapsis,
      orbitalPeriod,
      orbitalVelocity: v,
      specificEnergy,
      angularMomentum: h * orbiterMass,
    };
  }

  /**
   * N-body simulation step using Velocity Verlet integration
   */
  static nBodyStep(
    particles: Particle[],
    dt: number,
    includeElectrostatic: boolean = false
  ): Particle[] {
    const n = particles.length;
    const accelerations: Vector3D[] = new Array(n).fill(null).map(() => ({ x: 0, y: 0, z: 0 }));

    // Calculate accelerations
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const r = this.vectorSubtract(particles[j].position, particles[i].position);

        // Gravitational force
        const gravForce = this.gravitationalForce(
          particles[i].mass,
          particles[j].mass,
          r
        );

        // Electrostatic force (if applicable)
        let elecForce: Vector3D = { x: 0, y: 0, z: 0 };
        if (includeElectrostatic && particles[i].charge && particles[j].charge) {
          elecForce = this.electrostaticForce(
            particles[i].charge,
            particles[j].charge,
            r
          );
        }

        const totalForce = this.vectorAdd(gravForce, elecForce);

        // F = ma, so a = F/m
        accelerations[i] = this.vectorAdd(
          accelerations[i],
          this.vectorScale(totalForce, -1 / particles[i].mass)
        );
        accelerations[j] = this.vectorAdd(
          accelerations[j],
          this.vectorScale(totalForce, 1 / particles[j].mass)
        );
      }
    }

    // Update positions and velocities (Velocity Verlet)
    const newParticles: Particle[] = [];

    for (let i = 0; i < n; i++) {
      const p = particles[i];

      // x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
      const newPosition = this.vectorAdd(
        this.vectorAdd(p.position, this.vectorScale(p.velocity, dt)),
        this.vectorScale(accelerations[i], 0.5 * dt * dt)
      );

      // v(t+dt) = v(t) + a(t)*dt (simplified, should use average acceleration)
      const newVelocity = this.vectorAdd(
        p.velocity,
        this.vectorScale(accelerations[i], dt)
      );

      newParticles.push({
        ...p,
        position: newPosition,
        velocity: newVelocity,
      });
    }

    return newParticles;
  }

  /**
   * Elastic collision between two particles (1D)
   */
  static elasticCollision(
    m1: number, v1: number,
    m2: number, v2: number
  ): { v1Final: number; v2Final: number } {
    const totalMass = m1 + m2;

    const v1Final = ((m1 - m2) * v1 + 2 * m2 * v2) / totalMass;
    const v2Final = ((m2 - m1) * v2 + 2 * m1 * v1) / totalMass;

    return { v1Final, v2Final };
  }

  /**
   * Inelastic collision with coefficient of restitution
   */
  static inelasticCollision(
    m1: number, v1: number,
    m2: number, v2: number,
    restitution: number  // 0 = perfectly inelastic, 1 = elastic
  ): { v1Final: number; v2Final: number; energyLoss: number } {
    const totalMass = m1 + m2;

    // Conservation of momentum: m1*v1 + m2*v2 = m1*v1' + m2*v2'
    // Coefficient of restitution: e = (v2' - v1') / (v1 - v2)

    const v1Final = (m1 * v1 + m2 * v2 + m2 * restitution * (v2 - v1)) / totalMass;
    const v2Final = (m1 * v1 + m2 * v2 + m1 * restitution * (v1 - v2)) / totalMass;

    const initialKE = 0.5 * m1 * v1 * v1 + 0.5 * m2 * v2 * v2;
    const finalKE = 0.5 * m1 * v1Final * v1Final + 0.5 * m2 * v2Final * v2Final;
    const energyLoss = initialKE - finalKE;

    return { v1Final, v2Final, energyLoss };
  }

  /**
   * Relativistic calculations
   */
  static calculateRelativistic(
    restMass: number,
    velocity: number
  ): RelativisticProperties {
    const beta = velocity / PC.c;

    if (beta >= 1) {
      throw new Error('Velocity cannot equal or exceed the speed of light');
    }

    const lorentzFactor = 1 / Math.sqrt(1 - beta * beta);
    const relativisticMass = lorentzFactor * restMass;
    const totalEnergy = lorentzFactor * restMass * PC.c * PC.c;
    const restEnergy = restMass * PC.c * PC.c;
    const kineticEnergy = totalEnergy - restEnergy;
    const momentum = lorentzFactor * restMass * velocity;
    const properTime = 1 / lorentzFactor; // Time dilation factor

    return {
      restMass,
      relativisticMass,
      lorentzFactor,
      kineticEnergy,
      totalEnergy,
      momentum,
      properTime,
    };
  }

  /**
   * Calculate escape velocity
   * v_escape = sqrt(2GM/r)
   */
  static escapeVelocity(centralMass: number, radius: number): number {
    return Math.sqrt(2 * PC.G * centralMass / radius);
  }

  /**
   * Calculate Schwarzschild radius (event horizon of non-rotating black hole)
   * r_s = 2GM/c²
   */
  static schwarzschildRadius(mass: number): number {
    return 2 * PC.G * mass / (PC.c * PC.c);
  }

  /**
   * Simple pendulum period
   * T = 2π * sqrt(L/g)
   */
  static pendulumPeriod(length: number, gravity: number = 9.80665): number {
    return 2 * Math.PI * Math.sqrt(length / gravity);
  }

  /**
   * Drag force calculation
   * F_d = 0.5 * ρ * v² * C_d * A
   */
  static dragForce(
    fluidDensity: number,  // kg/m³
    velocity: number,       // m/s
    dragCoefficient: number,
    crossSectionalArea: number  // m²
  ): number {
    return 0.5 * fluidDensity * velocity * velocity * dragCoefficient * crossSectionalArea;
  }

  /**
   * Terminal velocity
   * v_t = sqrt(2mg / (ρ * A * C_d))
   */
  static terminalVelocity(
    mass: number,
    gravity: number,
    fluidDensity: number,
    crossSectionalArea: number,
    dragCoefficient: number
  ): number {
    return Math.sqrt(2 * mass * gravity / (fluidDensity * crossSectionalArea * dragCoefficient));
  }

  /**
   * Calculate kinetic energy
   */
  static kineticEnergy(mass: number, velocity: Vector3D): number {
    const v = this.vectorMagnitude(velocity);
    return 0.5 * mass * v * v;
  }

  /**
   * Calculate gravitational potential energy
   */
  static gravitationalPotentialEnergy(m1: number, m2: number, distance: number): number {
    return -PC.G * m1 * m2 / distance;
  }

  /**
   * Calculate center of mass for a system of particles
   */
  static centerOfMass(particles: Particle[]): Vector3D {
    let totalMass = 0;
    let sumMR: Vector3D = { x: 0, y: 0, z: 0 };

    for (const p of particles) {
      totalMass += p.mass;
      sumMR = this.vectorAdd(sumMR, this.vectorScale(p.position, p.mass));
    }

    return this.vectorScale(sumMR, 1 / totalMass);
  }

  /**
   * Calculate total momentum of a system
   */
  static totalMomentum(particles: Particle[]): Vector3D {
    let momentum: Vector3D = { x: 0, y: 0, z: 0 };

    for (const p of particles) {
      momentum = this.vectorAdd(momentum, this.vectorScale(p.velocity, p.mass));
    }

    return momentum;
  }
}
