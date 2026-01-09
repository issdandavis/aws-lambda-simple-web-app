/**
 * Particle Swarm Optimization (PSO) Module
 *
 * Implements various PSO algorithms for optimization problems in physics simulations:
 * - Standard PSO with inertia weight
 * - Constriction factor PSO
 * - Multi-objective PSO (MOPSO)
 * - Constrained optimization with penalty functions
 * - Adaptive parameter control
 */

export interface Particle {
  position: number[];
  velocity: number[];
  bestPosition: number[];
  bestFitness: number;
  fitness: number;
}

export interface PSOConfig {
  dimensions: number;
  swarmSize: number;
  maxIterations: number;
  bounds: { min: number[]; max: number[] };
  inertiaWeight?: number;
  cognitiveWeight?: number;  // c1
  socialWeight?: number;      // c2
  velocityClamp?: number;
  tolerance?: number;
  useConstriction?: boolean;
  adaptiveInertia?: boolean;
}

export interface PSOResult {
  bestPosition: number[];
  bestFitness: number;
  convergenceHistory: number[];
  iterations: number;
  swarmDiversity: number;
  converged: boolean;
}

export interface MOPSOResult {
  paretoFront: { position: number[]; objectives: number[] }[];
  iterations: number;
  hypervolume: number;
}

export interface ConstrainedResult extends PSOResult {
  constraintViolation: number;
  feasible: boolean;
}

type ObjectiveFunction = (x: number[]) => number;
type MultiObjectiveFunction = (x: number[]) => number[];
type ConstraintFunction = (x: number[]) => number;  // Should return <= 0 if satisfied

export class ParticleSwarmOptimizer {
  /**
   * Standard PSO with inertia weight
   */
  static optimize(
    objective: ObjectiveFunction,
    config: PSOConfig
  ): PSOResult {
    const {
      dimensions,
      swarmSize,
      maxIterations,
      bounds,
      inertiaWeight: w0 = 0.729,
      cognitiveWeight: c1 = 1.49445,
      socialWeight: c2 = 1.49445,
      velocityClamp = Infinity,
      tolerance = 1e-10,
      adaptiveInertia = true
    } = config;

    // Initialize swarm
    const swarm: Particle[] = [];
    let globalBestPosition = new Array(dimensions).fill(0);
    let globalBestFitness = Infinity;

    for (let i = 0; i < swarmSize; i++) {
      const position = bounds.min.map((min, d) =>
        min + Math.random() * (bounds.max[d] - min)
      );
      const velocity = bounds.min.map((min, d) =>
        (Math.random() - 0.5) * (bounds.max[d] - min) * 0.1
      );
      const fitness = objective(position);

      swarm.push({
        position,
        velocity,
        bestPosition: [...position],
        bestFitness: fitness,
        fitness
      });

      if (fitness < globalBestFitness) {
        globalBestFitness = fitness;
        globalBestPosition = [...position];
      }
    }

    const convergenceHistory: number[] = [globalBestFitness];
    let iteration = 0;
    let converged = false;
    let w = w0;

    // Main PSO loop
    for (iteration = 0; iteration < maxIterations; iteration++) {
      // Adaptive inertia weight (linearly decreasing)
      if (adaptiveInertia) {
        w = w0 - (w0 - 0.4) * (iteration / maxIterations);
      }

      for (const particle of swarm) {
        // Update velocity
        for (let d = 0; d < dimensions; d++) {
          const r1 = Math.random();
          const r2 = Math.random();

          particle.velocity[d] =
            w * particle.velocity[d] +
            c1 * r1 * (particle.bestPosition[d] - particle.position[d]) +
            c2 * r2 * (globalBestPosition[d] - particle.position[d]);

          // Velocity clamping
          const maxVel = Math.min(velocityClamp, (bounds.max[d] - bounds.min[d]) * 0.5);
          particle.velocity[d] = Math.max(-maxVel, Math.min(maxVel, particle.velocity[d]));
        }

        // Update position
        for (let d = 0; d < dimensions; d++) {
          particle.position[d] += particle.velocity[d];

          // Boundary handling (reflection)
          if (particle.position[d] < bounds.min[d]) {
            particle.position[d] = bounds.min[d];
            particle.velocity[d] *= -0.5;
          } else if (particle.position[d] > bounds.max[d]) {
            particle.position[d] = bounds.max[d];
            particle.velocity[d] *= -0.5;
          }
        }

        // Evaluate fitness
        particle.fitness = objective(particle.position);

        // Update personal best
        if (particle.fitness < particle.bestFitness) {
          particle.bestFitness = particle.fitness;
          particle.bestPosition = [...particle.position];

          // Update global best
          if (particle.fitness < globalBestFitness) {
            globalBestFitness = particle.fitness;
            globalBestPosition = [...particle.position];
          }
        }
      }

      convergenceHistory.push(globalBestFitness);

      // Check convergence
      if (convergenceHistory.length > 50) {
        const recent = convergenceHistory.slice(-50);
        const improvement = recent[0] - recent[recent.length - 1];
        if (Math.abs(improvement) < tolerance) {
          converged = true;
          break;
        }
      }
    }

    // Calculate swarm diversity
    const swarmDiversity = this.calculateDiversity(swarm, bounds);

    return {
      bestPosition: globalBestPosition,
      bestFitness: globalBestFitness,
      convergenceHistory,
      iterations: iteration + 1,
      swarmDiversity,
      converged
    };
  }

  /**
   * PSO with constriction factor (Clerc-Kennedy)
   * Often provides better convergence than inertia weight
   */
  static optimizeConstriction(
    objective: ObjectiveFunction,
    config: PSOConfig
  ): PSOResult {
    const phi = (config.cognitiveWeight || 2.05) + (config.socialWeight || 2.05);
    const chi = 2 / Math.abs(2 - phi - Math.sqrt(phi * phi - 4 * phi));

    return this.optimize(objective, {
      ...config,
      inertiaWeight: chi,
      cognitiveWeight: chi * (config.cognitiveWeight || 2.05),
      socialWeight: chi * (config.socialWeight || 2.05),
      useConstriction: true,
      adaptiveInertia: false
    });
  }

  /**
   * Multi-Objective PSO using Pareto dominance
   */
  static optimizeMultiObjective(
    objectives: MultiObjectiveFunction,
    config: PSOConfig,
    archiveSize: number = 100
  ): MOPSOResult {
    const {
      dimensions,
      swarmSize,
      maxIterations,
      bounds,
      inertiaWeight: w0 = 0.4,
      cognitiveWeight: c1 = 1.0,
      socialWeight: c2 = 2.0
    } = config;

    interface MOParticle extends Particle {
      objectives: number[];
      bestObjectives: number[];
    }

    // Initialize swarm
    const swarm: MOParticle[] = [];
    const archive: { position: number[]; objectives: number[] }[] = [];

    for (let i = 0; i < swarmSize; i++) {
      const position = bounds.min.map((min, d) =>
        min + Math.random() * (bounds.max[d] - min)
      );
      const velocity = bounds.min.map((min, d) =>
        (Math.random() - 0.5) * (bounds.max[d] - min) * 0.1
      );
      const objs = objectives(position);

      swarm.push({
        position,
        velocity,
        bestPosition: [...position],
        bestFitness: 0,
        fitness: 0,
        objectives: objs,
        bestObjectives: [...objs]
      });
    }

    // Update archive with initial particles
    this.updateArchive(archive, swarm, archiveSize);

    // Main MOPSO loop
    for (let iteration = 0; iteration < maxIterations; iteration++) {
      const w = w0 + (0.9 - w0) * (1 - iteration / maxIterations);

      for (const particle of swarm) {
        // Select leader from archive (crowding distance tournament)
        const leader = this.selectLeader(archive);

        // Update velocity
        for (let d = 0; d < dimensions; d++) {
          const r1 = Math.random();
          const r2 = Math.random();

          particle.velocity[d] =
            w * particle.velocity[d] +
            c1 * r1 * (particle.bestPosition[d] - particle.position[d]) +
            c2 * r2 * (leader.position[d] - particle.position[d]);
        }

        // Update position
        for (let d = 0; d < dimensions; d++) {
          particle.position[d] += particle.velocity[d];
          particle.position[d] = Math.max(bounds.min[d],
            Math.min(bounds.max[d], particle.position[d]));
        }

        // Evaluate objectives
        particle.objectives = objectives(particle.position);

        // Update personal best (if current dominates or random selection)
        if (this.dominates(particle.objectives, particle.bestObjectives) ||
            (!this.dominates(particle.bestObjectives, particle.objectives) && Math.random() < 0.5)) {
          particle.bestPosition = [...particle.position];
          particle.bestObjectives = [...particle.objectives];
        }
      }

      // Update archive
      this.updateArchive(archive, swarm, archiveSize);
    }

    // Calculate hypervolume indicator
    const hypervolume = this.calculateHypervolume(archive, bounds);

    return {
      paretoFront: archive,
      iterations: maxIterations,
      hypervolume
    };
  }

  /**
   * Constrained optimization with penalty function
   */
  static optimizeConstrained(
    objective: ObjectiveFunction,
    constraints: ConstraintFunction[],
    config: PSOConfig,
    penaltyCoefficient: number = 1000
  ): ConstrainedResult {
    // Create penalized objective function
    const penalizedObjective = (x: number[]): number => {
      let fitness = objective(x);
      let totalViolation = 0;

      for (const constraint of constraints) {
        const violation = constraint(x);
        if (violation > 0) {
          totalViolation += violation;
          fitness += penaltyCoefficient * violation * violation;
        }
      }

      return fitness;
    };

    const result = this.optimize(penalizedObjective, config);

    // Check final constraint violation
    let constraintViolation = 0;
    for (const constraint of constraints) {
      const violation = constraint(result.bestPosition);
      if (violation > 0) {
        constraintViolation += violation;
      }
    }

    return {
      ...result,
      constraintViolation,
      feasible: constraintViolation < 1e-6
    };
  }

  /**
   * Local best PSO with ring topology (slower convergence, better exploration)
   */
  static optimizeLocalBest(
    objective: ObjectiveFunction,
    config: PSOConfig,
    neighborhoodSize: number = 3
  ): PSOResult {
    const {
      dimensions,
      swarmSize,
      maxIterations,
      bounds,
      inertiaWeight: w0 = 0.729,
      cognitiveWeight: c1 = 1.49445,
      socialWeight: c2 = 1.49445,
      tolerance = 1e-10
    } = config;

    // Initialize swarm
    const swarm: Particle[] = [];
    let globalBestPosition = new Array(dimensions).fill(0);
    let globalBestFitness = Infinity;

    for (let i = 0; i < swarmSize; i++) {
      const position = bounds.min.map((min, d) =>
        min + Math.random() * (bounds.max[d] - min)
      );
      const velocity = bounds.min.map((min, d) =>
        (Math.random() - 0.5) * (bounds.max[d] - min) * 0.1
      );
      const fitness = objective(position);

      swarm.push({
        position,
        velocity,
        bestPosition: [...position],
        bestFitness: fitness,
        fitness
      });

      if (fitness < globalBestFitness) {
        globalBestFitness = fitness;
        globalBestPosition = [...position];
      }
    }

    const convergenceHistory: number[] = [globalBestFitness];
    let iteration = 0;
    let converged = false;

    // Main PSO loop with ring topology
    for (iteration = 0; iteration < maxIterations; iteration++) {
      const w = w0 - (w0 - 0.4) * (iteration / maxIterations);

      for (let i = 0; i < swarmSize; i++) {
        const particle = swarm[i];

        // Find local best in neighborhood
        let localBestFitness = particle.bestFitness;
        let localBestPosition = [...particle.bestPosition];

        const halfNeighborhood = Math.floor(neighborhoodSize / 2);
        for (let j = -halfNeighborhood; j <= halfNeighborhood; j++) {
          const neighborIdx = (i + j + swarmSize) % swarmSize;
          if (swarm[neighborIdx].bestFitness < localBestFitness) {
            localBestFitness = swarm[neighborIdx].bestFitness;
            localBestPosition = [...swarm[neighborIdx].bestPosition];
          }
        }

        // Update velocity using local best
        for (let d = 0; d < dimensions; d++) {
          const r1 = Math.random();
          const r2 = Math.random();

          particle.velocity[d] =
            w * particle.velocity[d] +
            c1 * r1 * (particle.bestPosition[d] - particle.position[d]) +
            c2 * r2 * (localBestPosition[d] - particle.position[d]);
        }

        // Update position
        for (let d = 0; d < dimensions; d++) {
          particle.position[d] += particle.velocity[d];
          particle.position[d] = Math.max(bounds.min[d],
            Math.min(bounds.max[d], particle.position[d]));
        }

        // Evaluate fitness
        particle.fitness = objective(particle.position);

        // Update personal best
        if (particle.fitness < particle.bestFitness) {
          particle.bestFitness = particle.fitness;
          particle.bestPosition = [...particle.position];

          // Update global best
          if (particle.fitness < globalBestFitness) {
            globalBestFitness = particle.fitness;
            globalBestPosition = [...particle.position];
          }
        }
      }

      convergenceHistory.push(globalBestFitness);

      // Check convergence
      if (convergenceHistory.length > 50) {
        const recent = convergenceHistory.slice(-50);
        const improvement = recent[0] - recent[recent.length - 1];
        if (Math.abs(improvement) < tolerance) {
          converged = true;
          break;
        }
      }
    }

    return {
      bestPosition: globalBestPosition,
      bestFitness: globalBestFitness,
      convergenceHistory,
      iterations: iteration + 1,
      swarmDiversity: this.calculateDiversity(swarm, bounds),
      converged
    };
  }

  /**
   * Differential Evolution (bonus optimizer)
   * Good for non-separable, multimodal functions
   */
  static differentialEvolution(
    objective: ObjectiveFunction,
    config: {
      dimensions: number;
      populationSize: number;
      maxIterations: number;
      bounds: { min: number[]; max: number[] };
      mutationFactor?: number;  // F
      crossoverRate?: number;   // CR
      tolerance?: number;
    }
  ): PSOResult {
    const {
      dimensions,
      populationSize,
      maxIterations,
      bounds,
      mutationFactor: F = 0.8,
      crossoverRate: CR = 0.9,
      tolerance = 1e-10
    } = config;

    // Initialize population
    let population: number[][] = [];
    let fitness: number[] = [];

    for (let i = 0; i < populationSize; i++) {
      const individual = bounds.min.map((min, d) =>
        min + Math.random() * (bounds.max[d] - min)
      );
      population.push(individual);
      fitness.push(objective(individual));
    }

    let bestIdx = fitness.indexOf(Math.min(...fitness));
    let bestFitness = fitness[bestIdx];
    let bestPosition = [...population[bestIdx]];
    const convergenceHistory: number[] = [bestFitness];

    let iteration = 0;
    let converged = false;

    // Main DE loop
    for (iteration = 0; iteration < maxIterations; iteration++) {
      for (let i = 0; i < populationSize; i++) {
        // Select three random distinct individuals
        const indices = new Set<number>();
        indices.add(i);
        while (indices.size < 4) {
          indices.add(Math.floor(Math.random() * populationSize));
        }
        const [, r1, r2, r3] = Array.from(indices);

        // Mutation (DE/rand/1)
        const mutant = population[r1].map((x, d) =>
          x + F * (population[r2][d] - population[r3][d])
        );

        // Crossover
        const jRand = Math.floor(Math.random() * dimensions);
        const trial = population[i].map((x, d) =>
          (Math.random() < CR || d === jRand) ? mutant[d] : x
        );

        // Boundary handling
        for (let d = 0; d < dimensions; d++) {
          if (trial[d] < bounds.min[d] || trial[d] > bounds.max[d]) {
            trial[d] = bounds.min[d] + Math.random() * (bounds.max[d] - bounds.min[d]);
          }
        }

        // Selection
        const trialFitness = objective(trial);
        if (trialFitness <= fitness[i]) {
          population[i] = trial;
          fitness[i] = trialFitness;

          if (trialFitness < bestFitness) {
            bestFitness = trialFitness;
            bestPosition = [...trial];
          }
        }
      }

      convergenceHistory.push(bestFitness);

      // Check convergence
      if (convergenceHistory.length > 50) {
        const recent = convergenceHistory.slice(-50);
        const improvement = recent[0] - recent[recent.length - 1];
        if (Math.abs(improvement) < tolerance) {
          converged = true;
          break;
        }
      }
    }

    // Calculate diversity
    const meanPos = new Array(dimensions).fill(0);
    for (const ind of population) {
      for (let d = 0; d < dimensions; d++) {
        meanPos[d] += ind[d] / populationSize;
      }
    }
    let diversity = 0;
    for (const ind of population) {
      let dist = 0;
      for (let d = 0; d < dimensions; d++) {
        const normalized = (ind[d] - meanPos[d]) / (bounds.max[d] - bounds.min[d]);
        dist += normalized * normalized;
      }
      diversity += Math.sqrt(dist);
    }
    diversity /= populationSize;

    return {
      bestPosition,
      bestFitness,
      convergenceHistory,
      iterations: iteration + 1,
      swarmDiversity: diversity,
      converged
    };
  }

  /**
   * Simulated Annealing (bonus optimizer)
   * Good for avoiding local minima
   */
  static simulatedAnnealing(
    objective: ObjectiveFunction,
    config: {
      dimensions: number;
      maxIterations: number;
      bounds: { min: number[]; max: number[] };
      initialTemperature?: number;
      coolingRate?: number;
      stepSize?: number;
    }
  ): PSOResult {
    const {
      dimensions,
      maxIterations,
      bounds,
      initialTemperature = 1000,
      coolingRate = 0.995,
      stepSize = 0.1
    } = config;

    // Initialize
    let current = bounds.min.map((min, d) =>
      min + Math.random() * (bounds.max[d] - min)
    );
    let currentFitness = objective(current);
    let best = [...current];
    let bestFitness = currentFitness;
    let temperature = initialTemperature;

    const convergenceHistory: number[] = [bestFitness];

    // Main SA loop
    for (let iteration = 0; iteration < maxIterations; iteration++) {
      // Generate neighbor
      const neighbor = current.map((x, d) => {
        const range = bounds.max[d] - bounds.min[d];
        const delta = (Math.random() - 0.5) * 2 * stepSize * range * Math.sqrt(temperature / initialTemperature);
        let newX = x + delta;
        newX = Math.max(bounds.min[d], Math.min(bounds.max[d], newX));
        return newX;
      });

      const neighborFitness = objective(neighbor);
      const delta = neighborFitness - currentFitness;

      // Accept or reject
      if (delta < 0 || Math.random() < Math.exp(-delta / temperature)) {
        current = neighbor;
        currentFitness = neighborFitness;

        if (currentFitness < bestFitness) {
          best = [...current];
          bestFitness = currentFitness;
        }
      }

      // Cool down
      temperature *= coolingRate;

      convergenceHistory.push(bestFitness);
    }

    return {
      bestPosition: best,
      bestFitness,
      convergenceHistory,
      iterations: maxIterations,
      swarmDiversity: 0,  // Not applicable for SA
      converged: temperature < 1e-6
    };
  }

  // Helper methods

  private static calculateDiversity(swarm: Particle[], bounds: { min: number[]; max: number[] }): number {
    const dimensions = bounds.min.length;
    const n = swarm.length;

    // Calculate centroid
    const centroid = new Array(dimensions).fill(0);
    for (const particle of swarm) {
      for (let d = 0; d < dimensions; d++) {
        centroid[d] += particle.position[d] / n;
      }
    }

    // Calculate average distance from centroid (normalized)
    let diversity = 0;
    for (const particle of swarm) {
      let dist = 0;
      for (let d = 0; d < dimensions; d++) {
        const normalized = (particle.position[d] - centroid[d]) / (bounds.max[d] - bounds.min[d]);
        dist += normalized * normalized;
      }
      diversity += Math.sqrt(dist);
    }

    return diversity / n;
  }

  private static dominates(a: number[], b: number[]): boolean {
    let dominated = false;
    for (let i = 0; i < a.length; i++) {
      if (a[i] > b[i]) return false;
      if (a[i] < b[i]) dominated = true;
    }
    return dominated;
  }

  private static updateArchive(
    archive: { position: number[]; objectives: number[] }[],
    swarm: { position: number[]; objectives: number[] }[],
    maxSize: number
  ): void {
    // Add non-dominated solutions from swarm
    for (const particle of swarm) {
      let dominated = false;
      const toRemove: number[] = [];

      for (let i = 0; i < archive.length; i++) {
        if (this.dominates(archive[i].objectives, particle.objectives)) {
          dominated = true;
          break;
        }
        if (this.dominates(particle.objectives, archive[i].objectives)) {
          toRemove.push(i);
        }
      }

      if (!dominated) {
        // Remove dominated archive members
        for (let i = toRemove.length - 1; i >= 0; i--) {
          archive.splice(toRemove[i], 1);
        }
        archive.push({
          position: [...particle.position],
          objectives: [...particle.objectives]
        });
      }
    }

    // Trim archive using crowding distance
    while (archive.length > maxSize) {
      const crowding = this.calculateCrowdingDistances(archive);
      let minIdx = 0;
      for (let i = 1; i < crowding.length; i++) {
        if (crowding[i] < crowding[minIdx]) {
          minIdx = i;
        }
      }
      archive.splice(minIdx, 1);
    }
  }

  private static calculateCrowdingDistances(
    archive: { position: number[]; objectives: number[] }[]
  ): number[] {
    const n = archive.length;
    if (n === 0) return [];

    const numObjectives = archive[0].objectives.length;
    const distances = new Array(n).fill(0);

    for (let m = 0; m < numObjectives; m++) {
      // Sort by objective m
      const indices = Array.from({ length: n }, (_, i) => i);
      indices.sort((a, b) => archive[a].objectives[m] - archive[b].objectives[m]);

      // Boundary points get infinite distance
      distances[indices[0]] = Infinity;
      distances[indices[n - 1]] = Infinity;

      // Calculate crowding distance
      const range = archive[indices[n - 1]].objectives[m] - archive[indices[0]].objectives[m];
      if (range > 0) {
        for (let i = 1; i < n - 1; i++) {
          distances[indices[i]] +=
            (archive[indices[i + 1]].objectives[m] - archive[indices[i - 1]].objectives[m]) / range;
        }
      }
    }

    return distances;
  }

  private static selectLeader(
    archive: { position: number[]; objectives: number[] }[]
  ): { position: number[]; objectives: number[] } {
    if (archive.length === 0) {
      throw new Error('Archive is empty');
    }

    // Tournament selection based on crowding distance
    const crowding = this.calculateCrowdingDistances(archive);
    const tournamentSize = Math.min(3, archive.length);

    let bestIdx = Math.floor(Math.random() * archive.length);
    let bestCrowding = crowding[bestIdx];

    for (let i = 1; i < tournamentSize; i++) {
      const idx = Math.floor(Math.random() * archive.length);
      if (crowding[idx] > bestCrowding) {
        bestIdx = idx;
        bestCrowding = crowding[idx];
      }
    }

    return archive[bestIdx];
  }

  private static calculateHypervolume(
    archive: { position: number[]; objectives: number[] }[],
    bounds: { min: number[]; max: number[] }
  ): number {
    if (archive.length === 0) return 0;

    // Simple 2D hypervolume calculation
    // For higher dimensions, use Monte Carlo estimation
    const numObjectives = archive[0].objectives.length;

    if (numObjectives === 2) {
      // Sort by first objective
      const sorted = [...archive].sort((a, b) => a.objectives[0] - b.objectives[0]);

      // Reference point (worst case)
      const ref = [
        Math.max(...archive.map(a => a.objectives[0])) * 1.1,
        Math.max(...archive.map(a => a.objectives[1])) * 1.1
      ];

      let hypervolume = 0;
      let prevY = ref[1];

      for (const point of sorted) {
        if (point.objectives[1] < prevY) {
          hypervolume += (ref[0] - point.objectives[0]) * (prevY - point.objectives[1]);
          prevY = point.objectives[1];
        }
      }

      return hypervolume;
    }

    // Monte Carlo estimation for higher dimensions
    const samples = 10000;
    let inside = 0;

    const maxObjs = archive[0].objectives.map((_, i) =>
      Math.max(...archive.map(a => a.objectives[i]))
    );
    const minObjs = archive[0].objectives.map((_, i) =>
      Math.min(...archive.map(a => a.objectives[i]))
    );

    for (let s = 0; s < samples; s++) {
      const point = minObjs.map((min, i) =>
        min + Math.random() * (maxObjs[i] * 1.1 - min)
      );

      // Check if point is dominated by any archive member
      for (const member of archive) {
        if (this.dominates(member.objectives, point)) {
          inside++;
          break;
        }
      }
    }

    // Calculate volume of bounding box
    let boxVolume = 1;
    for (let i = 0; i < numObjectives; i++) {
      boxVolume *= maxObjs[i] * 1.1 - minObjs[i];
    }

    return (inside / samples) * boxVolume;
  }
}

// Physics-specific optimization problems

export class PhysicsOptimization {
  /**
   * Optimize trajectory for minimum fuel consumption
   */
  static optimizeTrajectory(
    initialState: { position: number[]; velocity: number[] },
    targetState: { position: number[]; velocity: number[] },
    constraints: {
      maxThrust: number;
      maxTime: number;
      fuelMass: number;
      exhaustVelocity: number;
    }
  ): PSOResult {
    // Parameterize trajectory as thrust angles and burn times
    const dimensions = 6;  // 3 thrust angles + 3 burn durations

    const objective = (x: number[]): number => {
      // x[0-2]: thrust angles (radians)
      // x[3-5]: burn durations (seconds)

      let fuel = 0;
      let pos = [...initialState.position];
      let vel = [...initialState.velocity];

      // Simple trajectory simulation
      for (let i = 0; i < 3; i++) {
        const burnTime = x[3 + i];
        const angle = x[i];

        // Thrust direction
        const thrustDir = [
          Math.cos(angle),
          Math.sin(angle),
          0
        ];

        // Fuel consumed
        fuel += constraints.maxThrust * burnTime / constraints.exhaustVelocity;

        // Velocity change (simplified)
        for (let d = 0; d < 3; d++) {
          vel[d] += thrustDir[d] * constraints.maxThrust * burnTime /
                    (constraints.fuelMass - fuel + 1);
        }
      }

      // Position error
      let posError = 0;
      let velError = 0;
      for (let d = 0; d < 3; d++) {
        posError += (pos[d] - targetState.position[d]) ** 2;
        velError += (vel[d] - targetState.velocity[d]) ** 2;
      }

      return fuel + 1000 * Math.sqrt(posError) + 100 * Math.sqrt(velError);
    };

    return ParticleSwarmOptimizer.optimize(objective, {
      dimensions,
      swarmSize: 50,
      maxIterations: 500,
      bounds: {
        min: [0, 0, 0, 0, 0, 0],
        max: [2 * Math.PI, 2 * Math.PI, 2 * Math.PI,
              constraints.maxTime / 3, constraints.maxTime / 3, constraints.maxTime / 3]
      }
    });
  }

  /**
   * Optimize heat exchanger design
   */
  static optimizeHeatExchanger(
    requirements: {
      heatDuty: number;        // W
      hotInletTemp: number;    // K
      coldInletTemp: number;   // K
      maxPressureDrop: number; // Pa
    }
  ): ConstrainedResult {
    // Design variables: tube diameter, length, number of tubes
    const objective = (x: number[]): number => {
      const diameter = x[0];
      const length = x[1];
      const numTubes = Math.round(x[2]);

      // Cost function (simplified): material + pumping
      const surfaceArea = Math.PI * diameter * length * numTubes;
      const materialCost = surfaceArea * 50;  // $/m²

      return materialCost;
    };

    const constraints: ConstraintFunction[] = [
      // Heat transfer constraint
      (x: number[]) => {
        const diameter = x[0];
        const length = x[1];
        const numTubes = Math.round(x[2]);
        const surfaceArea = Math.PI * diameter * length * numTubes;
        const U = 500;  // Overall heat transfer coefficient (W/m²K)
        const LMTD = (requirements.hotInletTemp - requirements.coldInletTemp) / 2;
        const Q = U * surfaceArea * LMTD;
        return requirements.heatDuty - Q;  // Should be <= 0
      },
      // Pressure drop constraint
      (x: number[]) => {
        const diameter = x[0];
        const length = x[1];
        // Simplified pressure drop
        const velocity = 2;  // m/s assumed
        const f = 0.02;  // Friction factor
        const rho = 1000;  // Water density
        const deltaP = f * (length / diameter) * 0.5 * rho * velocity ** 2;
        return deltaP - requirements.maxPressureDrop;
      }
    ];

    return ParticleSwarmOptimizer.optimizeConstrained(objective, constraints, {
      dimensions: 3,
      swarmSize: 30,
      maxIterations: 200,
      bounds: {
        min: [0.01, 1, 10],    // 1cm diameter, 1m length, 10 tubes
        max: [0.1, 10, 1000]   // 10cm diameter, 10m length, 1000 tubes
      }
    });
  }

  /**
   * Multi-objective lens design
   */
  static optimizeLens(
    requirements: {
      focalLength: number;     // mm
      aperture: number;        // mm
      wavelengths: number[];   // nm (for chromatic aberration)
    }
  ): MOPSOResult {
    // Design variables: curvatures, thickness, refractive indices
    const objectives = (x: number[]): number[] => {
      const r1 = x[0];  // Front curvature
      const r2 = x[1];  // Back curvature
      const t = x[2];   // Thickness
      const n = x[3];   // Refractive index

      // Lensmaker's equation
      const power = (n - 1) * (1/r1 - 1/r2 + (n-1)*t/(n*r1*r2));
      const focalLength = 1 / power;

      // Objective 1: Focal length error
      const focalError = Math.abs(focalLength - requirements.focalLength);

      // Objective 2: Spherical aberration (simplified)
      const sphericalAberration = Math.abs(r1 - r2) * requirements.aperture ** 2 / (r1 * r2);

      // Objective 3: Weight (proportional to volume)
      const weight = t * requirements.aperture ** 2;

      return [focalError, sphericalAberration, weight];
    };

    return ParticleSwarmOptimizer.optimizeMultiObjective(objectives, {
      dimensions: 4,
      swarmSize: 50,
      maxIterations: 300,
      bounds: {
        min: [10, -100, 1, 1.4],     // Curvatures, thickness, refractive index
        max: [100, -10, 20, 1.9]
      }
    });
  }

  /**
   * Optimize pendulum damping for critical damping
   */
  static optimizeDamping(
    mass: number,
    length: number,
    initialAngle: number
  ): PSOResult {
    const g = 9.81;
    const omega0 = Math.sqrt(g / length);  // Natural frequency
    const criticalDamping = 2 * mass * omega0;

    // Find damping coefficient that minimizes settling time
    const objective = (x: number[]): number => {
      const damping = x[0];
      const zeta = damping / criticalDamping;  // Damping ratio

      // Simulate pendulum
      let theta = initialAngle;
      let omega = 0;
      const dt = 0.01;
      let settlingTime = 0;
      const threshold = 0.01 * initialAngle;

      for (let t = 0; t < 100; t += dt) {
        // Equation of motion: θ'' + (c/m)θ' + (g/L)sin(θ) = 0
        const alpha = -(damping / mass) * omega - (g / length) * Math.sin(theta);
        omega += alpha * dt;
        theta += omega * dt;

        if (Math.abs(theta) > threshold || Math.abs(omega) > threshold) {
          settlingTime = t;
        }
      }

      // Penalize underdamping oscillations
      const overshootPenalty = zeta < 1 ? 10 * (1 - zeta) : 0;

      return settlingTime + overshootPenalty;
    };

    return ParticleSwarmOptimizer.optimize(objective, {
      dimensions: 1,
      swarmSize: 20,
      maxIterations: 100,
      bounds: {
        min: [0.1],
        max: [criticalDamping * 3]
      }
    });
  }
}
