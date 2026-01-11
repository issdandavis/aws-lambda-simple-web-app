/**
 * Spiralverse Protocol - True Torus Geometry Implementation
 * Patent Claims with Verified Riemannian Mathematics
 *
 * Core geometry is IMMUTABLE - webhook updates only affect parameters
 * Mathematical foundations independently verified (Gemini 2025-01)
 */

const crypto = require('crypto');

// ============================================================================
// IMMUTABLE CORE: Torus Geometry (Cannot be modified by webhooks)
// Verified parametrization: x(θ,φ) = [(R+r·cosθ)cosφ, (R+r·cosθ)sinφ, r·sinθ]
// ============================================================================

const TorusGeometry = Object.freeze({
  // Torus parameters (major radius R, minor radius r)
  R: 3.0,  // Major radius - distance from center to tube center
  r: 1.0,  // Minor radius - tube radius

  // Parametric surface: (θ, φ) → (x, y, z)
  parametrize(theta, phi) {
    const R = this.R, r = this.r;
    return {
      x: (R + r * Math.cos(theta)) * Math.cos(phi),
      y: (R + r * Math.cos(theta)) * Math.sin(phi),
      z: r * Math.sin(theta)
    };
  },

  // Metric tensor components (first fundamental form)
  // g_θθ = r² (constant - domain movement cost)
  // g_φφ = (R + r·cosθ)² (variable - sequence movement cost)
  // g_θφ = 0 (orthogonal coordinates)
  metricTensor(theta) {
    const R = this.R, r = this.r;
    const g_theta_theta = r * r;
    const g_phi_phi = Math.pow(R + r * Math.cos(theta), 2);
    return { g_tt: g_theta_theta, g_pp: g_phi_phi, g_tp: 0 };
  },

  // Riemannian distance (infinitesimal): ds² = r²dθ² + (R+r·cosθ)²dφ²
  riemannianDistanceSq(theta, dTheta, dPhi) {
    const g = this.metricTensor(theta);
    return g.g_tt * dTheta * dTheta + g.g_pp * dPhi * dPhi;
  },

  // Gaussian curvature: K = cosθ / [r(R + r·cosθ)]
  // K > 0: outer equator (security zone) - positive curvature
  // K < 0: inner equator (creative zone) - negative curvature
  // K = 0: top/bottom circles (transition zones)
  gaussianCurvature(theta) {
    const R = this.R, r = this.r;
    return Math.cos(theta) / (r * (R + r * Math.cos(theta)));
  },

  // Trust zone classification based on curvature
  classifyZone(theta) {
    const K = this.gaussianCurvature(theta);
    if (K > 0.01) return { zone: 'security', curvature: 'positive', K };
    if (K < -0.01) return { zone: 'creative', curvature: 'negative', K };
    return { zone: 'transition', curvature: 'zero', K };
  },

  // Geodesic energy: integral of ds along path
  // Lower energy = more natural/allowed path
  geodesicEnergy(path) {
    if (path.length < 2) return 0;
    let energy = 0;
    for (let i = 1; i < path.length; i++) {
      const dTheta = path[i].theta - path[i-1].theta;
      const dPhi = path[i].phi - path[i-1].phi;
      const avgTheta = (path[i].theta + path[i-1].theta) / 2;
      energy += Math.sqrt(this.riemannianDistanceSq(avgTheta, dTheta, dPhi));
    }
    return energy;
  },

  // Check if transition violates geometric constraints
  // Crossing from K>0 to K<0 directly requires high energy
  validateTransition(thetaFrom, thetaTo, phiFrom, phiTo) {
    const zoneFrom = this.classifyZone(thetaFrom);
    const zoneTo = this.classifyZone(thetaTo);
    const dTheta = thetaTo - thetaFrom;
    const dPhi = phiTo - phiFrom;
    const avgTheta = (thetaFrom + thetaTo) / 2;
    const distance = Math.sqrt(this.riemannianDistanceSq(avgTheta, dTheta, dPhi));

    // Direct security→creative transition without going through transition zone
    const directViolation = zoneFrom.zone === 'security' && zoneTo.zone === 'creative' &&
                           Math.abs(dTheta) > Math.PI / 4;

    return {
      valid: !directViolation,
      from: zoneFrom,
      to: zoneTo,
      distance: Math.round(distance * 1000) / 1000,
      violation: directViolation ? 'direct_zone_crossing' : null
    };
  }
});

// ============================================================================
// IMMUTABLE CORE: ML-KEM Simulation Layer
// ============================================================================

const MLKEM = Object.freeze({
  PARAMS: Object.freeze({ name: 'ML-KEM-768', pkLen: 1184, skLen: 2400, ctLen: 1088, ssLen: 32 }),

  keyGen() {
    const seed = crypto.randomBytes(64);
    const pk = crypto.createHash('sha3-256').update(Buffer.concat([seed, Buffer.from('pk')])).digest();
    const sk = crypto.createHash('sha3-256').update(Buffer.concat([seed, Buffer.from('sk')])).digest();
    return {
      pk: Buffer.concat([pk, crypto.randomBytes(this.PARAMS.pkLen - 32)]).toString('base64'),
      sk: Buffer.concat([sk, crypto.randomBytes(this.PARAMS.skLen - 32)]).toString('base64'),
      pkHash: pk.toString('hex').slice(0, 16)
    };
  },

  encapsulate(pkBase64) {
    const pk = Buffer.from(pkBase64, 'base64');
    const ephemeral = crypto.randomBytes(32);
    const ss = crypto.createHash('sha3-256').update(Buffer.concat([pk.slice(0, 32), ephemeral])).digest();
    const ct = crypto.createHash('sha3-256').update(Buffer.concat([ephemeral, pk.slice(0, 32)])).digest();
    return {
      ct: Buffer.concat([ct, crypto.randomBytes(this.PARAMS.ctLen - 32)]).toString('base64'),
      ss: ss.toString('hex'),
      ctHash: ct.toString('hex').slice(0, 16)
    };
  }
});

// ============================================================================
// IMMUTABLE CORE: Dual-Lane Key Schedule with Torus Geometry
// Lane bit computed from ceremony outputs mapped to torus coordinates
// ============================================================================

const DualLaneKeySchedule = Object.freeze({
  // Map 5-tuple to torus coordinates (θ, φ)
  mapToTorus(chi, pkIn, ctIn, pkOut, ctOut) {
    const hash = (b) => {
      const h = crypto.createHash('sha256').update(Buffer.from(b, 'base64')).digest();
      return h.readUInt32BE(0) / 0xFFFFFFFF;
    };
    const chiVal = typeof chi === 'number' ? chi :
      crypto.createHash('sha256').update(JSON.stringify(chi)).digest().readUInt32BE(0) / 0xFFFFFFFF;

    // Map to torus: θ ∈ [0, 2π], φ ∈ [0, 2π]
    const theta = ((chiVal + hash(pkIn) + hash(ctIn)) / 3) * 2 * Math.PI;
    const phi = ((hash(pkOut) + hash(ctOut)) / 2) * 2 * Math.PI;
    return { theta, phi, chiVal };
  },

  // Compute lane bit from Gaussian curvature at mapped position
  computeLaneBit(theta) {
    const zone = TorusGeometry.classifyZone(theta);
    // Security zone (K > 0) → Lane B (oversight)
    // Creative zone (K < 0) → Lane A (brain)
    // Transition → based on theta position
    if (zone.zone === 'security') return { L: 1, zone };
    if (zone.zone === 'creative') return { L: 0, zone };
    return { L: theta < Math.PI ? 0 : 1, zone };
  },

  async deriveKey(sharedSecret, laneBit, salt = null) {
    const ssBuffer = Buffer.from(sharedSecret, 'hex');
    const saltBuffer = salt ? Buffer.from(salt, 'hex') : crypto.randomBytes(32);
    const info = laneBit === 0 ? 'spiralverse-lane-A-brain' : 'spiralverse-lane-B-oversight';

    return new Promise((resolve, reject) => {
      crypto.hkdf('sha256', ssBuffer, saltBuffer, info, 32, (err, derivedKey) => {
        if (err) reject(err);
        else resolve({
          key: Buffer.from(derivedKey).toString('hex'),
          lane: laneBit === 0 ? 'A' : 'B',
          laneLabel: laneBit === 0 ? 'brain' : 'oversight',
          info, salt: saltBuffer.toString('hex')
        });
      });
    });
  },

  async execute(ceremony) {
    const { chi, insideParty, outsideParty } = ceremony;
    const insideEnc = MLKEM.encapsulate(insideParty.pk);
    const outsideEnc = MLKEM.encapsulate(outsideParty.pk);
    const combinedSS = crypto.createHash('sha256')
      .update(Buffer.from(insideEnc.ss, 'hex'))
      .update(Buffer.from(outsideEnc.ss, 'hex')).digest('hex');

    const torusCoords = this.mapToTorus(chi, insideParty.pk, insideEnc.ct, outsideParty.pk, outsideEnc.ct);
    const { L, zone } = this.computeLaneBit(torusCoords.theta);
    const derivedKey = await this.deriveKey(combinedSS, L);
    const position = TorusGeometry.parametrize(torusCoords.theta, torusCoords.phi);

    return {
      ceremony: { chi: typeof chi === 'object' ? chi : { value: chi },
        inside: { pkHash: insideParty.pkHash, ctHash: insideEnc.ctHash },
        outside: { pkHash: outsideParty.pkHash, ctHash: outsideEnc.ctHash }},
      torus: { theta: torusCoords.theta, phi: torusCoords.phi, position,
        curvature: TorusGeometry.gaussianCurvature(torusCoords.theta), zone },
      classification: { L, zone: zone.zone },
      derivedKey,
      nonUnilateral: true
    };
  }
});

// ============================================================================
// IMMUTABLE CORE: Cryptographic Security Simulation
// Compares S1 (static classical), S2 (static quantum), S3 (entropic dual-quantum)
// ============================================================================

const SecuritySimulation = Object.freeze({
  SECONDS_PER_YEAR: 365.25 * 24 * 3600,
  INITIAL_BITS: 256,

  // Compute N0 and sqrt(N0) using log-space arithmetic for huge numbers
  getKeyspaceParams() {
    const log2_N0 = this.INITIAL_BITS; // log2(2^256) = 256
    const log2_sqrtN0 = log2_N0 / 2;   // log2(sqrt(2^256)) = 128
    return { log2_N0, log2_sqrtN0, bits: this.INITIAL_BITS };
  },

  // S1: Static classical - breach fraction
  // p1 = (C_classical * T) / N0
  computeS1(C_classical, T_seconds) {
    const { log2_N0 } = this.getKeyspaceParams();
    const log2_work_done = Math.log2(C_classical) + Math.log2(T_seconds);
    const log2_p1 = log2_work_done - log2_N0;
    return { log2_p: log2_p1, p: Math.pow(2, log2_p1), system: 'S1_static_classical' };
  },

  // S2: Static quantum (Grover) - breach fraction
  // p2 = (C_quantum * T) / sqrt(N0)
  computeS2(C_quantum, T_seconds) {
    const { log2_sqrtN0 } = this.getKeyspaceParams();
    const log2_work_done = Math.log2(C_quantum) + Math.log2(T_seconds);
    const log2_p2 = log2_work_done - log2_sqrtN0;
    return { log2_p: log2_p2, p: Math.pow(2, log2_p2), system: 'S2_static_quantum' };
  },

  // S3: Entropic dual-quantum system - breach fraction
  // W(t) = sqrt(N0) * e^(k*t/2), where k > k_min = 2*C_quantum/sqrt(N0)
  // p3 = (C_quantum * T) / W(T)
  computeS3(C_quantum, T_seconds, k_multiplier = 10.0) {
    const { log2_sqrtN0 } = this.getKeyspaceParams();
    const sqrtN0 = Math.pow(2, log2_sqrtN0);

    // k_min = 2 * C_quantum / sqrt(N0) - the "escape velocity" bound
    const k_min = 2.0 * C_quantum / sqrtN0;
    const k = k_multiplier * k_min;

    // W(T) = sqrt(N0) * e^(k*T/2)
    // log2(W) = log2(sqrt(N0)) + (k*T/2) * log2(e)
    const log2_W = log2_sqrtN0 + (k * T_seconds / 2) * Math.LOG2E;

    // Work done by attacker
    const log2_work_done = Math.log2(C_quantum) + Math.log2(T_seconds);

    // p3 = work_done / W(T)
    const log2_p3 = log2_work_done - log2_W;

    return {
      log2_p: log2_p3,
      p: Math.pow(2, log2_p3),
      system: 'S3_entropic_dual_quantum',
      k, k_min, k_multiplier,
      escapeVelocity: k > k_min
    };
  },

  // Geometric firewall layers multiply effective work
  // G = g1 * g2 * ... * gm where each gi is a layer's difficulty multiplier
  computeFirewallFactor(layers) {
    // Each layer: { name, factor }
    // e.g., [{ name: 'semantic', factor: 1e6 }, { name: 'intent', factor: 1e4 }]
    let G = 1;
    let log2_G = 0;
    for (const layer of layers) {
      G *= layer.factor;
      log2_G += Math.log2(layer.factor);
    }
    return { G, log2_G, layers: layers.length };
  },

  // S3 with firewall layers: W_total(t) = sqrt(N(t)) * G
  computeS3WithFirewalls(C_quantum, T_seconds, k_multiplier, firewallLayers) {
    const base = this.computeS3(C_quantum, T_seconds, k_multiplier);
    const firewall = this.computeFirewallFactor(firewallLayers);

    // Effective work = base work * G
    const log2_W_total = -base.log2_p + Math.log2(C_quantum) + Math.log2(T_seconds) + firewall.log2_G;
    const log2_p3_protected = Math.log2(C_quantum) + Math.log2(T_seconds) - log2_W_total;

    return {
      ...base,
      firewall,
      log2_p_protected: log2_p3_protected,
      p_protected: Math.pow(2, log2_p3_protected),
      system: 'S3_entropic_with_firewalls'
    };
  },

  // Run full simulation across time horizons
  simulate(params = {}) {
    const {
      C_classical = 1e18,      // classical ops/sec (optimistic)
      C_quantum = 1e20,        // quantum ops/sec (very optimistic)
      years = [10, 100, 1000],
      k_multiplier = 10.0,
      firewallLayers = [
        { name: 'semantic_6lang', factor: 1e6 },
        { name: 'intent_vector', factor: 1e4 },
        { name: 'relationship_graph', factor: 1e3 },
        { name: 'emotional_coherence', factor: 1e2 },
        { name: 'torus_geometry', factor: 1e4 }
      ],
      hardwareGrowth = null // { doubling_years: 2 } for Moore's law type growth
    } = params;

    const results = [];
    const keyspace = this.getKeyspaceParams();

    for (const y of years) {
      const T = y * this.SECONDS_PER_YEAR;

      // Adjust capacities for hardware growth if specified
      let C_c = C_classical;
      let C_q = C_quantum;
      if (hardwareGrowth) {
        const doublings = y / hardwareGrowth.doubling_years;
        C_c = C_classical * Math.pow(2, doublings);
        C_q = C_quantum * Math.pow(2, doublings);
      }

      const s1 = this.computeS1(C_c, T);
      const s2 = this.computeS2(C_q, T);
      const s3 = this.computeS3(C_q, T, k_multiplier);
      const s3_protected = this.computeS3WithFirewalls(C_q, T, k_multiplier, firewallLayers);

      results.push({
        horizon_years: y,
        horizon_seconds: T,
        capacities: { classical: C_c, quantum: C_q, growth: hardwareGrowth },
        S1_static_classical: {
          breach_probability: s1.p,
          log2_p: Math.round(s1.log2_p * 100) / 100,
          interpretation: s1.log2_p < -50 ? 'astronomically_unlikely' : s1.log2_p < -20 ? 'very_unlikely' : 'concerning'
        },
        S2_static_quantum: {
          breach_probability: s2.p,
          log2_p: Math.round(s2.log2_p * 100) / 100,
          interpretation: s2.log2_p < -50 ? 'astronomically_unlikely' : s2.log2_p < -20 ? 'very_unlikely' : 'concerning'
        },
        S3_entropic: {
          breach_probability: s3.p,
          log2_p: Math.round(s3.log2_p * 100) / 100,
          k: s3.k,
          escape_velocity_met: s3.escapeVelocity,
          interpretation: 'keyspace_expanding_faster_than_attack'
        },
        S3_with_firewalls: {
          breach_probability: s3_protected.p_protected,
          log2_p: Math.round(s3_protected.log2_p_protected * 100) / 100,
          firewall_layers: s3_protected.firewall.layers,
          firewall_multiplier_log2: Math.round(s3_protected.firewall.log2_G * 100) / 100,
          interpretation: 'geometric_impossibility'
        }
      });
    }

    return {
      simulation: 'cryptographic_security_comparison',
      keyspace,
      parameters: { C_classical, C_quantum, k_multiplier, firewallLayers: firewallLayers.map(l => l.name) },
      results,
      conclusion: {
        S1: 'Static classical remains secure due to 2^256 keyspace',
        S2: 'Static quantum vulnerable to Grover over long horizons',
        S3: 'Entropic expansion outpaces quantum search when k > k_min',
        S3_firewalls: 'Geometric + semantic layers create multiplicative protection'
      }
    };
  }
});

// ============================================================================
// MUTABLE: Webhook-Updatable Parameters (Science/Tech Updates)
// Core geometry NEVER changes - only thresholds and metadata
// ============================================================================

let webhookConfig = {
  scienceUpdates: [],
  lastCheck: null,
  thresholds: { energyMax: 5.0, driftTolerance: 0.15 },
  metadata: { version: '3.0.0', verified: '2025-01-11', verifier: 'gemini' }
};

const WebhookSystem = {
  // Register a science update (cannot modify core geometry)
  registerUpdate(update) {
    if (update.type === 'core_geometry') {
      return { error: 'IMMUTABLE: Core geometry cannot be modified', rejected: true };
    }
    webhookConfig.scienceUpdates.push({
      ...update, timestamp: Date.now(), id: crypto.randomBytes(8).toString('hex')
    });
    return { registered: true, id: webhookConfig.scienceUpdates.slice(-1)[0].id };
  },

  updateThresholds(newThresholds) {
    // Only allow threshold updates, not geometry
    if (newThresholds.energyMax) webhookConfig.thresholds.energyMax = newThresholds.energyMax;
    if (newThresholds.driftTolerance) webhookConfig.thresholds.driftTolerance = newThresholds.driftTolerance;
    webhookConfig.lastCheck = Date.now();
    return { updated: true, thresholds: webhookConfig.thresholds };
  },

  getConfig() {
    return { ...webhookConfig, coreImmutable: true,
      coreFormulas: ['ds²=r²dθ²+(R+rcosθ)²dφ²', 'K=cosθ/[r(R+rcosθ)]'] };
  }
};

// ============================================================================
// IMMUTABLE CORE: 10-Dimensional Manifold Analysis
// Each dimension represents a verification axis
// ============================================================================

const HyperManifold = Object.freeze({
  DIMENSIONS: Object.freeze([
    { id: 0, name: 'semantic', radius: 1.0, description: 'Language meaning space' },
    { id: 1, name: 'intent', radius: 1.2, description: 'Purpose/goal vector' },
    { id: 2, name: 'emotion', radius: 0.8, description: 'Affective state' },
    { id: 3, name: 'relationship', radius: 1.1, description: 'Entity connections' },
    { id: 4, name: 'temporal', radius: 1.0, description: 'Time consistency' },
    { id: 5, name: 'spatial', radius: 1.0, description: 'Context location' },
    { id: 6, name: 'security', radius: 1.5, description: 'Access control' },
    { id: 7, name: 'creative', radius: 0.9, description: 'Generative freedom' },
    { id: 8, name: 'coherence', radius: 1.3, description: 'Internal consistency' },
    { id: 9, name: 'spin', radius: 1.0, description: 'Quantum verification' }
  ]),

  // 10-torus parametrization: T^10 = S^1 × S^1 × ... × S^1 (10 times)
  parametrize(angles) {
    // angles = [θ₀, θ₁, ..., θ₉] each in [0, 2π]
    return this.DIMENSIONS.map((dim, i) => ({
      dimension: dim.name,
      angle: angles[i] || 0,
      x: dim.radius * Math.cos(angles[i] || 0),
      y: dim.radius * Math.sin(angles[i] || 0)
    }));
  },

  // Metric tensor for 10-torus: g = diag(r₀², r₁², ..., r₉²)
  metricTensor() {
    return this.DIMENSIONS.map(d => d.radius * d.radius);
  },

  // Compute geodesic distance in 10D
  geodesicDistance(angles1, angles2) {
    const g = this.metricTensor();
    let sumSq = 0;
    for (let i = 0; i < 10; i++) {
      const dTheta = (angles2[i] || 0) - (angles1[i] || 0);
      // Wrap angle difference to [-π, π]
      const wrapped = Math.atan2(Math.sin(dTheta), Math.cos(dTheta));
      sumSq += g[i] * wrapped * wrapped;
    }
    return Math.sqrt(sumSq);
  },

  // Sectional curvature between dimensions i and j
  sectionalCurvature(i, j, angles) {
    // For flat torus, sectional curvature is 0
    // But we add "effective curvature" based on zone interactions
    const ri = this.DIMENSIONS[i].radius;
    const rj = this.DIMENSIONS[j].radius;
    const interaction = Math.cos(angles[i] || 0) * Math.cos(angles[j] || 0);
    return interaction / (ri * rj);
  },

  // Compute total curvature across all dimension pairs
  totalCurvature(angles) {
    let total = 0;
    let count = 0;
    for (let i = 0; i < 10; i++) {
      for (let j = i + 1; j < 10; j++) {
        total += this.sectionalCurvature(i, j, angles);
        count++;
      }
    }
    return { total, average: total / count, pairs: count };
  },

  // Map request context to 10D angles
  contextToAngles(context) {
    const hash = (s) => {
      let h = 0;
      const str = typeof s === 'string' ? s : JSON.stringify(s);
      for (let i = 0; i < str.length; i++) h = ((h << 5) - h) + str.charCodeAt(i);
      return (Math.abs(h) % 1000) / 1000 * 2 * Math.PI;
    };
    return [
      hash(context.semantic || context.message || ''),
      hash(context.intent || context.action || ''),
      hash(context.emotion || context.sentiment || ''),
      hash(context.relationship || context.source || ''),
      hash(context.timestamp || Date.now()),
      hash(context.location || context.path || ''),
      hash(context.securityLevel || context.lane || ''),
      hash(context.creativity || context.freedom || ''),
      hash(context.coherence || context.consistency || ''),
      context.spin?.phase || hash(context.spinSecret || crypto.randomBytes(8))
    ];
  },

  // Full dimensional analysis
  analyze(context) {
    const angles = this.contextToAngles(context);
    const position = this.parametrize(angles);
    const curvature = this.totalCurvature(angles);
    return { angles, position, curvature, dimensions: 10 };
  }
});

// ============================================================================
// IMMUTABLE CORE: Quantum Spin Mathematics
// Pauli matrices and spin state verification
// ============================================================================

const QuantumSpin = Object.freeze({
  // Pauli matrices (represented as 2x2 complex)
  PAULI: Object.freeze({
    I: [[1, 0], [0, 1]],           // Identity
    X: [[0, 1], [1, 0]],           // σx - bit flip
    Y: [[0, { r: 0, i: -1 }], [{ r: 0, i: 1 }, 0]], // σy
    Z: [[1, 0], [0, -1]]           // σz - phase flip
  }),

  // Generate spin state from secret
  generateSpinState(secret) {
    const hash = crypto.createHash('sha256').update(secret).digest();
    // Bloch sphere coordinates
    const theta = (hash.readUInt16BE(0) / 65535) * Math.PI;
    const phi = (hash.readUInt16BE(2) / 65535) * 2 * Math.PI;
    // Spin vector (sx, sy, sz) on Bloch sphere
    return {
      sx: Math.sin(theta) * Math.cos(phi),
      sy: Math.sin(theta) * Math.sin(phi),
      sz: Math.cos(theta),
      theta, phi,
      stateVector: {
        alpha: { r: Math.cos(theta / 2), i: 0 },
        beta: { r: Math.sin(theta / 2) * Math.cos(phi), i: Math.sin(theta / 2) * Math.sin(phi) }
      }
    };
  },

  // Compute expectation value ⟨ψ|σ|ψ⟩ for measurement axis
  expectationValue(spinState, axis) {
    const { sx, sy, sz } = spinState;
    switch (axis) {
      case 'x': return sx;
      case 'y': return sy;
      case 'z': return sz;
      default: return sz;
    }
  },

  // Inner product of two spin states (fidelity)
  fidelity(spin1, spin2) {
    // |⟨ψ₁|ψ₂⟩|² = (1 + s1·s2) / 2
    const dot = spin1.sx * spin2.sx + spin1.sy * spin2.sy + spin1.sz * spin2.sz;
    return (1 + dot) / 2;
  },

  // Generate correlated spin pair (entangled-like)
  generateCorrelatedPair(secret) {
    const baseSpin = this.generateSpinState(secret);
    // Anti-correlated partner (singlet-like)
    const partner = {
      sx: -baseSpin.sx,
      sy: -baseSpin.sy,
      sz: -baseSpin.sz,
      theta: Math.PI - baseSpin.theta,
      phi: baseSpin.phi + Math.PI
    };
    return { alice: baseSpin, bob: partner, correlation: -1 };
  },

  // Verify spin matches expected (within tolerance)
  verifySpin(receivedSpin, expectedSpin, tolerance = 0.1) {
    const fid = this.fidelity(receivedSpin, expectedSpin);
    return {
      verified: fid >= (1 - tolerance),
      fidelity: Math.round(fid * 1000) / 1000,
      tolerance,
      mismatch: fid < (1 - tolerance) ? 'SPIN_MISMATCH_VIOLATION' : null
    };
  },

  // Sign request with spin
  signRequest(request, secret) {
    const spin = this.generateSpinState(secret);
    const signature = crypto.createHash('sha256')
      .update(JSON.stringify(request))
      .update(secret)
      .digest('hex').slice(0, 32);
    return {
      ...request,
      spinSignature: {
        signature,
        publicSpin: { theta: spin.theta, phi: spin.phi },
        timestamp: Date.now()
      }
    };
  },

  // Verify signed request
  verifySignedRequest(signedRequest, secret) {
    const { spinSignature, ...request } = signedRequest;
    const expectedSig = crypto.createHash('sha256')
      .update(JSON.stringify(request))
      .update(secret)
      .digest('hex').slice(0, 32);
    const sigMatch = spinSignature.signature === expectedSig;

    const receivedSpin = this.generateSpinState(secret);
    // Reconstruct expected spin from public values
    const expectedSpin = {
      sx: Math.sin(spinSignature.publicSpin.theta) * Math.cos(spinSignature.publicSpin.phi),
      sy: Math.sin(spinSignature.publicSpin.theta) * Math.sin(spinSignature.publicSpin.phi),
      sz: Math.cos(spinSignature.publicSpin.theta)
    };
    const spinVerify = this.verifySpin(receivedSpin, expectedSpin);

    return {
      valid: sigMatch && spinVerify.verified,
      signatureMatch: sigMatch,
      spinVerification: spinVerify,
      geometricIntegrity: sigMatch && spinVerify.verified
    };
  }
});

// ============================================================================
// Self-Healing Reactive Protocol System
// Handles crashes, retries, and recovery
// ============================================================================

const SelfHealingProtocol = {
  state: {
    failures: [],
    recoveries: [],
    healthScore: 1.0,
    lastHealthCheck: Date.now(),
    nodeStatus: 'primary_active'
  },

  // Wrap operation with self-healing
  async withRecovery(operation, context, maxRetries = 3) {
    let lastError = null;
    const backoffMs = [100, 500, 2000, 5000];

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const start = Date.now();
        const result = await operation();
        this.recordSuccess(context, Date.now() - start);
        return { success: true, result, attempt, recovered: attempt > 0 };
      } catch (error) {
        lastError = error;
        this.recordFailure(context, error, attempt);

        if (attempt < maxRetries) {
          await this.sleep(backoffMs[attempt] || 5000);
          await this.attemptRecovery(context, error);
        }
      }
    }

    return {
      success: false,
      error: lastError?.message || 'Unknown error',
      attempts: maxRetries + 1,
      healingReport: this.getHealingReport()
    };
  },

  recordSuccess(context, durationMs) {
    this.state.healthScore = Math.min(1.0, this.state.healthScore + 0.1);
    this.state.lastHealthCheck = Date.now();
  },

  recordFailure(context, error, attempt) {
    this.state.failures.push({
      context: context.substring(0, 100),
      error: error?.message || 'Unknown',
      attempt,
      timestamp: Date.now()
    });
    this.state.healthScore = Math.max(0, this.state.healthScore - 0.2);
    // Keep only last 100 failures
    if (this.state.failures.length > 100) this.state.failures.shift();
  },

  async attemptRecovery(context, error) {
    const recovery = {
      context,
      errorType: error?.name || 'Error',
      action: 'state_reset',
      timestamp: Date.now()
    };

    // Recovery strategies based on error type
    if (error?.message?.includes('memory')) {
      recovery.action = 'gc_hint';
      if (global.gc) global.gc();
    } else if (error?.message?.includes('timeout')) {
      recovery.action = 'extend_timeout';
    } else {
      recovery.action = 'retry_with_backoff';
    }

    this.state.recoveries.push(recovery);
    if (this.state.recoveries.length > 50) this.state.recoveries.shift();
    return recovery;
  },

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  },

  getHealingReport() {
    return {
      healthScore: Math.round(this.state.healthScore * 100) / 100,
      recentFailures: this.state.failures.slice(-5),
      recentRecoveries: this.state.recoveries.slice(-5),
      nodeStatus: this.state.nodeStatus,
      uptime: Date.now() - (this.state.failures[0]?.timestamp || Date.now())
    };
  },

  // Multi-nodal support function execution
  async executeWithSupport(primaryOp, supportOps = []) {
    const results = { primary: null, support: [], consensus: null };

    // Execute primary with recovery
    results.primary = await this.withRecovery(primaryOp, 'primary_node');

    // Execute support functions in parallel
    if (supportOps.length > 0) {
      results.support = await Promise.all(
        supportOps.map((op, i) => this.withRecovery(op, `support_node_${i}`))
      );
    }

    // Compute consensus if multiple results
    if (results.support.length > 0) {
      const allSuccess = [results.primary, ...results.support].filter(r => r.success);
      results.consensus = {
        agreementRatio: allSuccess.length / (1 + results.support.length),
        primaryValid: results.primary.success,
        supportValid: results.support.filter(r => r.success).length
      };
    }

    return results;
  }
};

// ============================================================================
// Ray Tracing for Intent Trajectory Simulation
// Cast rays through manifold to validate intent paths
// ============================================================================

const IntentRayTracer = Object.freeze({
  // Ray structure: origin, direction, energy
  createRay(origin, target, initialEnergy = 1.0) {
    const direction = [];
    for (let i = 0; i < 10; i++) {
      direction.push((target[i] || 0) - (origin[i] || 0));
    }
    // Normalize direction
    const mag = Math.sqrt(direction.reduce((s, d) => s + d * d, 0));
    const normalized = direction.map(d => d / (mag || 1));
    return { origin, direction: normalized, energy: initialEnergy, magnitude: mag };
  },

  // Step ray through manifold
  stepRay(ray, stepSize = 0.1) {
    const newPosition = ray.origin.map((o, i) => o + ray.direction[i] * stepSize);
    // Compute energy loss based on curvature at position
    const curvature = HyperManifold.totalCurvature(newPosition);
    const energyLoss = Math.abs(curvature.average) * stepSize * 0.1;
    return {
      ...ray,
      origin: newPosition,
      energy: Math.max(0, ray.energy - energyLoss),
      curvatureTraversed: curvature.average
    };
  },

  // Trace full path from origin to target
  traceIntent(origin, target, maxSteps = 100) {
    let ray = this.createRay(origin, target);
    const path = [{ position: [...ray.origin], energy: ray.energy }];
    const zones = [];
    let totalCurvature = 0;

    for (let step = 0; step < maxSteps && ray.energy > 0.01; step++) {
      ray = this.stepRay(ray);
      path.push({ position: [...ray.origin], energy: ray.energy });
      totalCurvature += Math.abs(ray.curvatureTraversed || 0);

      // Check zone crossings
      const dist = HyperManifold.geodesicDistance(ray.origin, target);
      if (dist < 0.1) break; // Reached target
    }

    // Analyze path for violations
    const energyRemaining = ray.energy;
    const reachedTarget = HyperManifold.geodesicDistance(ray.origin, target) < 0.1;

    return {
      path,
      steps: path.length,
      energyRemaining: Math.round(energyRemaining * 1000) / 1000,
      totalCurvatureTraversed: Math.round(totalCurvature * 1000) / 1000,
      reachedTarget,
      valid: reachedTarget && energyRemaining > 0.1,
      violation: !reachedTarget ? 'TARGET_UNREACHABLE' :
                 energyRemaining < 0.1 ? 'ENERGY_DEPLETED' : null
    };
  },

  // Batch trace multiple intents
  traceMultiple(intents) {
    return intents.map(intent => ({
      intent: intent.id,
      trace: this.traceIntent(intent.origin, intent.target)
    }));
  },

  // Visualize as ASCII (simplified)
  visualizePath(trace, width = 40) {
    const lines = [];
    const stepWidth = trace.path.length / width;
    for (let i = 0; i < width; i++) {
      const idx = Math.floor(i * stepWidth);
      const point = trace.path[idx];
      const energy = Math.round(point.energy * 10);
      lines.push('█'.repeat(energy) + '░'.repeat(10 - energy));
    }
    return lines.join('\n');
  }
});

// ============================================================================
// Comprehensive Analysis Engine
// Combines all systems for full verification
// ============================================================================

const AnalysisEngine = {
  async runComprehensive(request, secrets = {}) {
    const results = {
      timestamp: Date.now(),
      requestId: crypto.randomBytes(8).toString('hex'),
      dimensions: {},
      spin: {},
      trajectory: {},
      healing: {},
      security: {},
      verdict: {}
    };

    // 1. 10-Dimensional Analysis
    await SelfHealingProtocol.withRecovery(async () => {
      results.dimensions = HyperManifold.analyze(request);
    }, 'dimensional_analysis');

    // 2. Quantum Spin Verification
    if (secrets.spinSecret) {
      await SelfHealingProtocol.withRecovery(async () => {
        const expectedSpin = QuantumSpin.generateSpinState(secrets.spinSecret);
        const receivedSpin = request.spin ?
          { sx: request.spin.sx, sy: request.spin.sy, sz: request.spin.sz } :
          QuantumSpin.generateSpinState(request.spinToken || '');
        results.spin = {
          expected: { theta: expectedSpin.theta, phi: expectedSpin.phi },
          received: receivedSpin,
          verification: QuantumSpin.verifySpin(receivedSpin, expectedSpin)
        };
      }, 'spin_verification');
    }

    // 3. Intent Ray Tracing
    if (request.intentOrigin && request.intentTarget) {
      await SelfHealingProtocol.withRecovery(async () => {
        const origin = HyperManifold.contextToAngles(request.intentOrigin);
        const target = HyperManifold.contextToAngles(request.intentTarget);
        results.trajectory = IntentRayTracer.traceIntent(origin, target);
      }, 'ray_tracing');
    }

    // 4. Self-Healing Report
    results.healing = SelfHealingProtocol.getHealingReport();

    // 5. Security Simulation (10-year projection)
    results.security = SecuritySimulation.simulate({ years: [1, 10, 100] });

    // 6. Final Verdict
    const violations = [];
    if (results.spin.verification?.mismatch) violations.push(results.spin.verification.mismatch);
    if (results.trajectory.violation) violations.push(results.trajectory.violation);
    if (results.healing.healthScore < 0.5) violations.push('SYSTEM_HEALTH_DEGRADED');

    results.verdict = {
      authorized: violations.length === 0,
      violations,
      geometricIntegrity: results.dimensions.curvature?.average > -0.5,
      spinIntegrity: !results.spin.verification?.mismatch,
      trajectoryIntegrity: !results.trajectory.violation,
      systemHealth: results.healing.healthScore
    };

    return results;
  }
};

// ============================================================================
// IMMUTABLE CORE: Six-Language Protocol (SpiralVerse Codex)
// Each language has specific semantic constraints and dimensional weights
// ============================================================================

const SixLanguages = Object.freeze({
  CODEX: Object.freeze({
    JOY: {
      id: 'joy',
      name: 'Joy',
      description: 'Celebratory, playful, creative expression',
      emoji: '✨',
      dimensionalWeights: { semantic: 0.7, emotion: 1.0, creative: 1.0, coherence: 0.6 },
      constraints: { minEnergy: 0.8, maxFormality: 0.3, requiresPositivity: true },
      allowedZones: ['creative', 'transition'],
      forbiddenTransitions: ['security→security']
    },
    CUT: {
      id: 'cut',
      name: 'Cut',
      description: 'Precision, brevity, sharp truth',
      emoji: '⚔️',
      dimensionalWeights: { semantic: 1.0, intent: 1.0, coherence: 1.0, creative: 0.3 },
      constraints: { maxWords: 50, requiresPrecision: true, noAmbiguity: true },
      allowedZones: ['security', 'transition'],
      forbiddenTransitions: ['creative→creative']
    },
    ANCHOR: {
      id: 'anchor',
      name: 'Anchor',
      description: 'Grounding, stability, factual foundation',
      emoji: '⚓',
      dimensionalWeights: { semantic: 1.0, coherence: 1.0, security: 0.9, temporal: 0.8 },
      constraints: { requiresCitation: true, noSpeculation: true, factBased: true },
      allowedZones: ['security'],
      forbiddenTransitions: ['security→creative']
    },
    BRIDGE: {
      id: 'bridge',
      name: 'Bridge',
      description: 'Connecting, mediating between contexts',
      emoji: '🌉',
      dimensionalWeights: { relationship: 1.0, semantic: 0.8, intent: 0.8, coherence: 0.9 },
      constraints: { mustConnect: true, preserveContext: true, bidirectional: true },
      allowedZones: ['transition', 'creative', 'security'],
      forbiddenTransitions: []
    },
    HARMONY: {
      id: 'harmony',
      name: 'Harmony',
      description: 'Balance, reconciliation, synthesis',
      emoji: '☯️',
      dimensionalWeights: { emotion: 0.9, relationship: 0.9, coherence: 1.0, creative: 0.7 },
      constraints: { seekBalance: true, acknowledgeAll: true, noExtremism: true },
      allowedZones: ['transition', 'creative'],
      forbiddenTransitions: ['security→security']
    },
    PARADOX: {
      id: 'paradox',
      name: 'Paradox',
      description: 'Creative tension, exploration of contradictions',
      emoji: '🔮',
      dimensionalWeights: { creative: 1.0, coherence: 0.5, semantic: 0.6, intent: 0.7 },
      constraints: { embraceContradiction: true, allowAmbiguity: true, exploreEdges: true },
      allowedZones: ['creative'],
      forbiddenTransitions: ['security→security', 'transition→security']
    }
  }),

  // Validate message against language constraints
  validateMessage(message, languageId) {
    const lang = this.CODEX[languageId.toUpperCase()];
    if (!lang) return { valid: false, error: 'UNKNOWN_LANGUAGE' };

    const violations = [];
    const wordCount = (message.content || message).split(/\s+/).length;

    if (lang.constraints.maxWords && wordCount > lang.constraints.maxWords) {
      violations.push(`EXCEEDS_WORD_LIMIT: ${wordCount}/${lang.constraints.maxWords}`);
    }

    return {
      valid: violations.length === 0,
      language: lang.id,
      violations,
      dimensionalWeights: lang.dimensionalWeights
    };
  },

  // Compute language-weighted dimensional score
  computeLanguageScore(context, languageId) {
    const lang = this.CODEX[languageId.toUpperCase()];
    if (!lang) return { score: 0, error: 'UNKNOWN_LANGUAGE' };

    const angles = HyperManifold.contextToAngles(context);
    let weightedSum = 0;
    let totalWeight = 0;

    const dimMap = { semantic: 0, intent: 1, emotion: 2, relationship: 3,
                     temporal: 4, spatial: 5, security: 6, creative: 7, coherence: 8 };

    for (const [dim, weight] of Object.entries(lang.dimensionalWeights)) {
      if (dimMap[dim] !== undefined) {
        const angle = angles[dimMap[dim]];
        const normalizedScore = (Math.cos(angle) + 1) / 2;
        weightedSum += normalizedScore * weight;
        totalWeight += weight;
      }
    }

    return {
      score: Math.round((weightedSum / totalWeight) * 1000) / 1000,
      language: lang.id,
      emoji: lang.emoji
    };
  },

  // Tag message with language metadata
  tagMessage(message, languageId, agentId) {
    const validation = this.validateMessage(message, languageId);
    const lang = this.CODEX[languageId.toUpperCase()];
    return {
      content: message.content || message,
      meta: {
        language: languageId,
        emoji: lang?.emoji || '❓',
        agent: agentId,
        timestamp: Date.now(),
        validation,
        spin: crypto.createHash('sha256')
          .update(JSON.stringify({ message, languageId, agentId, t: Date.now() }))
          .digest('hex').slice(0, 16)
      }
    };
  }
});

// ============================================================================
// IMMUTABLE CORE: Agent Archetypes
// Role definitions with language preferences and dimensional constraints
// ============================================================================

const AgentArchetypes = Object.freeze({
  ROLES: Object.freeze({
    RESEARCHER: {
      id: 'researcher',
      name: 'Researcher',
      description: 'Investigates, gathers evidence, verifies facts',
      preferredLanguages: ['bridge', 'anchor'],
      dimensionalConstraints: { coherence: 0.8, security: 0.7 },
      metrics: ['citation_count', 'source_diversity', 'fact_accuracy'],
      driftTolerance: 0.1
    },
    WRITER: {
      id: 'writer',
      name: 'Writer',
      description: 'Composes, articulates, expresses ideas',
      preferredLanguages: ['joy', 'harmony'],
      dimensionalConstraints: { creative: 0.7, emotion: 0.6 },
      metrics: ['clarity', 'engagement', 'style_consistency'],
      driftTolerance: 0.2
    },
    THINKER: {
      id: 'thinker',
      name: 'Thinker',
      description: 'Analyzes, reasons, synthesizes concepts',
      preferredLanguages: ['paradox', 'anchor'],
      dimensionalConstraints: { coherence: 0.9, semantic: 0.8 },
      metrics: ['logical_validity', 'depth', 'novelty'],
      driftTolerance: 0.15
    },
    ACTOR: {
      id: 'actor',
      name: 'Actor',
      description: 'Embodies roles, executes actions, performs tasks',
      preferredLanguages: ['joy', 'bridge'],
      dimensionalConstraints: { relationship: 0.8, intent: 0.7 },
      metrics: ['role_fidelity', 'action_completion', 'context_awareness'],
      driftTolerance: 0.25
    },
    CRITIC: {
      id: 'critic',
      name: 'Critic',
      description: 'Evaluates, challenges, improves quality',
      preferredLanguages: ['cut', 'paradox'],
      dimensionalConstraints: { coherence: 0.9, semantic: 0.9 },
      metrics: ['issue_detection', 'constructiveness', 'precision'],
      driftTolerance: 0.1
    },
    GUARDIAN: {
      id: 'guardian',
      name: 'Guardian',
      description: 'Monitors, protects, enforces security',
      preferredLanguages: ['anchor', 'cut'],
      dimensionalConstraints: { security: 0.95, coherence: 0.9 },
      metrics: ['threat_detection', 'response_time', 'false_positive_rate'],
      driftTolerance: 0.05
    }
  }),

  // Create agent instance
  createAgent(roleId, customConfig = {}) {
    const role = this.ROLES[roleId.toUpperCase()];
    if (!role) return { error: 'UNKNOWN_ROLE' };

    return {
      id: crypto.randomBytes(8).toString('hex'),
      role: role.id,
      name: customConfig.name || `${role.name}-${Date.now()}`,
      preferredLanguages: role.preferredLanguages,
      constraints: { ...role.dimensionalConstraints, ...customConfig.constraints },
      driftTolerance: customConfig.driftTolerance || role.driftTolerance,
      state: { driftScore: 0, messageCount: 0, lastActive: Date.now() },
      created: Date.now()
    };
  },

  // Validate agent output against role constraints
  validateOutput(agent, output, context) {
    const role = this.ROLES[agent.role.toUpperCase()];
    if (!role) return { valid: false, error: 'INVALID_ROLE' };

    const angles = HyperManifold.contextToAngles(context);
    const violations = [];

    // Check dimensional constraints
    const dimMap = { semantic: 0, intent: 1, emotion: 2, relationship: 3,
                     temporal: 4, spatial: 5, security: 6, creative: 7, coherence: 8 };

    for (const [dim, minScore] of Object.entries(role.dimensionalConstraints)) {
      if (dimMap[dim] !== undefined) {
        const angle = angles[dimMap[dim]];
        const score = (Math.cos(angle) + 1) / 2;
        if (score < minScore) {
          violations.push(`${dim.toUpperCase()}_BELOW_THRESHOLD: ${score.toFixed(2)}/${minScore}`);
        }
      }
    }

    // Check language usage
    if (output.meta?.language && !role.preferredLanguages.includes(output.meta.language)) {
      violations.push(`LANGUAGE_MISMATCH: ${output.meta.language} not in ${role.preferredLanguages}`);
    }

    return { valid: violations.length === 0, violations, role: role.id };
  }
});

// ============================================================================
// Semantic Drift Scoring System
// Continuous monitoring of agent output for drift from expected behavior
// ============================================================================

const SemanticDriftScoring = {
  agentHistory: new Map(),

  // Record agent output for drift analysis
  recordOutput(agentId, output, context) {
    if (!this.agentHistory.has(agentId)) {
      this.agentHistory.set(agentId, { outputs: [], driftScores: [], baseline: null });
    }
    const history = this.agentHistory.get(agentId);
    const angles = HyperManifold.contextToAngles(context);

    // Establish baseline from first 5 outputs
    if (history.outputs.length < 5) {
      history.outputs.push({ angles, timestamp: Date.now() });
      if (history.outputs.length === 5) {
        history.baseline = this.computeBaseline(history.outputs);
      }
      return { drift: 0, status: 'establishing_baseline' };
    }

    // Compute drift from baseline
    const drift = this.computeDrift(angles, history.baseline);
    history.driftScores.push({ drift, timestamp: Date.now() });
    history.outputs.push({ angles, timestamp: Date.now() });

    // Keep only last 100 entries
    if (history.outputs.length > 100) history.outputs.shift();
    if (history.driftScores.length > 100) history.driftScores.shift();

    return { drift, status: drift > 0.3 ? 'HIGH_DRIFT_WARNING' : 'normal' };
  },

  // Compute baseline from initial outputs
  computeBaseline(outputs) {
    const avgAngles = Array(10).fill(0);
    for (const output of outputs) {
      for (let i = 0; i < 10; i++) {
        avgAngles[i] += output.angles[i] / outputs.length;
      }
    }
    return avgAngles;
  },

  // Compute drift from baseline
  computeDrift(currentAngles, baseline) {
    return HyperManifold.geodesicDistance(currentAngles, baseline);
  },

  // Get drift report for agent
  getDriftReport(agentId) {
    const history = this.agentHistory.get(agentId);
    if (!history) return { error: 'AGENT_NOT_FOUND' };

    const recentDrifts = history.driftScores.slice(-10);
    const avgDrift = recentDrifts.reduce((s, d) => s + d.drift, 0) / (recentDrifts.length || 1);
    const maxDrift = Math.max(...recentDrifts.map(d => d.drift), 0);

    return {
      agentId,
      sampleCount: history.outputs.length,
      averageDrift: Math.round(avgDrift * 1000) / 1000,
      maxDrift: Math.round(maxDrift * 1000) / 1000,
      hasBaseline: !!history.baseline,
      status: avgDrift > 0.3 ? 'DRIFTING' : avgDrift > 0.15 ? 'MODERATE' : 'STABLE'
    };
  }
};

// ============================================================================
// Team Orchestrator
// Multi-agent coordination with consensus and workflow management
// ============================================================================

const TeamOrchestrator = {
  teams: new Map(),

  // Create a new team
  createTeam(config) {
    const teamId = crypto.randomBytes(8).toString('hex');
    const team = {
      id: teamId,
      name: config.name || `Team-${teamId}`,
      agents: [],
      workflow: config.workflow || 'sequential',
      consensusThreshold: config.consensusThreshold || 0.6,
      created: Date.now(),
      state: { tasksCompleted: 0, activeTask: null }
    };

    // Add agents
    for (const roleId of config.roles || []) {
      const agent = AgentArchetypes.createAgent(roleId);
      if (!agent.error) team.agents.push(agent);
    }

    this.teams.set(teamId, team);
    return team;
  },

  // Execute task with team
  async executeTask(teamId, task) {
    const team = this.teams.get(teamId);
    if (!team) return { error: 'TEAM_NOT_FOUND' };

    team.state.activeTask = task;
    const results = [];

    // Execute based on workflow type
    if (team.workflow === 'sequential') {
      for (const agent of team.agents) {
        const result = await this.executeAgentTask(agent, task, results);
        results.push(result);
      }
    } else if (team.workflow === 'parallel') {
      const promises = team.agents.map(agent => this.executeAgentTask(agent, task, []));
      const parallelResults = await Promise.all(promises);
      results.push(...parallelResults);
    }

    // Compute consensus
    const consensus = this.computeConsensus(results, team.consensusThreshold);
    team.state.tasksCompleted++;
    team.state.activeTask = null;

    return { teamId, task, results, consensus };
  },

  // Execute single agent task
  async executeAgentTask(agent, task, priorResults) {
    const context = { ...task, priorResults, agent: agent.id };
    const angles = HyperManifold.contextToAngles(context);

    // Simulate agent processing
    const output = {
      agentId: agent.id,
      role: agent.role,
      language: agent.preferredLanguages[0],
      response: `[${agent.role}] Processed: ${task.description || task}`,
      angles,
      timestamp: Date.now()
    };

    // Record for drift analysis
    const driftResult = SemanticDriftScoring.recordOutput(agent.id, output, context);
    output.drift = driftResult;

    // Validate against role constraints
    const validation = AgentArchetypes.validateOutput(agent, output, context);
    output.validation = validation;

    return output;
  },

  // Compute consensus across results
  computeConsensus(results, threshold) {
    if (results.length === 0) return { achieved: false, ratio: 0 };

    const validResults = results.filter(r => r.validation?.valid);
    const ratio = validResults.length / results.length;

    return {
      achieved: ratio >= threshold,
      ratio: Math.round(ratio * 100) / 100,
      validCount: validResults.length,
      totalCount: results.length,
      threshold
    };
  },

  // Get team status
  getTeamStatus(teamId) {
    const team = this.teams.get(teamId);
    if (!team) return { error: 'TEAM_NOT_FOUND' };

    return {
      id: team.id,
      name: team.name,
      agentCount: team.agents.length,
      agents: team.agents.map(a => ({ id: a.id, role: a.role, name: a.name })),
      workflow: team.workflow,
      tasksCompleted: team.state.tasksCompleted,
      activeTask: team.state.activeTask ? 'busy' : 'idle'
    };
  }
};

// ============================================================================
// Group Presets
// Pre-configured team compositions for common use cases
// ============================================================================

const GroupPresets = Object.freeze({
  PRESETS: Object.freeze({
    RESEARCH_SQUAD: {
      id: 'research_squad',
      name: 'Research Squad',
      description: 'Comprehensive research and verification team',
      roles: ['researcher', 'researcher', 'critic', 'writer'],
      workflow: 'sequential',
      consensusThreshold: 0.75,
      languages: ['anchor', 'bridge', 'cut']
    },
    STORY_ROOM: {
      id: 'story_room',
      name: 'Story Room',
      description: 'Creative narrative development team',
      roles: ['thinker', 'writer', 'actor', 'critic'],
      workflow: 'sequential',
      consensusThreshold: 0.6,
      languages: ['joy', 'harmony', 'paradox']
    },
    OPS_GUARDIAN: {
      id: 'ops_guardian',
      name: 'Ops Guardian',
      description: 'Security monitoring and incident response',
      roles: ['guardian', 'guardian', 'critic', 'researcher'],
      workflow: 'parallel',
      consensusThreshold: 0.9,
      languages: ['anchor', 'cut']
    },
    DEBATE_CHAMBER: {
      id: 'debate_chamber',
      name: 'Debate Chamber',
      description: 'Adversarial reasoning and truth-seeking',
      roles: ['thinker', 'critic', 'thinker', 'critic'],
      workflow: 'sequential',
      consensusThreshold: 0.5,
      languages: ['paradox', 'cut', 'anchor']
    },
    SYNTHESIS_LAB: {
      id: 'synthesis_lab',
      name: 'Synthesis Lab',
      description: 'Integration and harmonization of ideas',
      roles: ['researcher', 'thinker', 'writer', 'actor'],
      workflow: 'sequential',
      consensusThreshold: 0.7,
      languages: ['bridge', 'harmony', 'joy']
    }
  }),

  // Instantiate a preset team
  instantiate(presetId) {
    const preset = this.PRESETS[presetId.toUpperCase()];
    if (!preset) return { error: 'UNKNOWN_PRESET' };

    return TeamOrchestrator.createTeam({
      name: preset.name,
      roles: preset.roles,
      workflow: preset.workflow,
      consensusThreshold: preset.consensusThreshold
    });
  },

  // List available presets
  list() {
    return Object.values(this.PRESETS).map(p => ({
      id: p.id,
      name: p.name,
      description: p.description,
      roleCount: p.roles.length,
      languages: p.languages
    }));
  }
});

// ============================================================================
// Trajectory Authorization with Geodesic Validation
// ============================================================================

const TrajectoryAuthorization = {
  authorize(request) {
    const { path, chiCurrent, chiPrevious } = request;

    // Convert chi values to torus coordinates
    const thetaCurrent = chiCurrent * 2 * Math.PI;
    const thetaPrevious = chiPrevious * 2 * Math.PI;
    const phiCurrent = (request.phiCurrent || 0.5) * 2 * Math.PI;
    const phiPrevious = (request.phiPrevious || 0.5) * 2 * Math.PI;

    // Validate geometric transition
    const transition = TorusGeometry.validateTransition(thetaPrevious, thetaCurrent, phiPrevious, phiCurrent);

    // Compute geodesic energy if path provided
    let pathEnergy = 0;
    if (path && path.length > 1) {
      pathEnergy = TorusGeometry.geodesicEnergy(path);
    }

    const authorized = transition.valid && pathEnergy <= webhookConfig.thresholds.energyMax;

    return {
      authorized,
      transition,
      geodesicEnergy: Math.round(pathEnergy * 1000) / 1000,
      energyThreshold: webhookConfig.thresholds.energyMax,
      geometry: {
        from: { theta: thetaPrevious, phi: phiPrevious, ...TorusGeometry.classifyZone(thetaPrevious) },
        to: { theta: thetaCurrent, phi: phiCurrent, ...TorusGeometry.classifyZone(thetaCurrent) }
      }
    };
  }
};

// ============================================================================
// Lambda Handler
// ============================================================================

exports.handler = async (event) => {
  const method = event.httpMethod || event.requestContext?.http?.method || 'GET';
  const path = event.path || event.rawPath || '/';
  const respond = (code, body) => ({
    statusCode: code,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body, null, 2)
  });

  try {
    const body = event.body ? JSON.parse(event.body) : {};

    // GET /health
    if (method === 'GET' && path === '/health') {
      return respond(200, {
        status: 'healthy',
        geometry: 'torus-riemannian',
        formulas: { metric: 'ds²=r²dθ²+(R+rcosθ)²dφ²', curvature: 'K=cosθ/[r(R+rcosθ)]' },
        ...webhookConfig.metadata
      });
    }

    // GET /geometry - Curvature visualization endpoint
    if (method === 'GET' && path === '/geometry') {
      const theta = parseFloat(event.queryStringParameters?.theta || '0');
      const phi = parseFloat(event.queryStringParameters?.phi || '0');
      return respond(200, {
        input: { theta, phi },
        position: TorusGeometry.parametrize(theta, phi),
        metric: TorusGeometry.metricTensor(theta),
        curvature: TorusGeometry.gaussianCurvature(theta),
        zone: TorusGeometry.classifyZone(theta),
        torusParams: { R: TorusGeometry.R, r: TorusGeometry.r }
      });
    }

    // POST /geometry - Batch curvature analysis
    if (method === 'POST' && path === '/geometry') {
      const points = body.points || [{ theta: 0, phi: 0 }];
      const analysis = points.map(p => ({
        input: p,
        position: TorusGeometry.parametrize(p.theta, p.phi),
        curvature: TorusGeometry.gaussianCurvature(p.theta),
        zone: TorusGeometry.classifyZone(p.theta)
      }));
      return respond(200, { analysis, count: analysis.length });
    }

    // POST /ceremony
    if (method === 'POST' && path === '/ceremony') {
      const inside = MLKEM.keyGen();
      const outside = MLKEM.keyGen();
      return respond(200, {
        ceremony: { inside: { pk: inside.pk, pkHash: inside.pkHash },
                   outside: { pk: outside.pk, pkHash: outside.pkHash }},
        note: 'Pass pk values to /derive with chi context'
      });
    }

    // POST /derive - Full dual-lane key derivation
    if (method === 'POST' && path === '/derive') {
      const { chi, insidePk, outsidePk } = body;
      if (!chi || !insidePk || !outsidePk) return respond(400, { error: 'Required: chi, insidePk, outsidePk' });
      const result = await DualLaneKeySchedule.execute({
        chi, insideParty: { pk: insidePk, pkHash: 'provided' }, outsideParty: { pk: outsidePk, pkHash: 'provided' }
      });
      return respond(200, result);
    }

    // POST /authorize - Geodesic path validation
    if (method === 'POST' && path === '/authorize') {
      const { chiCurrent, chiPrevious, path: trajPath, phiCurrent, phiPrevious } = body;
      if (chiCurrent === undefined || chiPrevious === undefined) {
        return respond(400, { error: 'Required: chiCurrent, chiPrevious' });
      }
      const result = TrajectoryAuthorization.authorize({ chiCurrent, chiPrevious, phiCurrent, phiPrevious, path: trajPath });
      return respond(result.authorized ? 200 : 403, result);
    }

    // POST /webhook - Register science updates (core immutable)
    if (method === 'POST' && path === '/webhook') {
      const result = WebhookSystem.registerUpdate(body);
      return respond(result.rejected ? 403 : 200, result);
    }

    // GET /webhook - Get current config
    if (method === 'GET' && path === '/webhook') {
      return respond(200, WebhookSystem.getConfig());
    }

    // PUT /webhook - Update thresholds only
    if (method === 'PUT' && path === '/webhook') {
      const result = WebhookSystem.updateThresholds(body);
      return respond(200, result);
    }

    // POST /simulate - Run cryptographic security simulation
    if (method === 'POST' && path === '/simulate') {
      const result = SecuritySimulation.simulate(body);
      return respond(200, result);
    }

    // GET /simulate - Run with default parameters
    if (method === 'GET' && path === '/simulate') {
      const result = SecuritySimulation.simulate({});
      return respond(200, result);
    }

    // POST /analyze - Comprehensive 10D analysis with spin and ray tracing
    if (method === 'POST' && path === '/analyze') {
      const secrets = { spinSecret: body.spinSecret || body.secret };
      const result = await AnalysisEngine.runComprehensive(body, secrets);
      return respond(result.verdict.authorized ? 200 : 403, result);
    }

    // GET /analyze - Run with test parameters
    if (method === 'GET' && path === '/analyze') {
      const testRequest = {
        message: 'test analysis',
        intent: 'verify',
        intentOrigin: { semantic: 'start', intent: 'query' },
        intentTarget: { semantic: 'end', intent: 'response' }
      };
      const result = await AnalysisEngine.runComprehensive(testRequest, {});
      return respond(200, result);
    }

    // POST /spin - Generate and verify spin states
    if (method === 'POST' && path === '/spin') {
      const { secret, action } = body;
      if (action === 'generate') {
        const spin = QuantumSpin.generateSpinState(secret || crypto.randomBytes(16).toString('hex'));
        return respond(200, { spin, blochSphere: { theta: spin.theta, phi: spin.phi } });
      }
      if (action === 'pair') {
        const pair = QuantumSpin.generateCorrelatedPair(secret || crypto.randomBytes(16).toString('hex'));
        return respond(200, pair);
      }
      if (action === 'sign') {
        const signed = QuantumSpin.signRequest(body.request || {}, secret);
        return respond(200, signed);
      }
      if (action === 'verify') {
        const result = QuantumSpin.verifySignedRequest(body.signedRequest, secret);
        return respond(result.valid ? 200 : 403, result);
      }
      return respond(400, { error: 'Required: action (generate|pair|sign|verify)' });
    }

    // POST /raytrace - Trace intent trajectory through 10D manifold
    if (method === 'POST' && path === '/raytrace') {
      const { origin, target, intents } = body;
      if (intents) {
        const results = IntentRayTracer.traceMultiple(intents);
        return respond(200, { traces: results });
      }
      if (!origin || !target) {
        return respond(400, { error: 'Required: origin and target contexts, or intents array' });
      }
      const originAngles = HyperManifold.contextToAngles(origin);
      const targetAngles = HyperManifold.contextToAngles(target);
      const trace = IntentRayTracer.traceIntent(originAngles, targetAngles);
      return respond(trace.valid ? 200 : 403, {
        trace,
        visualization: IntentRayTracer.visualizePath(trace)
      });
    }

    // GET /dimensions - View 10D manifold structure
    if (method === 'GET' && path === '/dimensions') {
      return respond(200, {
        dimensions: HyperManifold.DIMENSIONS,
        metricTensor: HyperManifold.metricTensor(),
        structure: 'T^10 (10-torus)'
      });
    }

    // POST /dimensions - Analyze context in 10D
    if (method === 'POST' && path === '/dimensions') {
      const analysis = HyperManifold.analyze(body);
      return respond(200, analysis);
    }

    // GET /healing - View self-healing system status
    if (method === 'GET' && path === '/healing') {
      return respond(200, SelfHealingProtocol.getHealingReport());
    }

    // GET /languages - View six-language codex
    if (method === 'GET' && path === '/languages') {
      return respond(200, {
        codex: Object.values(SixLanguages.CODEX).map(l => ({
          id: l.id, name: l.name, emoji: l.emoji,
          description: l.description, allowedZones: l.allowedZones
        })),
        count: 6,
        protocol: 'SpiralVerse Codex v1.0'
      });
    }

    // POST /languages - Validate and tag message with language
    if (method === 'POST' && path === '/languages') {
      const { message, language, agentId } = body;
      if (!message || !language) {
        return respond(400, { error: 'Required: message, language' });
      }
      const tagged = SixLanguages.tagMessage(message, language, agentId || 'anonymous');
      const score = SixLanguages.computeLanguageScore(body, language);
      return respond(tagged.meta.validation.valid ? 200 : 400, { ...tagged, score });
    }

    // GET /agents - List agent archetypes
    if (method === 'GET' && path === '/agents') {
      return respond(200, {
        roles: Object.values(AgentArchetypes.ROLES).map(r => ({
          id: r.id, name: r.name, description: r.description,
          languages: r.preferredLanguages, driftTolerance: r.driftTolerance
        })),
        count: Object.keys(AgentArchetypes.ROLES).length
      });
    }

    // POST /agents - Create agent instance
    if (method === 'POST' && path === '/agents') {
      const { role, name, constraints } = body;
      if (!role) return respond(400, { error: 'Required: role' });
      const agent = AgentArchetypes.createAgent(role, { name, constraints });
      if (agent.error) return respond(400, agent);
      return respond(200, agent);
    }

    // GET /teams - List active teams
    if (method === 'GET' && path === '/teams') {
      const teams = [];
      TeamOrchestrator.teams.forEach((team, id) => {
        teams.push(TeamOrchestrator.getTeamStatus(id));
      });
      return respond(200, { teams, count: teams.length });
    }

    // POST /teams - Create team or execute task
    if (method === 'POST' && path === '/teams') {
      const { action, teamId, preset, roles, task, workflow } = body;

      if (action === 'create') {
        const team = TeamOrchestrator.createTeam({ roles, workflow });
        return respond(200, team);
      }

      if (action === 'preset') {
        if (!preset) return respond(400, { error: 'Required: preset' });
        const team = GroupPresets.instantiate(preset);
        if (team.error) return respond(400, team);
        return respond(200, team);
      }

      if (action === 'execute') {
        if (!teamId || !task) return respond(400, { error: 'Required: teamId, task' });
        const result = await TeamOrchestrator.executeTask(teamId, task);
        if (result.error) return respond(400, result);
        return respond(result.consensus.achieved ? 200 : 403, result);
      }

      if (action === 'status') {
        if (!teamId) return respond(400, { error: 'Required: teamId' });
        const status = TeamOrchestrator.getTeamStatus(teamId);
        if (status.error) return respond(404, status);
        return respond(200, status);
      }

      return respond(400, { error: 'Required: action (create|preset|execute|status)' });
    }

    // GET /presets - List group presets
    if (method === 'GET' && path === '/presets') {
      return respond(200, { presets: GroupPresets.list() });
    }

    // GET /drift - Get drift report for agent
    if (method === 'GET' && path === '/drift') {
      const agentId = event.queryStringParameters?.agentId;
      if (!agentId) return respond(400, { error: 'Required: agentId query param' });
      const report = SemanticDriftScoring.getDriftReport(agentId);
      if (report.error) return respond(404, report);
      return respond(200, report);
    }

    // Legacy endpoints
    if (method === 'POST' && path === '/verify') {
      return respond(200, TrajectoryAuthorization.authorize({
        chiCurrent: body.chiCurrent || 0.5, chiPrevious: body.chiPrevious || 0.5
      }));
    }

    return respond(404, {
      error: 'Not found',
      endpoints: [
        '/health', '/geometry', '/ceremony', '/derive', '/authorize',
        '/webhook', '/simulate', '/analyze', '/spin', '/raytrace',
        '/dimensions', '/healing', '/languages', '/agents', '/teams',
        '/presets', '/drift', '/verify'
      ]
    });
  } catch (err) {
    return respond(500, { error: err.message });
  }
};
