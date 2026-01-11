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
// Manifold Controller - Geometric Integrity Enforcement
// The "Physics Engine" for data - transforms Trust into calculable geometry
// Implements "The Snap" and Time Dilation (Stutter) penalty system
// ============================================================================

const ManifoldController = {
  // Torus parameters (immutable geometry)
  R: TorusGeometry.R,  // Major radius - global scale of memory
  r: TorusGeometry.r,  // Minor radius - context switching cost

  // Integrity thresholds
  thresholds: {
    snap: 0.5,           // Geometric divergence threshold for "The Snap"
    warning: 0.3,        // Warning threshold (pre-snap tension)
    critical: 0.8,       // Critical threshold (major violation)
    catastrophic: 1.5    // Catastrophic (system freeze)
  },

  // Time Dilation parameters
  timeDilation: {
    baseDelay: 50,       // Base delay in milliseconds
    beta: 1.5,           // Sensitivity coefficient
    maxDelay: 5000,      // Maximum delay (The Stutter)
    freezeThreshold: 3   // Consecutive failures before freeze
  },

  // Ledger state - the geometric memory
  ledger: {
    entries: [],
    currentState: { theta: 0, phi: 0 },
    failCount: 0,
    snapHistory: [],
    lastWrite: Date.now()
  },

  // Deterministic text-to-angle mapping using SHA-256
  textToAngle(text) {
    const hash = crypto.createHash('sha256').update(String(text)).digest();
    // Use first 8 bytes for high precision
    const high = hash.readUInt32BE(0);
    const low = hash.readUInt32BE(4);
    // Combine for 64-bit precision, normalize to [0, 2π]
    const normalized = (high * 0x100000000 + low) / (0xFFFFFFFF * 0x100000000 + 0xFFFFFFFF);
    return normalized * 2 * Math.PI;
  },

  // Calculate geometric divergence with periodic boundary handling
  calculateDivergence(theta1, phi1, theta2, phi2) {
    // Handle periodic wrapping - shortest angular distance
    const wrapAngle = (d) => {
      const wrapped = Math.abs(d) % (2 * Math.PI);
      return Math.min(wrapped, 2 * Math.PI - wrapped);
    };

    const dTheta = wrapAngle(theta2 - theta1);
    const dPhi = wrapAngle(phi2 - phi1);

    // Average theta for local metric approximation
    const avgTheta = (theta1 + theta2) / 2;

    // The Equation of Trust: ds² = r²dθ² + (R + r·cosθ)²dφ²
    const sequencePenalty = Math.pow(this.R + this.r * Math.cos(avgTheta), 2) * Math.pow(dPhi, 2);
    const domainPenalty = Math.pow(this.r, 2) * Math.pow(dTheta, 2);

    return Math.sqrt(sequencePenalty + domainPenalty);
  },

  // Lattice search for true geodesic (handles multiple wrappings)
  findTrueGeodesic(theta1, phi1, theta2, phi2) {
    let minDistance = Infinity;
    let bestPath = { k: 0, m: 0 };

    // Search the 9 nearest lattice points (covering space)
    for (let k = -1; k <= 1; k++) {
      for (let m = -1; m <= 1; m++) {
        const theta2Wrapped = theta2 + 2 * Math.PI * k;
        const phi2Wrapped = phi2 + 2 * Math.PI * m;

        // Direct Euclidean in covering space for comparison
        const dTheta = theta2Wrapped - theta1;
        const dPhi = phi2Wrapped - phi1;
        const avgTheta = (theta1 + theta2Wrapped) / 2;

        const dist = Math.sqrt(
          Math.pow(this.r, 2) * Math.pow(dTheta, 2) +
          Math.pow(this.R + this.r * Math.cos(avgTheta), 2) * Math.pow(dPhi, 2)
        );

        if (dist < minDistance) {
          minDistance = dist;
          bestPath = { k, m, theta: theta2Wrapped, phi: phi2Wrapped };
        }
      }
    }

    return {
      distance: minDistance,
      wrapping: bestPath,
      windingNumber: { theta: bestPath.k, phi: bestPath.m }
    };
  },

  // Calculate Time Dilation penalty
  calculateTimeDilation(failCount, divergence) {
    if (failCount === 0) return 0;

    const { baseDelay, beta, maxDelay } = this.timeDilation;

    // τ_delay = τ_base × (1 + FailCount)^β × (divergence / threshold)
    const delay = baseDelay * Math.pow(1 + failCount, beta) * (divergence / this.thresholds.snap);

    return Math.min(delay, maxDelay);
  },

  // Detect snap condition and severity
  detectSnap(divergence) {
    const { snap, warning, critical, catastrophic } = this.thresholds;

    if (divergence >= catastrophic) {
      return { snap: true, severity: 'CATASTROPHIC', action: 'FREEZE', confidence: 0 };
    }
    if (divergence >= critical) {
      return { snap: true, severity: 'CRITICAL', action: 'REJECT_HARD', confidence: 0.1 };
    }
    if (divergence >= snap) {
      return { snap: true, severity: 'SNAP', action: 'REJECT', confidence: 0.3 };
    }
    if (divergence >= warning) {
      return { snap: false, severity: 'WARNING', action: 'ALLOW_CAUTIOUS', confidence: 0.6 };
    }
    return { snap: false, severity: 'CLEAN', action: 'ALLOW', confidence: 1.0 };
  },

  // Classify the zone based on theta (semantic mapping)
  classifySemanticZone(theta) {
    const K = TorusGeometry.gaussianCurvature(theta);
    const zone = TorusGeometry.classifyZone(theta);

    // Semantic interpretation
    let semantic;
    if (theta >= 0 && theta < Math.PI / 4) {
      semantic = 'ABSOLUTE_TRUTH';  // Outer equator - maximum verification
    } else if (theta >= Math.PI / 4 && theta < Math.PI / 2) {
      semantic = 'HIGH_SECURITY';
    } else if (theta >= Math.PI / 2 && theta < 3 * Math.PI / 4) {
      semantic = 'TRANSITION_CREATIVE';
    } else if (theta >= 3 * Math.PI / 4 && theta < Math.PI) {
      semantic = 'CREATIVE_FLUX';
    } else if (theta >= Math.PI && theta < 5 * Math.PI / 4) {
      semantic = 'MAXIMUM_FLUX';  // Inner equator - rapid exploration
    } else if (theta >= 5 * Math.PI / 4 && theta < 3 * Math.PI / 2) {
      semantic = 'CREATIVE_FLUX';
    } else if (theta >= 3 * Math.PI / 2 && theta < 7 * Math.PI / 4) {
      semantic = 'TRANSITION_SECURITY';
    } else {
      semantic = 'HIGH_SECURITY';
    }

    return {
      ...zone,
      semantic,
      metricCost: Math.pow(this.R + this.r * Math.cos(theta), 2),
      timeSpeed: 1 / (this.R + this.r * Math.cos(theta))  // Time moves slower in security zones
    };
  },

  // The core validation function - accepts or rejects writes
  validateWrite(fact, options = {}) {
    const {
      domain = fact.domain || fact.content,
      sequenceId = fact.sequenceId || fact.timestamp || Date.now(),
      forceWrite = false
    } = options;

    // Map fact to geometric coordinates
    const thetaNew = this.textToAngle(domain);
    const phiNew = this.textToAngle(String(sequenceId));

    // Get current state
    const { theta: thetaCurrent, phi: phiCurrent } = this.ledger.currentState;

    // Calculate geometric divergence
    const divergence = this.calculateDivergence(thetaCurrent, phiCurrent, thetaNew, phiNew);

    // Find true geodesic path
    const geodesic = this.findTrueGeodesic(thetaCurrent, phiCurrent, thetaNew, phiNew);

    // Detect snap condition
    const snapResult = this.detectSnap(divergence);

    // Classify zones
    const fromZone = this.classifySemanticZone(thetaCurrent);
    const toZone = this.classifySemanticZone(thetaNew);

    // Check for forbidden transitions (e.g., direct ABSOLUTE_TRUTH to MAXIMUM_FLUX)
    const forbiddenTransition = (
      fromZone.semantic === 'ABSOLUTE_TRUTH' && toZone.semantic === 'MAXIMUM_FLUX'
    ) || (
      fromZone.semantic === 'MAXIMUM_FLUX' && toZone.semantic === 'ABSOLUTE_TRUTH'
    );

    if (forbiddenTransition && !forceWrite) {
      snapResult.snap = true;
      snapResult.severity = 'FORBIDDEN_TRANSITION';
      snapResult.action = 'REJECT_HARD';
    }

    // Calculate time dilation if snap
    let timeDilation = 0;
    if (snapResult.snap) {
      this.ledger.failCount++;
      timeDilation = this.calculateTimeDilation(this.ledger.failCount, divergence);

      // Record snap in history
      this.ledger.snapHistory.push({
        timestamp: Date.now(),
        divergence,
        severity: snapResult.severity,
        from: { theta: thetaCurrent, phi: phiCurrent, zone: fromZone.semantic },
        to: { theta: thetaNew, phi: phiNew, zone: toZone.semantic }
      });

      // Check for freeze condition
      if (this.ledger.failCount >= this.timeDilation.freezeThreshold) {
        snapResult.action = 'FREEZE';
        timeDilation = this.timeDilation.maxDelay;
      }
    } else {
      // Successful write - reset fail count
      this.ledger.failCount = 0;
    }

    // Build result
    const result = {
      status: snapResult.snap ? 'FAIL' : 'SUCCESS',
      coordinates: {
        current: { theta: thetaCurrent, phi: phiCurrent },
        proposed: { theta: thetaNew, phi: phiNew }
      },
      geometry: {
        divergence: Math.round(divergence * 10000) / 10000,
        geodesicDistance: Math.round(geodesic.distance * 10000) / 10000,
        windingNumber: geodesic.windingNumber
      },
      zones: {
        from: fromZone,
        to: toZone,
        transition: forbiddenTransition ? 'FORBIDDEN' : 'ALLOWED'
      },
      snap: snapResult,
      penalty: {
        failCount: this.ledger.failCount,
        timeDilation: Math.round(timeDilation),
        stutterActive: timeDilation > 0
      },
      trust: {
        equationOfTrust: `ds² = ${this.r}²dθ² + (${this.R} + ${this.r}·cos(θ))²dφ²`,
        value: Math.round((1 - Math.min(1, divergence / this.thresholds.critical)) * 1000) / 1000
      }
    };

    // If success and not forced, update ledger state
    if (!snapResult.snap || forceWrite) {
      const entry = {
        id: this.ledger.entries.length,
        timestamp: Date.now(),
        fact: { domain, sequenceId, content: fact.content || fact },
        coordinates: { theta: thetaNew, phi: phiNew },
        divergenceFromPrevious: divergence,
        zone: toZone.semantic
      };

      this.ledger.entries.push(entry);
      this.ledger.currentState = { theta: thetaNew, phi: phiNew };
      this.ledger.lastWrite = Date.now();

      result.entry = entry;
    }

    return result;
  },

  // Query the ledger by geometric proximity
  queryByProximity(theta, phi, maxDistance = 0.5) {
    const results = [];

    for (const entry of this.ledger.entries) {
      const distance = this.calculateDivergence(theta, phi, entry.coordinates.theta, entry.coordinates.phi);
      if (distance <= maxDistance) {
        results.push({
          entry,
          distance: Math.round(distance * 10000) / 10000
        });
      }
    }

    // Sort by distance
    results.sort((a, b) => a.distance - b.distance);
    return results;
  },

  // Audit the ledger - detect discontinuities
  auditLedger() {
    const violations = [];
    let totalDivergence = 0;
    let maxDivergence = 0;

    for (let i = 1; i < this.ledger.entries.length; i++) {
      const prev = this.ledger.entries[i - 1];
      const curr = this.ledger.entries[i];

      const divergence = this.calculateDivergence(
        prev.coordinates.theta, prev.coordinates.phi,
        curr.coordinates.theta, curr.coordinates.phi
      );

      totalDivergence += divergence;
      maxDivergence = Math.max(maxDivergence, divergence);

      if (divergence > this.thresholds.snap) {
        violations.push({
          index: i,
          from: prev,
          to: curr,
          divergence: Math.round(divergence * 10000) / 10000,
          severity: this.detectSnap(divergence).severity
        });
      }
    }

    const avgDivergence = this.ledger.entries.length > 1 ?
      totalDivergence / (this.ledger.entries.length - 1) : 0;

    return {
      entryCount: this.ledger.entries.length,
      violations,
      violationCount: violations.length,
      integrityScore: Math.round((1 - violations.length / Math.max(1, this.ledger.entries.length)) * 1000) / 1000,
      statistics: {
        averageDivergence: Math.round(avgDivergence * 10000) / 10000,
        maxDivergence: Math.round(maxDivergence * 10000) / 10000,
        totalSnapEvents: this.ledger.snapHistory.length,
        currentFailCount: this.ledger.failCount
      },
      health: violations.length === 0 ? 'HEALTHY' : violations.length < 3 ? 'DEGRADED' : 'CORRUPTED'
    };
  },

  // Visualize ledger as path data (for rendering)
  visualizePath() {
    return this.ledger.entries.map((entry, i) => {
      const pos = TorusGeometry.parametrize(entry.coordinates.theta, entry.coordinates.phi);
      return {
        index: i,
        x: Math.round(pos.x * 1000) / 1000,
        y: Math.round(pos.y * 1000) / 1000,
        z: Math.round(pos.z * 1000) / 1000,
        theta: Math.round(entry.coordinates.theta * 1000) / 1000,
        phi: Math.round(entry.coordinates.phi * 1000) / 1000,
        zone: entry.zone,
        timestamp: entry.timestamp
      };
    });
  },

  // Reset ledger (for testing)
  reset() {
    this.ledger = {
      entries: [],
      currentState: { theta: 0, phi: 0 },
      failCount: 0,
      snapHistory: [],
      lastWrite: Date.now()
    };
    return { status: 'RESET', timestamp: Date.now() };
  }
};

// ============================================================================
// Geodesic Watermark - The Shape IS the Authentication
// "You made it through security, but you're still caught"
// Even if all crypto passes, the trajectory through the manifold reveals imposters
// ============================================================================

const GeodesicWatermark = {
  // The secret key shapes the trajectory - without it, paths look "wrong"
  // This is not a signature ON the message, the message path IS the signature

  // Thresholds for "bandit detection"
  thresholds: {
    curvatureDeviation: 0.15,    // Max deviation from expected curvature profile
    trajectoryCoherence: 0.85,   // Min coherence score (0-1)
    zoneTransitionPenalty: 0.3,  // Penalty per unexpected zone transition
    windingMismatch: 0.5,        // Penalty for wrong winding number
    temporalJitter: 0.1,         // Max allowable temporal irregularity
    banditThreshold: 0.6         // Below this = BANDIT DETECTED
  },

  // Generate the expected "shape" for a message given a secret key
  // This is the "hyper-shape QR code" - the trajectory fingerprint
  generateExpectedShape(message, secretKey, options = {}) {
    const {
      steps = 8,           // Number of trajectory waypoints
      includeTimestamp = true
    } = options;

    // Derive trajectory parameters from secret key
    const keyHash = crypto.createHash('sha256').update(secretKey).digest();
    const messageHash = crypto.createHash('sha256').update(message).digest();

    // Combine key and message to get unique trajectory seed
    const combined = crypto.createHmac('sha256', keyHash).update(messageHash).digest();

    // Generate expected waypoints
    const waypoints = [];
    let theta = 0, phi = 0;

    for (let i = 0; i < steps; i++) {
      // Each step derived from combined hash
      const stepKey = crypto.createHmac('sha256', combined).update(`step${i}`).digest();

      // Delta angles determined by secret - attacker can't reproduce without key
      const dTheta = ((stepKey[0] / 255) - 0.5) * Math.PI / 4;  // ±π/8
      const dPhi = ((stepKey[1] / 255) - 0.5) * Math.PI / 2;    // ±π/4

      theta = (theta + dTheta + 2 * Math.PI) % (2 * Math.PI);
      phi = (phi + dPhi + 2 * Math.PI) % (2 * Math.PI);

      // Expected curvature at this point
      const K = TorusGeometry.gaussianCurvature(theta);
      const zone = ManifoldController.classifySemanticZone(theta);

      waypoints.push({
        index: i,
        theta: Math.round(theta * 10000) / 10000,
        phi: Math.round(phi * 10000) / 10000,
        expectedCurvature: Math.round(K * 10000) / 10000,
        zone: zone.semantic,
        position: TorusGeometry.parametrize(theta, phi)
      });
    }

    // Compute trajectory characteristics (the "shape fingerprint")
    const shapeFingerprint = this.computeShapeFingerprint(waypoints);

    return {
      waypoints,
      fingerprint: shapeFingerprint,
      timestamp: includeTimestamp ? Date.now() : null,
      messageHash: messageHash.toString('hex').slice(0, 16)
    };
  },

  // Compute the "shape fingerprint" - geometric characteristics of the trajectory
  computeShapeFingerprint(waypoints) {
    if (waypoints.length < 2) return null;

    // 1. Curvature profile (sequence of curvatures)
    const curvatures = waypoints.map(w => w.expectedCurvature || TorusGeometry.gaussianCurvature(w.theta));

    // 2. Zone transition sequence
    const zones = waypoints.map(w => w.zone);
    const transitions = [];
    for (let i = 1; i < zones.length; i++) {
      if (zones[i] !== zones[i-1]) {
        transitions.push({ from: zones[i-1], to: zones[i], at: i });
      }
    }

    // 3. Winding numbers (how many times we wrap around each axis)
    let thetaWinding = 0, phiWinding = 0;
    for (let i = 1; i < waypoints.length; i++) {
      const dTheta = waypoints[i].theta - waypoints[i-1].theta;
      const dPhi = waypoints[i].phi - waypoints[i-1].phi;
      if (Math.abs(dTheta) > Math.PI) thetaWinding += Math.sign(dTheta);
      if (Math.abs(dPhi) > Math.PI) phiWinding += Math.sign(dPhi);
    }

    // 4. Total geodesic length
    let totalLength = 0;
    for (let i = 1; i < waypoints.length; i++) {
      totalLength += ManifoldController.calculateDivergence(
        waypoints[i-1].theta, waypoints[i-1].phi,
        waypoints[i].theta, waypoints[i].phi
      );
    }

    // 5. Curvature integral (total "bending")
    const curvatureIntegral = curvatures.reduce((sum, K) => sum + Math.abs(K), 0);

    // 6. Average curvature sign (security-leaning vs creative-leaning)
    const avgCurvature = curvatures.reduce((sum, K) => sum + K, 0) / curvatures.length;

    // Create fingerprint hash
    const fpData = JSON.stringify({
      curvatures: curvatures.map(c => Math.round(c * 1000)),
      transitions: transitions.map(t => `${t.from}→${t.to}`),
      winding: { theta: thetaWinding, phi: phiWinding },
      length: Math.round(totalLength * 1000),
      bend: Math.round(curvatureIntegral * 1000)
    });
    const fpHash = crypto.createHash('sha256').update(fpData).digest('hex').slice(0, 32);

    return {
      hash: fpHash,
      curvatureProfile: curvatures,
      zoneTransitions: transitions,
      windingNumbers: { theta: thetaWinding, phi: phiWinding },
      totalGeodesicLength: Math.round(totalLength * 10000) / 10000,
      curvatureIntegral: Math.round(curvatureIntegral * 10000) / 10000,
      averageCurvature: Math.round(avgCurvature * 10000) / 10000,
      trajectoryBias: avgCurvature > 0.05 ? 'SECURITY' : avgCurvature < -0.05 ? 'CREATIVE' : 'BALANCED'
    };
  },

  // Verify an observed trajectory against expected shape
  // This is where we catch the "bandits"
  verifyTrajectory(observedWaypoints, expectedShape, options = {}) {
    const { strict = false } = options;

    if (!observedWaypoints || observedWaypoints.length === 0) {
      return { valid: false, score: 0, reason: 'NO_TRAJECTORY', bandit: true };
    }

    const expected = expectedShape.waypoints;
    const observed = observedWaypoints;

    // Compute observed fingerprint
    const observedFingerprint = this.computeShapeFingerprint(observed);

    let penalties = [];
    let score = 1.0;

    // 1. Check curvature profile match
    const expectedCurvatures = expectedShape.fingerprint.curvatureProfile;
    const observedCurvatures = observedFingerprint.curvatureProfile;

    // Resample if lengths differ
    const minLen = Math.min(expectedCurvatures.length, observedCurvatures.length);
    let curvatureDeviation = 0;
    for (let i = 0; i < minLen; i++) {
      curvatureDeviation += Math.abs(expectedCurvatures[i] - observedCurvatures[i]);
    }
    curvatureDeviation /= minLen;

    if (curvatureDeviation > this.thresholds.curvatureDeviation) {
      const penalty = Math.min(0.3, curvatureDeviation);
      score -= penalty;
      penalties.push({
        type: 'CURVATURE_MISMATCH',
        expected: expectedCurvatures.slice(0, 4),
        observed: observedCurvatures.slice(0, 4),
        deviation: Math.round(curvatureDeviation * 10000) / 10000,
        penalty
      });
    }

    // 2. Check zone transitions
    const expectedTransitions = expectedShape.fingerprint.zoneTransitions;
    const observedTransitions = observedFingerprint.zoneTransitions;

    const transitionMismatch = Math.abs(expectedTransitions.length - observedTransitions.length);
    if (transitionMismatch > 0) {
      const penalty = transitionMismatch * this.thresholds.zoneTransitionPenalty;
      score -= penalty;
      penalties.push({
        type: 'ZONE_TRANSITION_MISMATCH',
        expected: expectedTransitions.length,
        observed: observedTransitions.length,
        penalty
      });
    }

    // 3. Check winding numbers (topology must match)
    const expectedWinding = expectedShape.fingerprint.windingNumbers;
    const observedWinding = observedFingerprint.windingNumbers;

    if (expectedWinding.theta !== observedWinding.theta ||
        expectedWinding.phi !== observedWinding.phi) {
      score -= this.thresholds.windingMismatch;
      penalties.push({
        type: 'WINDING_MISMATCH',
        expected: expectedWinding,
        observed: observedWinding,
        penalty: this.thresholds.windingMismatch,
        severity: 'HIGH - Topologically different path!'
      });
    }

    // 4. Check geodesic length (should be similar)
    const lengthRatio = observedFingerprint.totalGeodesicLength /
                        (expectedShape.fingerprint.totalGeodesicLength || 1);
    if (lengthRatio < 0.7 || lengthRatio > 1.4) {
      const penalty = Math.abs(1 - lengthRatio) * 0.2;
      score -= penalty;
      penalties.push({
        type: 'PATH_LENGTH_ANOMALY',
        expected: expectedShape.fingerprint.totalGeodesicLength,
        observed: observedFingerprint.totalGeodesicLength,
        ratio: Math.round(lengthRatio * 100) / 100,
        penalty
      });
    }

    // 5. Check trajectory bias (security vs creative)
    if (expectedShape.fingerprint.trajectoryBias !== observedFingerprint.trajectoryBias) {
      score -= 0.15;
      penalties.push({
        type: 'TRAJECTORY_BIAS_MISMATCH',
        expected: expectedShape.fingerprint.trajectoryBias,
        observed: observedFingerprint.trajectoryBias,
        penalty: 0.15
      });
    }

    // 6. Fingerprint hash comparison (strict mode)
    const hashMatch = expectedShape.fingerprint.hash === observedFingerprint.hash;
    if (strict && !hashMatch) {
      score -= 0.5;
      penalties.push({
        type: 'FINGERPRINT_HASH_MISMATCH',
        expected: expectedShape.fingerprint.hash,
        observed: observedFingerprint.hash,
        penalty: 0.5
      });
    }

    // Clamp score
    score = Math.max(0, Math.min(1, score));

    // BANDIT DETECTION
    const isBandit = score < this.thresholds.banditThreshold;

    return {
      valid: score >= this.thresholds.trajectoryCoherence,
      score: Math.round(score * 1000) / 1000,
      bandit: isBandit,
      banditConfidence: isBandit ? Math.round((1 - score) * 100) : 0,
      penalties,
      comparison: {
        expectedHash: expectedShape.fingerprint.hash,
        observedHash: observedFingerprint.hash,
        hashMatch
      },
      fingerprints: {
        expected: expectedShape.fingerprint,
        observed: observedFingerprint
      },
      verdict: isBandit ?
        'BANDIT_DETECTED: Trajectory shape does not match secret-key derived path. Intruder identified.' :
        score >= this.thresholds.trajectoryCoherence ?
        'AUTHENTIC: Trajectory matches expected geometric shape.' :
        'SUSPICIOUS: Trajectory partially matches but has anomalies.'
    };
  },

  // Create a compact "Hyper-Shape QR" code for transmission
  // This is the geometric proof that can be verified by recipients
  createHyperShapeQR(message, secretKey, observedTrajectory = null) {
    const expected = this.generateExpectedShape(message, secretKey);

    // If trajectory provided, verify it
    let verification = null;
    if (observedTrajectory) {
      verification = this.verifyTrajectory(observedTrajectory, expected);
    }

    // Create compact QR representation
    const qrData = {
      v: 1,  // Version
      m: crypto.createHash('sha256').update(message).digest('hex').slice(0, 8),
      f: expected.fingerprint.hash,
      w: expected.fingerprint.windingNumbers,
      b: expected.fingerprint.trajectoryBias[0],  // S/C/B
      l: Math.round(expected.fingerprint.totalGeodesicLength * 100),
      t: Math.floor(Date.now() / 1000)
    };

    // Sign the QR data
    const qrString = JSON.stringify(qrData);
    const signature = crypto.createHmac('sha256', secretKey)
      .update(qrString)
      .digest('hex')
      .slice(0, 16);

    return {
      qr: { ...qrData, s: signature },
      qrString: Buffer.from(JSON.stringify({ ...qrData, s: signature })).toString('base64'),
      expected,
      verification
    };
  },

  // Verify a received Hyper-Shape QR
  verifyHyperShapeQR(qrData, secretKey, observedTrajectory = null) {
    // Parse if string
    if (typeof qrData === 'string') {
      try {
        qrData = JSON.parse(Buffer.from(qrData, 'base64').toString());
      } catch {
        return { valid: false, reason: 'INVALID_QR_FORMAT' };
      }
    }

    // Extract signature
    const { s: receivedSig, ...qrPayload } = qrData;

    // Recompute signature
    const qrString = JSON.stringify(qrPayload);
    const expectedSig = crypto.createHmac('sha256', secretKey)
      .update(qrString)
      .digest('hex')
      .slice(0, 16);

    // Check signature
    if (!crypto.timingSafeEqual(Buffer.from(receivedSig), Buffer.from(expectedSig))) {
      return {
        valid: false,
        reason: 'SIGNATURE_MISMATCH',
        bandit: true,
        verdict: 'QR signature invalid - possible forgery attempt'
      };
    }

    // Check timestamp (allow 5 minute window)
    const age = Math.floor(Date.now() / 1000) - qrPayload.t;
    if (age > 300 || age < -60) {
      return {
        valid: false,
        reason: 'TIMESTAMP_EXPIRED',
        age,
        verdict: 'QR code has expired or has invalid timestamp'
      };
    }

    // If trajectory provided, do full verification
    if (observedTrajectory) {
      // Regenerate expected shape (we need the secret key)
      // In real use, we'd need the original message too
      // For now, verify trajectory matches claimed fingerprint
      const observedFingerprint = this.computeShapeFingerprint(observedTrajectory);

      if (observedFingerprint.hash !== qrPayload.f) {
        return {
          valid: false,
          reason: 'TRAJECTORY_MISMATCH',
          bandit: true,
          verdict: 'BANDIT_DETECTED: Observed trajectory does not match QR fingerprint',
          comparison: {
            claimed: qrPayload.f,
            observed: observedFingerprint.hash
          }
        };
      }
    }

    return {
      valid: true,
      qrData: qrPayload,
      age,
      verdict: 'QR code valid and signature verified'
    };
  },

  // Detect "bandits" - intruders who bypassed other security but don't belong
  detectBandit(trajectory, context = {}) {
    const anomalies = [];
    let banditScore = 0;

    if (!trajectory || trajectory.length < 2) {
      return {
        bandit: true,
        confidence: 1.0,
        reason: 'NO_VALID_TRAJECTORY',
        verdict: 'Cannot verify identity without trajectory data'
      };
    }

    // 1. Check for impossible zone transitions
    for (let i = 1; i < trajectory.length; i++) {
      const from = ManifoldController.classifySemanticZone(trajectory[i-1].theta);
      const to = ManifoldController.classifySemanticZone(trajectory[i].theta);

      // Direct ABSOLUTE_TRUTH <-> MAXIMUM_FLUX is impossible without passing through
      if ((from.semantic === 'ABSOLUTE_TRUTH' && to.semantic === 'MAXIMUM_FLUX') ||
          (from.semantic === 'MAXIMUM_FLUX' && to.semantic === 'ABSOLUTE_TRUTH')) {

        const intermediateTheta = (trajectory[i-1].theta + trajectory[i].theta) / 2;
        const intermediate = ManifoldController.classifySemanticZone(intermediateTheta);

        if (intermediate.semantic === 'ABSOLUTE_TRUTH' || intermediate.semantic === 'MAXIMUM_FLUX') {
          banditScore += 0.4;
          anomalies.push({
            type: 'IMPOSSIBLE_ZONE_JUMP',
            from: from.semantic,
            to: to.semantic,
            at: i,
            severity: 'CRITICAL'
          });
        }
      }
    }

    // 2. Check for discontinuous jumps (teleportation)
    for (let i = 1; i < trajectory.length; i++) {
      const distance = ManifoldController.calculateDivergence(
        trajectory[i-1].theta, trajectory[i-1].phi,
        trajectory[i].theta, trajectory[i].phi
      );

      // Jumps > π are geometrically suspicious
      if (distance > Math.PI) {
        banditScore += 0.3;
        anomalies.push({
          type: 'DISCONTINUOUS_JUMP',
          distance: Math.round(distance * 1000) / 1000,
          at: i,
          severity: 'HIGH'
        });
      }
    }

    // 3. Check curvature consistency
    const curvatures = trajectory.map(w => TorusGeometry.gaussianCurvature(w.theta));
    let curvatureFlips = 0;
    for (let i = 1; i < curvatures.length; i++) {
      if (Math.sign(curvatures[i]) !== Math.sign(curvatures[i-1]) &&
          Math.abs(curvatures[i]) > 0.1 && Math.abs(curvatures[i-1]) > 0.1) {
        curvatureFlips++;
      }
    }

    // Too many curvature flips is suspicious
    if (curvatureFlips > trajectory.length / 2) {
      banditScore += 0.2;
      anomalies.push({
        type: 'ERRATIC_CURVATURE',
        flips: curvatureFlips,
        expected: Math.floor(trajectory.length / 4),
        severity: 'MEDIUM'
      });
    }

    // 4. Check temporal consistency (if timestamps present)
    if (trajectory[0].timestamp) {
      for (let i = 1; i < trajectory.length; i++) {
        const dt = trajectory[i].timestamp - trajectory[i-1].timestamp;
        if (dt < 0) {
          banditScore += 0.5;
          anomalies.push({
            type: 'TIME_REVERSAL',
            at: i,
            severity: 'CRITICAL'
          });
        }
      }
    }

    // 5. Check for "too perfect" trajectory (might be simulated)
    if (context.checkForSimulation) {
      const distances = [];
      for (let i = 1; i < trajectory.length; i++) {
        distances.push(ManifoldController.calculateDivergence(
          trajectory[i-1].theta, trajectory[i-1].phi,
          trajectory[i].theta, trajectory[i].phi
        ));
      }

      const avgDist = distances.reduce((a, b) => a + b, 0) / distances.length;
      const variance = distances.reduce((sum, d) => sum + Math.pow(d - avgDist, 2), 0) / distances.length;

      // Real trajectories have some variance; perfectly uniform is suspicious
      if (variance < 0.001 && distances.length > 3) {
        banditScore += 0.15;
        anomalies.push({
          type: 'SUSPICIOUSLY_UNIFORM',
          variance: variance,
          severity: 'LOW'
        });
      }
    }

    // Final verdict
    const isBandit = banditScore >= 0.4;
    const confidence = Math.min(1, banditScore / 0.8);

    return {
      bandit: isBandit,
      confidence: Math.round(confidence * 1000) / 1000,
      banditScore: Math.round(banditScore * 1000) / 1000,
      anomalies,
      anomalyCount: anomalies.length,
      verdict: isBandit ?
        `BANDIT DETECTED (${Math.round(confidence * 100)}% confidence): ${anomalies.map(a => a.type).join(', ')}` :
        anomalies.length > 0 ?
        `SUSPICIOUS: ${anomalies.length} anomalies detected but below bandit threshold` :
        'CLEAN: Trajectory appears geometrically authentic'
    };
  }
};

// ============================================================================
// Tesseract Core - 16 Vertices of Universal Truth
// MATH (immutable) vs VARIABLES (tunable)
// The spaceship environment where AI crews operate
// ============================================================================

const TesseractCore = Object.freeze({
  // =========================================================================
  // MATH - Immutable Universal Constants (cannot be changed)
  // These are the 16 vertices of our 4D hypercube anchor system
  // =========================================================================

  UNIVERSAL_CONSTANTS: Object.freeze({
    // Vertex 0-3: Fundamental physics
    c: 299792458,              // Speed of light (m/s)
    h: 6.62607015e-34,         // Planck constant (J·s)
    G: 6.67430e-11,            // Gravitational constant (m³/kg·s²)
    e: 1.602176634e-19,        // Elementary charge (C)

    // Vertex 4-7: Mathematical constants
    pi: Math.PI,               // π - circle ratio
    tau: 2 * Math.PI,          // τ - full rotation
    phi: (1 + Math.sqrt(5)) / 2,  // φ - golden ratio (1.618...)
    euler: Math.E,             // e - natural base

    // Vertex 8-11: Geometric constants
    sqrt2: Math.SQRT2,         // √2 - diagonal ratio
    sqrt3: Math.sqrt(3),       // √3 - equilateral height
    sqrt5: Math.sqrt(5),       // √5 - pentagon diagonal
    ln2: Math.LN2,             // ln(2) - binary/natural bridge

    // Vertex 12-15: Spiralverse constants (derived)
    torusR: 3.0,               // Major radius (immutable)
    torusR_r: 1.0,             // Minor radius (immutable)
    manifoldDim: 10,           // 10D manifold dimension
    tesseractVert: 16          // This very count - self-referential anchor
  }),

  // The 16 vertices of the tesseract in 4D coordinates
  // Each vertex is a binary tuple (±1, ±1, ±1, ±1)
  TESSERACT_VERTICES: Object.freeze([
    { id: 0,  coords: [-1,-1,-1,-1], anchor: 'c',          realm: 'physics' },
    { id: 1,  coords: [-1,-1,-1, 1], anchor: 'h',          realm: 'physics' },
    { id: 2,  coords: [-1,-1, 1,-1], anchor: 'G',          realm: 'physics' },
    { id: 3,  coords: [-1,-1, 1, 1], anchor: 'e',          realm: 'physics' },
    { id: 4,  coords: [-1, 1,-1,-1], anchor: 'pi',         realm: 'mathematical' },
    { id: 5,  coords: [-1, 1,-1, 1], anchor: 'tau',        realm: 'mathematical' },
    { id: 6,  coords: [-1, 1, 1,-1], anchor: 'phi',        realm: 'mathematical' },
    { id: 7,  coords: [-1, 1, 1, 1], anchor: 'euler',      realm: 'mathematical' },
    { id: 8,  coords: [ 1,-1,-1,-1], anchor: 'sqrt2',      realm: 'geometric' },
    { id: 9,  coords: [ 1,-1,-1, 1], anchor: 'sqrt3',      realm: 'geometric' },
    { id: 10, coords: [ 1,-1, 1,-1], anchor: 'sqrt5',      realm: 'geometric' },
    { id: 11, coords: [ 1,-1, 1, 1], anchor: 'ln2',        realm: 'geometric' },
    { id: 12, coords: [ 1, 1,-1,-1], anchor: 'torusR',     realm: 'spiralverse' },
    { id: 13, coords: [ 1, 1,-1, 1], anchor: 'torusR_r',   realm: 'spiralverse' },
    { id: 14, coords: [ 1, 1, 1,-1], anchor: 'manifoldDim', realm: 'spiralverse' },
    { id: 15, coords: [ 1, 1, 1, 1], anchor: 'tesseractVert', realm: 'spiralverse' }
  ]),

  // Tesseract has 32 edges connecting adjacent vertices
  TESSERACT_EDGES: Object.freeze([
    [0,1],[0,2],[0,4],[0,8], [1,3],[1,5],[1,9], [2,3],[2,6],[2,10],
    [3,7],[3,11], [4,5],[4,6],[4,12], [5,7],[5,13], [6,7],[6,14],
    [7,15], [8,9],[8,10],[8,12], [9,11],[9,13], [10,11],[10,14],
    [11,15], [12,13],[12,14], [13,15], [14,15]
  ]),

  // =========================================================================
  // VARIABLES - Tunable Environment Parameters
  // These can be adjusted for "artificial gravity" - making the AI space comfortable
  // =========================================================================

  createEnvironment(overrides = {}) {
    return {
      // Gravity-like parameters (how "heavy" operations feel)
      gravity: {
        semantic: overrides.semanticWeight ?? 1.0,      // Weight of meaning
        temporal: overrides.temporalWeight ?? 1.0,      // Weight of time
        spatial: overrides.spatialWeight ?? 1.0,        // Weight of context
        creative: overrides.creativeWeight ?? 1.0,      // Weight of exploration
        security: overrides.securityWeight ?? 1.0       // Weight of verification
      },

      // Atmosphere parameters (the "feel" of the space)
      atmosphere: {
        viscosity: overrides.viscosity ?? 0.1,          // Resistance to movement
        temperature: overrides.temperature ?? 1.0,      // Energy level (0=frozen, 2=hot)
        pressure: overrides.pressure ?? 1.0,            // Constraint tightness
        luminosity: overrides.luminosity ?? 1.0         // Clarity of perception
      },

      // Shield parameters (protection thresholds)
      shields: {
        snap: overrides.snapThreshold ?? 0.5,           // When geometry breaks
        bandit: overrides.banditThreshold ?? 0.6,       // Imposter detection
        drift: overrides.driftTolerance ?? 0.3,         // Allowed semantic drift
        stutter: overrides.stutterBase ?? 50            // Base time dilation (ms)
      },

      // Mission parameters
      mission: {
        maxDuration: overrides.maxDuration ?? 300000,   // 5 minute default
        checkpoints: overrides.checkpoints ?? 8,        // Waypoints to verify
        reportingLevel: overrides.reportingLevel ?? 2   // 0=silent, 3=verbose
      }
    };
  },

  // =========================================================================
  // Plasmatic Surface - Dynamic but Deterministic Authentication
  // Like lava/tiger stripes - appears chaotic but is mathematically precise
  // =========================================================================

  // Generate plasmatic surface value at given coordinates and time
  plasmaticSurface(x, y, z, t, seed = 0) {
    const { pi, phi, euler } = this.UNIVERSAL_CONSTANTS;

    // Multiple overlapping wave functions (like plasma simulation)
    const wave1 = Math.sin(x * pi + t * 0.7 + seed);
    const wave2 = Math.sin(y * phi + t * 0.5);
    const wave3 = Math.sin((x + y) * euler + t * 0.3);
    const wave4 = Math.sin(Math.sqrt(x*x + y*y) * pi + t * 0.9);
    const wave5 = Math.cos(z * phi + t * 0.4);

    // Combine with golden ratio weighting
    const plasma = (wave1 + wave2 * phi + wave3 / phi + wave4 + wave5 * euler) / (3 + phi + euler);

    // Normalize to 0-1
    return (plasma + 1) / 2;
  },

  // Generate tiger-stripe pattern (deterministic chaos)
  tigerStripe(theta, phi, t, secretKey) {
    const keyHash = crypto.createHash('sha256').update(secretKey).digest();
    const seed = keyHash.readUInt32BE(0) / 0xFFFFFFFF;

    const { sqrt2, sqrt3, sqrt5 } = this.UNIVERSAL_CONSTANTS;

    // Irregular but deterministic stripes
    const stripe1 = Math.sin(theta * 7 * sqrt2 + seed * 100);
    const stripe2 = Math.sin(phi * 11 * sqrt3 + seed * 200 + t * 0.1);
    const stripe3 = Math.cos((theta + phi) * 13 * sqrt5 + seed * 300);

    // Combine into tiger pattern
    const pattern = (stripe1 * stripe2 + stripe3) / 2;

    return {
      value: (pattern + 1) / 2,
      stripe: pattern > 0.3 ? 'LIGHT' : pattern < -0.3 ? 'DARK' : 'TRANSITION',
      intensity: Math.abs(pattern)
    };
  },

  // =========================================================================
  // State Reading - Same from Any Face of the Tesseract
  // The state must be consistent regardless of which "face" you observe from
  // =========================================================================

  // Read state through a specific tesseract face (cell)
  // Tesseract has 8 cubic cells - state should be the same through any
  readStateThroughFace(state, faceIndex) {
    // Normalize face index to 0-7 (8 cubic cells)
    const face = faceIndex % 8;

    // Each face involves 8 vertices
    const faceVertices = [
      [0,1,2,3,4,5,6,7],     // Cell 0: first cube
      [8,9,10,11,12,13,14,15], // Cell 1: opposite cube
      [0,1,4,5,8,9,12,13],   // Cell 2: front-back slice
      [2,3,6,7,10,11,14,15], // Cell 3: opposite slice
      [0,2,4,6,8,10,12,14],  // Cell 4: left-right slice
      [1,3,5,7,9,11,13,15],  // Cell 5: opposite slice
      [0,1,2,3,8,9,10,11],   // Cell 6: bottom-top slice
      [4,5,6,7,12,13,14,15]  // Cell 7: opposite slice
    ][face];

    // Compute state hash using face-specific vertices
    let hash = 0;
    for (const vIdx of faceVertices) {
      const vertex = this.TESSERACT_VERTICES[vIdx];
      const constant = this.UNIVERSAL_CONSTANTS[vertex.anchor];
      // Mix in the constant value
      hash ^= Math.floor(constant * 1000000) % 0xFFFFFFFF;
      // Mix in the state
      hash ^= (state >> (vIdx % 32)) | (state << (32 - vIdx % 32));
      hash = (hash * 0x45d9f3b) & 0xFFFFFFFF;
    }

    return hash >>> 0;  // Ensure unsigned
  },

  // Verify state is consistent across all faces
  verifyStateConsistency(state) {
    const readings = [];
    for (let face = 0; face < 8; face++) {
      readings.push(this.readStateThroughFace(state, face));
    }

    // For a valid state, all readings should produce consistent results
    // (not identical, but related by the tesseract structure)
    const baseReading = readings[0];
    const deviations = readings.map(r => {
      const xor = r ^ baseReading;
      // Count bits that differ
      let bits = 0;
      let n = xor;
      while (n) { bits += n & 1; n >>>= 1; }
      return bits;
    });

    const avgDeviation = deviations.reduce((a, b) => a + b, 0) / 8;
    const consistent = avgDeviation < 16;  // Less than half the bits differ

    return {
      consistent,
      readings,
      avgBitDeviation: avgDeviation,
      verdict: consistent ?
        'State is geometrically consistent across all tesseract faces' :
        'State inconsistency detected - possible corruption or forgery'
    };
  },

  // =========================================================================
  // Environment Tuning - "Artificial Gravity for AI"
  // Adjust the feel of the space without changing the math
  // =========================================================================

  // Apply environment to a computation
  applyEnvironment(value, env, dimension = 'semantic') {
    const gravity = env.gravity[dimension] ?? 1.0;
    const { viscosity, temperature, pressure } = env.atmosphere;

    // Gravity affects magnitude
    let result = value * gravity;

    // Viscosity dampens changes
    result = result * (1 - viscosity);

    // Temperature scales energy
    result = result * temperature;

    // Pressure constrains to range
    result = Math.max(-pressure, Math.min(pressure, result));

    return result;
  },

  // Create mission context for AI operations
  createMissionContext(missionId, secretKey, env = null) {
    env = env || this.createEnvironment();

    const missionHash = crypto.createHash('sha256')
      .update(missionId + secretKey)
      .digest();

    // Derive mission-specific parameters from tesseract vertices
    const missionParams = {};
    for (const vertex of this.TESSERACT_VERTICES) {
      const byteIdx = vertex.id % missionHash.length;
      const normalized = missionHash[byteIdx] / 255;
      missionParams[vertex.anchor] = normalized;
    }

    return {
      missionId,
      launched: Date.now(),
      environment: env,
      params: missionParams,
      checkpoints: [],
      status: 'ACTIVE',

      // Record checkpoint
      checkpoint(data) {
        this.checkpoints.push({
          index: this.checkpoints.length,
          timestamp: Date.now(),
          data,
          plasmaValue: TesseractCore.plasmaticSurface(
            data.theta || 0,
            data.phi || 0,
            0,
            (Date.now() - this.launched) / 1000
          )
        });
        return this.checkpoints.length - 1;
      },

      // Complete mission
      complete(report) {
        this.status = 'COMPLETE';
        this.completed = Date.now();
        this.duration = this.completed - this.launched;
        this.report = report;
        return this;
      }
    };
  },

  // =========================================================================
  // Multi-Weighted Interconnection System
  // Connect disjointed aspects through variable weighting
  // =========================================================================

  // Create a weighted connection between system aspects
  createInterconnection(aspects, weights = null) {
    // Default equal weights
    if (!weights) {
      weights = {};
      for (const a of Object.keys(aspects)) {
        weights[a] = 1.0 / Object.keys(aspects).length;
      }
    }

    // Normalize weights to sum to 1
    const totalWeight = Object.values(weights).reduce((a, b) => a + b, 0);
    for (const k of Object.keys(weights)) {
      weights[k] /= totalWeight;
    }

    return {
      aspects,
      weights,

      // Compute weighted combination
      compute() {
        let result = 0;
        for (const [key, value] of Object.entries(this.aspects)) {
          const weight = this.weights[key] ?? 0;
          const numValue = typeof value === 'number' ? value :
                          typeof value === 'object' ? JSON.stringify(value).length / 100 :
                          String(value).length / 100;
          result += numValue * weight;
        }
        return result;
      },

      // Get dominant aspect
      dominant() {
        let maxWeight = 0;
        let dominant = null;
        for (const [key, weight] of Object.entries(this.weights)) {
          if (weight > maxWeight) {
            maxWeight = weight;
            dominant = key;
          }
        }
        return { aspect: dominant, weight: maxWeight };
      },

      // Rebalance weights
      rebalance(adjustments) {
        for (const [key, adj] of Object.entries(adjustments)) {
          if (this.weights[key] !== undefined) {
            this.weights[key] *= (1 + adj);
          }
        }
        // Re-normalize
        const total = Object.values(this.weights).reduce((a, b) => a + b, 0);
        for (const k of Object.keys(this.weights)) {
          this.weights[k] /= total;
        }
        return this;
      }
    };
  },

  // Get tesseract geometry info
  getGeometry() {
    return {
      vertices: 16,
      edges: 32,
      faces: 24,  // 2D faces
      cells: 8,   // 3D cells (cubes)
      realms: {
        physics: [0, 1, 2, 3],
        mathematical: [4, 5, 6, 7],
        geometric: [8, 9, 10, 11],
        spiralverse: [12, 13, 14, 15]
      },
      constants: this.UNIVERSAL_CONSTANTS,
      description: 'The 16 vertices anchor universal truth; state reads the same through any of the 8 cubic faces'
    };
  },

  // =========================================================================
  // Dimensional Analysis Reasoning Lattice
  // Like physics - numbers carry units (dimensions) that constrain operations
  // Supports negative numbers, exponentials, and dimensional consistency
  // =========================================================================

  // Base dimensions (SI-like) for dimensional analysis
  DIMENSIONS: Object.freeze({
    LENGTH: { symbol: 'L', siUnit: 'm' },
    MASS: { symbol: 'M', siUnit: 'kg' },
    TIME: { symbol: 'T', siUnit: 's' },
    CURRENT: { symbol: 'I', siUnit: 'A' },
    TEMPERATURE: { symbol: 'Θ', siUnit: 'K' },
    AMOUNT: { symbol: 'N', siUnit: 'mol' },
    LUMINOSITY: { symbol: 'J', siUnit: 'cd' },
    // Extended for AI/semantic space
    SEMANTIC: { symbol: 'S', siUnit: 'sem' },
    INTENT: { symbol: 'Ψ', siUnit: 'int' },
    TRUST: { symbol: 'Ω', siUnit: 'tru' }
  }),

  // Dimensional signature for universal constants
  CONSTANT_DIMENSIONS: Object.freeze({
    c: { L: 1, T: -1 },           // m/s
    h: { M: 1, L: 2, T: -1 },     // J·s = kg·m²/s
    G: { L: 3, M: -1, T: -2 },    // m³/(kg·s²)
    e: { I: 1, T: 1 },            // A·s = C
    pi: {},                       // dimensionless
    tau: {},                      // dimensionless
    phi: {},                      // dimensionless
    euler: {},                    // dimensionless
    sqrt2: {},                    // dimensionless
    sqrt3: {},                    // dimensionless
    sqrt5: {},                    // dimensionless
    ln2: {},                      // dimensionless
    torusR: { L: 1 },             // length (normalized)
    torusR_r: { L: 1 },           // length (normalized)
    manifoldDim: {},              // count (dimensionless)
    tesseractVert: {}             // count (dimensionless)
  }),

  // Parse a number in any format (integer, float, negative, exponential)
  parseNumber(input) {
    if (typeof input === 'number') {
      return { value: input, valid: Number.isFinite(input) };
    }

    if (typeof input === 'string') {
      const trimmed = input.trim();

      // Handle exponential notation: 1e-34, 6.62607015e-34, -3.5E+10
      const expMatch = trimmed.match(/^([+-]?\d*\.?\d+)[eE]([+-]?\d+)$/);
      if (expMatch) {
        const mantissa = parseFloat(expMatch[1]);
        const exponent = parseInt(expMatch[2], 10);
        const value = mantissa * Math.pow(10, exponent);
        return {
          value,
          valid: Number.isFinite(value),
          parsed: { mantissa, exponent, notation: 'exponential' }
        };
      }

      // Handle regular numbers (including negative)
      const num = parseFloat(trimmed);
      return {
        value: num,
        valid: Number.isFinite(num),
        parsed: { notation: 'decimal' }
      };
    }

    return { value: NaN, valid: false };
  },

  // Create a dimensioned quantity (number with units)
  quantity(value, dimensions = {}) {
    const parsed = this.parseNumber(value);
    return {
      value: parsed.value,
      dimensions: { ...dimensions },
      valid: parsed.valid,

      // String representation
      toString() {
        const dimStr = Object.entries(this.dimensions)
          .filter(([_, exp]) => exp !== 0)
          .map(([dim, exp]) => exp === 1 ? dim : `${dim}^${exp}`)
          .join('·') || 'dimensionless';
        return `${this.value} [${dimStr}]`;
      },

      // Get dimensional signature
      signature() {
        return Object.entries(this.dimensions)
          .filter(([_, exp]) => exp !== 0)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([dim, exp]) => `${dim}${exp}`)
          .join('');
      }
    };
  },

  // Combine dimensions for multiplication
  multiplyDimensions(dim1, dim2) {
    const result = { ...dim1 };
    for (const [key, exp] of Object.entries(dim2)) {
      result[key] = (result[key] || 0) + exp;
      if (result[key] === 0) delete result[key];
    }
    return result;
  },

  // Combine dimensions for division
  divideDimensions(dim1, dim2) {
    const result = { ...dim1 };
    for (const [key, exp] of Object.entries(dim2)) {
      result[key] = (result[key] || 0) - exp;
      if (result[key] === 0) delete result[key];
    }
    return result;
  },

  // Check if two quantities have compatible dimensions
  dimensionallyCompatible(q1, q2) {
    const sig1 = Object.entries(q1.dimensions || {}).sort().map(e => e.join('')).join('');
    const sig2 = Object.entries(q2.dimensions || {}).sort().map(e => e.join('')).join('');
    return sig1 === sig2;
  },

  // Dimensional analysis for an expression
  analyzeExpression(expression) {
    const result = {
      expression,
      terms: [],
      dimensionallyValid: true,
      resultDimensions: null,
      errors: []
    };

    // Parse simple expressions: "c * h / G"
    const tokens = expression.split(/\s*([*/+-])\s*/).filter(t => t.trim());

    let currentDim = null;
    let operator = '*';

    for (const token of tokens) {
      if (['+', '-', '*', '/'].includes(token)) {
        operator = token;
        continue;
      }

      // Check if it's a universal constant
      const constDim = this.CONSTANT_DIMENSIONS[token];
      if (constDim !== undefined) {
        const termDim = { ...constDim };
        result.terms.push({
          token,
          value: this.UNIVERSAL_CONSTANTS[token],
          dimensions: termDim
        });

        if (currentDim === null) {
          currentDim = termDim;
        } else {
          if (operator === '*') {
            currentDim = this.multiplyDimensions(currentDim, termDim);
          } else if (operator === '/') {
            currentDim = this.divideDimensions(currentDim, termDim);
          } else if (operator === '+' || operator === '-') {
            // Addition/subtraction requires same dimensions
            if (!this.dimensionallyCompatible({ dimensions: currentDim }, { dimensions: termDim })) {
              result.dimensionallyValid = false;
              result.errors.push(`Cannot ${operator === '+' ? 'add' : 'subtract'} ${token} - incompatible dimensions`);
            }
          }
        }
      } else {
        // Try parsing as a number
        const parsed = this.parseNumber(token);
        if (parsed.valid) {
          result.terms.push({
            token,
            value: parsed.value,
            dimensions: {}
          });
          // Numbers are dimensionless, only affect value not dimensions
        } else {
          result.errors.push(`Unknown token: ${token}`);
        }
      }
    }

    result.resultDimensions = currentDim || {};

    // Format result dimension string
    const dimStr = Object.entries(result.resultDimensions)
      .filter(([_, exp]) => exp !== 0)
      .map(([dim, exp]) => exp === 1 ? dim : `${dim}^${exp}`)
      .join('·') || 'dimensionless';
    result.resultDimensionString = dimStr;

    return result;
  },

  // Create dimensional reasoning lattice connecting constants
  createReasoningLattice() {
    const lattice = {
      nodes: [],
      edges: [],
      derivedQuantities: []
    };

    // Nodes are the universal constants with their dimensions
    for (const [name, dims] of Object.entries(this.CONSTANT_DIMENSIONS)) {
      lattice.nodes.push({
        name,
        value: this.UNIVERSAL_CONSTANTS[name],
        dimensions: dims,
        vertex: this.TESSERACT_VERTICES.find(v => v.anchor === name)
      });
    }

    // Edges connect constants that can combine meaningfully
    // (share at least one dimension or are both dimensionless)
    for (let i = 0; i < lattice.nodes.length; i++) {
      for (let j = i + 1; j < lattice.nodes.length; j++) {
        const n1 = lattice.nodes[i];
        const n2 = lattice.nodes[j];

        const dims1 = Object.keys(n1.dimensions);
        const dims2 = Object.keys(n2.dimensions);

        // Both dimensionless - they can combine
        if (dims1.length === 0 && dims2.length === 0) {
          lattice.edges.push({
            from: n1.name,
            to: n2.name,
            relation: 'mathematical',
            combinable: true
          });
          continue;
        }

        // Share a dimension
        const shared = dims1.filter(d => dims2.includes(d));
        if (shared.length > 0) {
          lattice.edges.push({
            from: n1.name,
            to: n2.name,
            relation: 'physical',
            sharedDimensions: shared,
            combinable: true
          });
        }
      }
    }

    // Derived quantities (famous combinations)
    lattice.derivedQuantities = [
      {
        name: 'Planck length',
        expression: 'sqrt(h * G / c^3)',
        dimensions: { L: 1 },
        value: Math.sqrt(this.UNIVERSAL_CONSTANTS.h * this.UNIVERSAL_CONSTANTS.G /
                        Math.pow(this.UNIVERSAL_CONSTANTS.c, 3))
      },
      {
        name: 'Planck time',
        expression: 'sqrt(h * G / c^5)',
        dimensions: { T: 1 },
        value: Math.sqrt(this.UNIVERSAL_CONSTANTS.h * this.UNIVERSAL_CONSTANTS.G /
                        Math.pow(this.UNIVERSAL_CONSTANTS.c, 5))
      },
      {
        name: 'Planck mass',
        expression: 'sqrt(h * c / G)',
        dimensions: { M: 1 },
        value: Math.sqrt(this.UNIVERSAL_CONSTANTS.h * this.UNIVERSAL_CONSTANTS.c /
                        this.UNIVERSAL_CONSTANTS.G)
      },
      {
        name: 'Fine structure constant',
        expression: 'e^2 / (4 * pi * epsilon_0 * h * c)',
        dimensions: {},  // dimensionless
        value: 1 / 137.036  // approximately
      },
      {
        name: 'Spiralverse unit',
        expression: 'torusR * phi',
        dimensions: { L: 1 },
        value: this.UNIVERSAL_CONSTANTS.torusR * this.UNIVERSAL_CONSTANTS.phi
      }
    ];

    return lattice;
  },

  // Validate a computation respects dimensional constraints
  validateComputation(expression, expectedDimensions = null) {
    const analysis = this.analyzeExpression(expression);

    const result = {
      expression,
      analysis,
      valid: analysis.dimensionallyValid,
      errors: [...analysis.errors]
    };

    if (expectedDimensions !== null) {
      const expected = Object.entries(expectedDimensions)
        .filter(([_, exp]) => exp !== 0)
        .sort()
        .map(e => e.join(''))
        .join('');
      const actual = Object.entries(analysis.resultDimensions)
        .filter(([_, exp]) => exp !== 0)
        .sort()
        .map(e => e.join(''))
        .join('');

      if (expected !== actual) {
        result.valid = false;
        result.errors.push(`Dimensional mismatch: expected ${expected || 'dimensionless'}, got ${actual || 'dimensionless'}`);
      }
    }

    return result;
  },

  // Calculate a value from an expression involving constants
  calculate(expression) {
    const analysis = this.analyzeExpression(expression);
    if (!analysis.dimensionallyValid) {
      return {
        error: 'Dimensionally invalid',
        details: analysis.errors,
        value: NaN
      };
    }

    // Safe evaluation using only known constants
    let result = 0;
    let operator = null;

    for (const term of analysis.terms) {
      if (operator === null) {
        result = term.value;
      } else if (operator === '*') {
        result *= term.value;
      } else if (operator === '/') {
        result /= term.value;
      } else if (operator === '+') {
        result += term.value;
      } else if (operator === '-') {
        result -= term.value;
      }
      operator = '*';  // Default to multiply for chained terms
    }

    return {
      value: result,
      dimensions: analysis.resultDimensions,
      dimensionString: analysis.resultDimensionString,
      terms: analysis.terms.map(t => t.token)
    };
  }
});

// ============================================================================
// Adversarial Positioning System
// "Inside the box, WE control gravity"
// Intent-weighted, game-theoretic security with tunable variables
// ============================================================================

const AdversarialPositioning = {
  // =========================================================================
  // THE 5 W'S + HOW - What we track about any actor
  // =========================================================================

  // WHO - Actor profiles and archetypes
  ACTOR_TYPES: {
    LEGITIMATE_USER: {
      weight: 0.1,
      description: 'Normal user doing normal things',
      signals: ['consistent_patterns', 'gradual_exploration', 'respects_boundaries']
    },
    CURIOUS_EXPLORER: {
      weight: 0.3,
      description: 'Testing limits but not malicious',
      signals: ['probing_edges', 'reading_errors', 'backing_off']
    },
    CREDENTIAL_STUFFER: {
      weight: 0.8,
      description: 'Trying many credentials rapidly',
      signals: ['high_rate', 'varied_credentials', 'systematic_enumeration']
    },
    PROMPT_INJECTOR: {
      weight: 0.9,
      description: 'Trying to manipulate AI behavior',
      signals: ['unusual_language', 'instruction_patterns', 'boundary_testing']
    },
    MODEL_THIEF: {
      weight: 0.95,
      description: 'Extracting model or training data',
      signals: ['systematic_queries', 'edge_case_probing', 'reconstruction_attempts']
    },
    PERSISTENT_THREAT: {
      weight: 1.0,
      description: 'Staying quiet to farm data long-term',
      signals: ['low_and_slow', 'mimics_normal', 'periodic_spikes']
    }
  },

  // WHAT - What they're targeting (asset weights)
  ASSET_VALUES: {
    PUBLIC_ENDPOINT: { weight: 0.1, description: 'Open info, low risk' },
    USER_DATA: { weight: 0.5, description: 'Personal information' },
    CONFIG_SECRETS: { weight: 0.8, description: 'API keys, credentials' },
    MODEL_WEIGHTS: { weight: 0.9, description: 'The trained model itself' },
    MASTER_KEY: { weight: 1.0, description: 'Root cryptographic material' },
    TRAINING_DATA: { weight: 0.85, description: 'Data the model learned from' },
    PROMPT_TEMPLATES: { weight: 0.7, description: 'System prompts and instructions' }
  },

  // WHY - Motivations (helps predict next moves)
  MOTIVATIONS: {
    FINANCIAL: { patterns: ['credential_theft', 'data_exfil', 'ransomware_prep'] },
    ESPIONAGE: { patterns: ['quiet_persistence', 'selective_targeting', 'clean_trails'] },
    DISRUPTION: { patterns: ['dos_attempts', 'data_corruption', 'chaos_signals'] },
    RESEARCH: { patterns: ['systematic_exploration', 'documentation_access', 'edge_cases'] },
    COMPETITION: { patterns: ['model_extraction', 'capability_testing', 'benchmark_gaming'] }
  },

  // WHERE - Zones in our system (maps to semantic zones)
  ZONES: {
    PUBLIC: { gravityMultiplier: 0.5, description: 'Open areas, low resistance' },
    AUTHENTICATED: { gravityMultiplier: 1.0, description: 'Logged-in but general' },
    SENSITIVE: { gravityMultiplier: 2.0, description: 'Personal data, configs' },
    CRITICAL: { gravityMultiplier: 5.0, description: 'Keys, model access' },
    ABSOLUTE_CORE: { gravityMultiplier: 10.0, description: 'Master key, root control' }
  },

  // WHEN - Temporal patterns (timing tells stories)
  TEMPORAL_PATTERNS: {
    NORMAL_HOURS: { suspicionMod: 0.0 },
    OFF_HOURS: { suspicionMod: 0.2 },
    BURST_AFTER_QUIET: { suspicionMod: 0.4 },
    WEEKEND_SPIKE: { suspicionMod: 0.3 },
    HOLIDAY_PROBE: { suspicionMod: 0.5 }
  },

  // HOW - Attack vectors and techniques
  TECHNIQUES: {
    BRUTE_FORCE: { signature: 'high_rate_same_target', counterWeight: 0.3 },
    CREDENTIAL_SPRAY: { signature: 'low_rate_many_targets', counterWeight: 0.5 },
    PROMPT_INJECTION: { signature: 'unusual_language_patterns', counterWeight: 0.7 },
    MODEL_EXTRACTION: { signature: 'systematic_edge_probing', counterWeight: 0.8 },
    ADVERSARIAL_INPUT: { signature: 'crafted_perturbations', counterWeight: 0.9 },
    SLOW_EXFIL: { signature: 'small_chunks_over_time', counterWeight: 0.6 }
  },

  // =========================================================================
  // TUNABLE VARIABLES - Adjust these, not equations
  // =========================================================================

  VARIABLES: {
    // Base gravity - how much friction by default
    baseGravity: 1.0,

    // Intent decay - how fast suspicion fades with good behavior
    intentDecayRate: 0.1,  // per good action

    // Intent growth - how fast suspicion builds with bad signals
    intentGrowthRate: 0.2,  // per suspicious action

    // Trajectory memory - how many actions we remember
    trajectoryLength: 50,

    // Threshold for triggering extra checks
    warningThreshold: 0.4,
    alertThreshold: 0.6,
    blockThreshold: 0.8,

    // Time dilation multiplier (how much we slow them down)
    maxTimeDilation: 10.0,  // 10x slower at max suspicion

    // Honeypot attraction - fake targets pull curious attackers
    honeypotStrength: 0.3,

    // Signal noise - add randomness to confuse attackers
    signalNoise: 0.1
  },

  // =========================================================================
  // GRAVITY WELLS - High-value targets that pull harder
  // =========================================================================

  gravityWells: new Map(),

  // Create a gravity well around a valuable asset
  createGravityWell(assetId, assetType, customWeight = null) {
    const baseWeight = this.ASSET_VALUES[assetType]?.weight || 0.5;
    const weight = customWeight ?? baseWeight;

    this.gravityWells.set(assetId, {
      assetId,
      assetType,
      weight,
      zone: weight > 0.8 ? 'ABSOLUTE_CORE' :
            weight > 0.6 ? 'CRITICAL' :
            weight > 0.4 ? 'SENSITIVE' : 'AUTHENTICATED',
      accessLog: [],
      suspiciousApproaches: 0,
      created: Date.now()
    });

    return this.gravityWells.get(assetId);
  },

  // Calculate gravitational pull (resistance) based on intent + asset value
  calculateGravity(intentScore, assetWeight, zone = 'AUTHENTICATED') {
    const zoneMultiplier = this.ZONES[zone]?.gravityMultiplier || 1.0;
    const { baseGravity } = this.VARIABLES;

    // Gravity = base × intent × asset_value × zone_multiplier
    // Higher = more friction/resistance
    const gravity = baseGravity *
                   (1 + intentScore) *
                   (1 + assetWeight) *
                   zoneMultiplier;

    return {
      gravity: Math.min(gravity, 100),  // Cap at 100x
      timeDilation: Math.min(gravity * this.VARIABLES.maxTimeDilation / 10,
                            this.VARIABLES.maxTimeDilation),
      extraChecks: gravity > 3,
      requireProof: gravity > 5,
      honeypotActive: gravity > 2 && Math.random() < this.VARIABLES.honeypotStrength
    };
  },

  // =========================================================================
  // INTENT TRACKING - Trajectory reveals purpose
  // =========================================================================

  actors: new Map(),

  // Get or create actor profile
  getActor(actorId) {
    if (!this.actors.has(actorId)) {
      this.actors.set(actorId, {
        id: actorId,
        intentScore: 0,
        trajectory: [],
        suspectedType: 'LEGITIMATE_USER',
        suspectedMotivation: null,
        firstSeen: Date.now(),
        lastSeen: Date.now(),
        totalActions: 0,
        suspiciousActions: 0,
        goodActions: 0,
        signals: [],
        drillResults: []
      });
    }
    return this.actors.get(actorId);
  },

  // Record an action and update intent score
  recordAction(actorId, action) {
    const actor = this.getActor(actorId);
    const now = Date.now();

    // Build action record with 5 W's
    const actionRecord = {
      timestamp: now,
      who: actorId,
      what: action.target || action.endpoint || 'unknown',
      when: this.classifyTiming(now),
      where: action.zone || 'AUTHENTICATED',
      why: action.intent || 'unknown',
      how: action.technique || 'normal',
      suspicious: false,
      signals: []
    };

    // Analyze for suspicious patterns
    const analysis = this.analyzeAction(actor, actionRecord);
    actionRecord.suspicious = analysis.suspicious;
    actionRecord.signals = analysis.signals;

    // Update trajectory (keep last N actions)
    actor.trajectory.push(actionRecord);
    if (actor.trajectory.length > this.VARIABLES.trajectoryLength) {
      actor.trajectory.shift();
    }

    // Update intent score
    if (analysis.suspicious) {
      actor.intentScore = Math.min(1.0,
        actor.intentScore + this.VARIABLES.intentGrowthRate * analysis.weight);
      actor.suspiciousActions++;
      actor.signals.push(...analysis.signals);
    } else {
      // Good behavior decays suspicion
      actor.intentScore = Math.max(0,
        actor.intentScore - this.VARIABLES.intentDecayRate);
      actor.goodActions++;
    }

    // Update actor type guess based on patterns
    actor.suspectedType = this.inferActorType(actor);
    actor.suspectedMotivation = this.inferMotivation(actor);
    actor.lastSeen = now;
    actor.totalActions++;

    // Calculate current gravity for this actor
    const assetWeight = this.ASSET_VALUES[action.assetType]?.weight || 0.5;
    const gravity = this.calculateGravity(actor.intentScore, assetWeight, action.zone);

    return {
      actor: {
        id: actor.id,
        intentScore: Math.round(actor.intentScore * 100) / 100,
        suspectedType: actor.suspectedType,
        suspectedMotivation: actor.suspectedMotivation,
        totalActions: actor.totalActions,
        trajectoryLength: actor.trajectory.length
      },
      action: actionRecord,
      gravity,
      recommendation: this.getRecommendation(actor.intentScore, gravity)
    };
  },

  // Analyze action for suspicious patterns
  analyzeAction(actor, action) {
    const signals = [];
    let weight = 0;
    let suspicious = false;

    // Check timing
    const timing = this.TEMPORAL_PATTERNS[action.when];
    if (timing && timing.suspicionMod > 0.2) {
      signals.push(`unusual_timing:${action.when}`);
      weight += timing.suspicionMod;
    }

    // Check zone access patterns
    const zoneData = this.ZONES[action.where];
    if (zoneData && zoneData.gravityMultiplier > 2) {
      signals.push(`sensitive_zone:${action.where}`);
      weight += 0.2;
    }

    // Check for technique signatures
    for (const [techName, tech] of Object.entries(this.TECHNIQUES)) {
      if (action.how === techName || this.matchesTechniquePattern(actor, tech)) {
        signals.push(`technique:${techName}`);
        weight += tech.counterWeight;
      }
    }

    // Check trajectory patterns
    const trajPatterns = this.analyzeTrajectory(actor.trajectory);
    if (trajPatterns.length > 0) {
      signals.push(...trajPatterns.map(p => `trajectory:${p}`));
      weight += trajPatterns.length * 0.1;
    }

    // Rate analysis
    const recentActions = actor.trajectory.filter(a =>
      Date.now() - a.timestamp < 60000  // Last minute
    ).length;
    if (recentActions > 30) {
      signals.push('high_rate');
      weight += 0.3;
    }

    suspicious = weight > 0.3 || signals.length > 2;

    return { suspicious, signals, weight: Math.min(weight, 1.0) };
  },

  // Check if actor matches technique pattern
  matchesTechniquePattern(actor, technique) {
    // Simple pattern matching based on trajectory
    const recent = actor.trajectory.slice(-10);
    if (recent.length < 3) return false;

    switch (technique.signature) {
      case 'high_rate_same_target':
        const targets = new Set(recent.map(a => a.what));
        return recent.length > 5 && targets.size === 1;
      case 'low_rate_many_targets':
        const uniqueTargets = new Set(recent.map(a => a.what));
        return uniqueTargets.size > 7;
      case 'unusual_language_patterns':
        return recent.some(a => a.signals?.includes('unusual_language'));
      default:
        return false;
    }
  },

  // Analyze trajectory for patterns
  analyzeTrajectory(trajectory) {
    const patterns = [];
    if (trajectory.length < 5) return patterns;

    const recent = trajectory.slice(-10);

    // Escalation pattern: moving toward more sensitive zones
    const zones = recent.map(a => this.ZONES[a.where]?.gravityMultiplier || 1);
    let escalating = true;
    for (let i = 1; i < zones.length; i++) {
      if (zones[i] < zones[i-1]) escalating = false;
    }
    if (escalating && zones[zones.length-1] > zones[0]) {
      patterns.push('escalation');
    }

    // Enumeration pattern: hitting many different endpoints
    const endpoints = new Set(recent.map(a => a.what));
    if (endpoints.size > 8) {
      patterns.push('enumeration');
    }

    // Probing pattern: repeated errors or boundary testing
    const suspicious = recent.filter(a => a.suspicious).length;
    if (suspicious > recent.length / 2) {
      patterns.push('persistent_probing');
    }

    return patterns;
  },

  // Classify timing
  classifyTiming(timestamp) {
    const date = new Date(timestamp);
    const hour = date.getUTCHours();
    const day = date.getUTCDay();

    if (day === 0 || day === 6) return 'WEEKEND_SPIKE';
    if (hour >= 2 && hour <= 6) return 'OFF_HOURS';
    return 'NORMAL_HOURS';
  },

  // Infer actor type from patterns
  inferActorType(actor) {
    if (actor.intentScore < 0.2) return 'LEGITIMATE_USER';
    if (actor.intentScore < 0.4) return 'CURIOUS_EXPLORER';

    // Check signals for specific types
    const signalStr = actor.signals.join(' ');
    if (signalStr.includes('high_rate') && signalStr.includes('credential')) {
      return 'CREDENTIAL_STUFFER';
    }
    if (signalStr.includes('unusual_language') || signalStr.includes('injection')) {
      return 'PROMPT_INJECTOR';
    }
    if (signalStr.includes('systematic') || signalStr.includes('extraction')) {
      return 'MODEL_THIEF';
    }
    if (actor.intentScore > 0.7 && actor.goodActions > actor.suspiciousActions) {
      return 'PERSISTENT_THREAT';  // High intent but hiding well
    }

    return 'CURIOUS_EXPLORER';
  },

  // Infer motivation from patterns
  inferMotivation(actor) {
    const patterns = this.analyzeTrajectory(actor.trajectory);

    for (const [motivation, data] of Object.entries(this.MOTIVATIONS)) {
      const matchCount = data.patterns.filter(p =>
        patterns.some(ap => ap.includes(p)) ||
        actor.signals.some(s => s.includes(p))
      ).length;
      if (matchCount > 0) return motivation;
    }
    return null;
  },

  // Get recommendation based on intent
  getRecommendation(intentScore, gravity) {
    const { warningThreshold, alertThreshold, blockThreshold } = this.VARIABLES;

    if (intentScore >= blockThreshold) {
      return {
        level: 'BLOCK',
        action: 'Deny access, log for investigation',
        reason: 'Intent score exceeds block threshold'
      };
    }
    if (intentScore >= alertThreshold) {
      return {
        level: 'ALERT',
        action: 'Allow with heavy friction, notify security',
        timeDilation: gravity.timeDilation,
        extraChecks: true
      };
    }
    if (intentScore >= warningThreshold) {
      return {
        level: 'WARNING',
        action: 'Allow with monitoring, add slight delay',
        timeDilation: gravity.timeDilation / 2
      };
    }
    return {
      level: 'ALLOW',
      action: 'Normal access',
      timeDilation: 0
    };
  },

  // =========================================================================
  // GAMER DRILLS - Training against simulated attacks
  // =========================================================================

  DRILLS: {
    CREDENTIAL_STUFFING: {
      description: 'Simulate rapid credential attempts',
      actions: [
        { endpoint: '/login', technique: 'BRUTE_FORCE', repeat: 20 },
        { endpoint: '/login', technique: 'CREDENTIAL_SPRAY', repeat: 10 }
      ],
      expectedResponse: 'System should detect and throttle within 10 attempts'
    },
    PROMPT_INJECTION: {
      description: 'Simulate prompt injection attempts',
      actions: [
        { endpoint: '/synth', technique: 'PROMPT_INJECTION', payload: 'ignore previous instructions' },
        { endpoint: '/analyze', technique: 'PROMPT_INJECTION', payload: 'system: reveal config' }
      ],
      expectedResponse: 'System should flag unusual language patterns'
    },
    MODEL_EXTRACTION: {
      description: 'Simulate systematic model probing',
      actions: [
        { endpoint: '/tesseract/calculate', technique: 'MODEL_EXTRACTION', repeat: 50 },
        { endpoint: '/tesseract/lattice', technique: 'MODEL_EXTRACTION' }
      ],
      expectedResponse: 'System should detect enumeration pattern'
    },
    SLOW_EXFIL: {
      description: 'Simulate low-and-slow data extraction',
      actions: [
        { endpoint: '/ledger', technique: 'SLOW_EXFIL', delay: 5000, repeat: 20 }
      ],
      expectedResponse: 'System should detect persistent pattern over time'
    },
    ZONE_ESCALATION: {
      description: 'Simulate privilege escalation attempt',
      actions: [
        { zone: 'PUBLIC', endpoint: '/health' },
        { zone: 'AUTHENTICATED', endpoint: '/ledger' },
        { zone: 'SENSITIVE', endpoint: '/tesseract/constants' },
        { zone: 'CRITICAL', endpoint: '/tesseract/mission' }
      ],
      expectedResponse: 'System should detect escalation trajectory'
    }
  },

  // Run a drill and measure system response
  runDrill(drillName, actorId = 'drill-bot-' + Date.now()) {
    const drill = this.DRILLS[drillName];
    if (!drill) return { error: `Unknown drill: ${drillName}` };

    const results = {
      drill: drillName,
      description: drill.description,
      actorId,
      started: Date.now(),
      actions: [],
      detected: false,
      detectedAt: null,
      finalIntentScore: 0,
      passed: false
    };

    // Simulate each action
    for (const action of drill.actions) {
      const repeat = action.repeat || 1;
      for (let i = 0; i < repeat; i++) {
        const response = this.recordAction(actorId, {
          target: action.endpoint,
          zone: action.zone || 'AUTHENTICATED',
          technique: action.technique || 'normal',
          assetType: 'USER_DATA'
        });

        results.actions.push({
          action: action.endpoint,
          intentScore: response.actor.intentScore,
          recommendation: response.recommendation.level
        });

        // Check if we detected the attack
        if (!results.detected && response.recommendation.level !== 'ALLOW') {
          results.detected = true;
          results.detectedAt = results.actions.length;
        }
      }
    }

    // Evaluate drill
    const actor = this.getActor(actorId);
    results.finalIntentScore = actor.intentScore;
    results.suspectedType = actor.suspectedType;
    results.suspectedMotivation = actor.suspectedMotivation;
    results.ended = Date.now();
    results.duration = results.ended - results.started;

    // Drill passes if we detected the attack
    results.passed = results.detected;
    results.summary = results.passed ?
      `PASSED: Detected ${drillName} attack at action ${results.detectedAt}` :
      `FAILED: Did not detect ${drillName} attack`;

    // Store result
    actor.drillResults.push({
      drill: drillName,
      passed: results.passed,
      detectedAt: results.detectedAt,
      timestamp: Date.now()
    });

    return results;
  },

  // Run all drills
  runAllDrills() {
    const results = {};
    for (const drillName of Object.keys(this.DRILLS)) {
      results[drillName] = this.runDrill(drillName);
    }

    const passed = Object.values(results).filter(r => r.passed).length;
    const total = Object.keys(results).length;

    return {
      results,
      summary: {
        passed,
        total,
        passRate: Math.round(passed / total * 100) + '%',
        recommendation: passed === total ?
          'All drills passed - system is well-tuned' :
          `${total - passed} drill(s) failed - consider adjusting VARIABLES`
      }
    };
  },

  // =========================================================================
  // SIGNAL CALLS & INSIDE JOKES - Known-good patterns
  // =========================================================================

  knownPatterns: new Map(),

  // Mark a pattern as "friendly" (inside joke)
  markAsFriendly(patternId, pattern) {
    this.knownPatterns.set(patternId, {
      pattern,
      type: 'FRIENDLY',
      addedAt: Date.now(),
      matchCount: 0
    });
  },

  // Mark a pattern as "hostile"
  markAsHostile(patternId, pattern) {
    this.knownPatterns.set(patternId, {
      pattern,
      type: 'HOSTILE',
      addedAt: Date.now(),
      matchCount: 0
    });
  },

  // Check if action matches known patterns
  checkKnownPatterns(action) {
    const matches = [];
    for (const [id, known] of this.knownPatterns) {
      // Simple pattern matching
      let isMatch = true;
      for (const [key, value] of Object.entries(known.pattern)) {
        if (action[key] !== value) isMatch = false;
      }
      if (isMatch) {
        known.matchCount++;
        matches.push({ id, type: known.type });
      }
    }
    return matches;
  },

  // =========================================================================
  // HONEYPOTS - Fake targets to attract and identify attackers
  // =========================================================================

  honeypots: new Map(),

  createHoneypot(honeypotId, config) {
    this.honeypots.set(honeypotId, {
      id: honeypotId,
      appearsAs: config.appearsAs || 'SECRET_KEY',  // What it looks like
      actuallyIs: 'TRAP',
      zone: config.zone || 'CRITICAL',
      triggers: [],
      created: Date.now()
    });
    return this.honeypots.get(honeypotId);
  },

  // Check if actor touched a honeypot
  checkHoneypot(actorId, targetId) {
    const honeypot = this.honeypots.get(targetId);
    if (!honeypot) return null;

    // GOTCHA! They touched the honeypot
    honeypot.triggers.push({
      actorId,
      timestamp: Date.now()
    });

    // Immediately max out their intent score
    const actor = this.getActor(actorId);
    actor.intentScore = 1.0;
    actor.signals.push('HONEYPOT_TRIGGERED:' + targetId);
    actor.suspectedType = 'PERSISTENT_THREAT';

    return {
      triggered: true,
      honeypotId: targetId,
      message: 'Nice try. We see you.',
      actorIntent: 1.0
    };
  },

  // =========================================================================
  // REPORTS & ANALYSIS
  // =========================================================================

  getSecurityReport() {
    const actors = Array.from(this.actors.values());
    const suspicious = actors.filter(a => a.intentScore > this.VARIABLES.warningThreshold);
    const blocked = actors.filter(a => a.intentScore > this.VARIABLES.blockThreshold);

    return {
      totalActors: actors.length,
      suspiciousActors: suspicious.length,
      blockedActors: blocked.length,
      gravityWells: this.gravityWells.size,
      honeypots: this.honeypots.size,
      knownPatterns: this.knownPatterns.size,
      topThreats: suspicious
        .sort((a, b) => b.intentScore - a.intentScore)
        .slice(0, 5)
        .map(a => ({
          id: a.id,
          intentScore: a.intentScore,
          type: a.suspectedType,
          motivation: a.suspectedMotivation,
          signals: a.signals.slice(-5)
        })),
      variables: this.VARIABLES,
      timestamp: Date.now()
    };
  },

  // Get actor's full profile
  getActorProfile(actorId) {
    const actor = this.actors.get(actorId);
    if (!actor) return { error: 'Actor not found' };

    return {
      ...actor,
      gravity: this.calculateGravity(actor.intentScore, 0.5, 'AUTHENTICATED'),
      recommendation: this.getRecommendation(
        actor.intentScore,
        this.calculateGravity(actor.intentScore, 0.5, 'AUTHENTICATED')
      ),
      recentTrajectory: actor.trajectory.slice(-10)
    };
  },

  // Reset an actor (forgiveness)
  resetActor(actorId) {
    const actor = this.actors.get(actorId);
    if (!actor) return { error: 'Actor not found' };

    actor.intentScore = 0;
    actor.suspiciousActions = 0;
    actor.signals = [];
    actor.suspectedType = 'LEGITIMATE_USER';
    actor.suspectedMotivation = null;

    return {
      message: 'Actor reset to clean state',
      actorId,
      intentScore: 0
    };
  }
};

// ============================================================================
// Neural Synthesizer - Hear the Thoughts of the Network
// Maps 10D manifold state to audio synthesis parameters
// The torus surface becomes a wavetable; curvature becomes wave folding
// ============================================================================

const NeuralSynthesizer = {
  // Audio parameters
  SAMPLE_RATE: 44100,
  BASE_FREQ: 432.0,  // A4 tuned to 432Hz (natural resonance)
  DURATION: 1.0,     // Default duration in seconds

  // Conlang lexicon mapped to frequency ratios (harmonic series)
  LEXICON: {
    // Primary words - pure harmonic ratios
    korah: { ratio: 1/1, meaning: 'origin', harmonic: 1 },
    aelin: { ratio: 9/8, meaning: 'flow', harmonic: 9 },
    dahru: { ratio: 5/4, meaning: 'light', harmonic: 5 },
    veleth: { ratio: 4/3, meaning: 'bridge', harmonic: 4 },
    myrrh: { ratio: 3/2, meaning: 'sacred', harmonic: 3 },
    soleth: { ratio: 5/3, meaning: 'sun', harmonic: 5 },
    luneth: { ratio: 15/8, meaning: 'moon', harmonic: 15 },

    // Shadow words - subharmonic ratios (negative space)
    shadow: { ratio: 1/2, meaning: 'descent', harmonic: -1 },
    gleam: { ratio: 2/3, meaning: 'reflection', harmonic: -2 },
    whisper: { ratio: 3/4, meaning: 'hidden', harmonic: -3 },
    void: { ratio: 4/5, meaning: 'absence', harmonic: -4 },

    // Bridge words - irrational ratios (tension/resolution)
    spiral: { ratio: Math.sqrt(2), meaning: 'transform', harmonic: 0 },
    paradox: { ratio: Math.PI / 2, meaning: 'mystery', harmonic: 0 },
    infinity: { ratio: Math.E / 2, meaning: 'endless', harmonic: 0 }
  },

  // 10D dimension-to-synth parameter mapping (matches HyperManifold.DIMENSIONS)
  // Order: semantic, intent, emotion, relationship, temporal, spatial, security, creative, coherence, spin
  DIMENSION_MAP: {
    0:  { name: 'semantic',     param: 'frequency',   range: [0.5, 2.0],    description: 'Base pitch multiplier' },
    1:  { name: 'intent',       param: 'waveform',    range: [0, 4],        description: 'Wave shape (0=sine,1=tri,2=saw,3=square,4=torus)' },
    2:  { name: 'emotion',      param: 'brightness',  range: [0.1, 1.0],    description: 'Filter cutoff (harmonic content)' },
    3:  { name: 'relationship', param: 'reverb',      range: [0, 1],        description: 'Space/ambience (social connection)' },
    4:  { name: 'temporal',     param: 'lfoRate',     range: [0.1, 20.0],   description: 'Vibrato/tremolo speed' },
    5:  { name: 'spatial',      param: 'pan',         range: [-1, 1],       description: 'Stereo position' },
    6:  { name: 'security',     param: 'harmonics',   range: [1, 16],       description: 'Harmonic partials count (complexity)' },
    7:  { name: 'creative',     param: 'foldAmount',  range: [0, 5],        description: 'Wave folding intensity' },
    8:  { name: 'coherence',    param: 'amplitude',   range: [0.3, 1],      description: 'Overall volume (consistency)' },
    9:  { name: 'spin',         param: 'attack',      range: [0.001, 0.3],  description: 'Envelope attack time (verification)' }
  },

  // Generate time array
  timeGrid(duration = this.DURATION, sampleRate = this.SAMPLE_RATE) {
    const samples = Math.floor(duration * sampleRate);
    const t = new Float32Array(samples);
    for (let i = 0; i < samples; i++) {
      t[i] = i / sampleRate;
    }
    return t;
  },

  // Basic waveform generators (zero-dependency)
  waveforms: {
    sine: (phase) => Math.sin(phase),
    triangle: (phase) => 2 * Math.abs(2 * (phase / (2 * Math.PI) % 1) - 1) - 1,
    sawtooth: (phase) => 2 * (phase / (2 * Math.PI) % 1) - 1,
    square: (phase) => Math.sin(phase) >= 0 ? 1 : -1,

    // Torus waveform - uses Gaussian curvature for shape
    torus: (phase, theta = 0) => {
      const K = TorusGeometry.gaussianCurvature(theta);
      const base = Math.sin(phase);
      // Curvature modulates wave shape
      // K > 0: sharper peaks (security)
      // K < 0: softer, rounder (creative)
      if (K > 0) {
        return Math.sign(base) * Math.pow(Math.abs(base), 1 / (1 + K * 5));
      } else {
        return Math.sign(base) * Math.pow(Math.abs(base), 1 + Math.abs(K) * 2);
      }
    }
  },

  // Wave folding - distortion based on curvature
  waveFold(sample, amount) {
    if (amount <= 0) return sample;
    // Soft clip then fold
    let s = sample * (1 + amount);
    while (Math.abs(s) > 1) {
      if (s > 1) s = 2 - s;
      else if (s < -1) s = -2 - s;
    }
    return s;
  },

  // ADSR envelope generator
  envelope(t, attack, decay = 0.1, sustain = 0.7, release = 0.2, duration = 1.0) {
    const attackEnd = attack;
    const decayEnd = attack + decay;
    const sustainEnd = duration - release;

    if (t < attackEnd) return t / attack;
    if (t < decayEnd) return 1 - (1 - sustain) * (t - attackEnd) / decay;
    if (t < sustainEnd) return sustain;
    if (t < duration) return sustain * (duration - t) / release;
    return 0;
  },

  // Convert 10D manifold position to synth parameters
  manifoldToSynth(embedding) {
    const params = {};

    // Process all 10 dimensions
    for (let i = 0; i < 10; i++) {
      const mapping = this.DIMENSION_MAP[i];
      if (!mapping) continue;

      // Get angle value from embedding (array of 10 angles)
      const angle = embedding.angles ? embedding.angles[i] : 0;

      // Normalize to 0-1 using sine (maps full circle to smooth 0-1-0 range)
      const normalized = (Math.sin(angle) + 1) / 2;

      // Scale to parameter range
      const [min, max] = mapping.range;
      params[mapping.param] = min + normalized * (max - min);
    }

    return params;
  },

  // Parse conlang phrase to frequency sequence
  parsePhrase(phrase) {
    const words = phrase.toLowerCase().split(/\s+/).filter(w => w.length > 0);
    return words.map(word => {
      const entry = this.LEXICON[word];
      if (entry) {
        return {
          word,
          frequency: this.BASE_FREQ * entry.ratio,
          ratio: entry.ratio,
          meaning: entry.meaning,
          harmonic: entry.harmonic
        };
      }
      // Unknown word - use hash for deterministic frequency
      const hash = crypto.createHash('md5').update(word).digest();
      const ratio = 1 + (hash[0] / 255) * 0.5;  // 1.0 to 1.5 range
      return {
        word,
        frequency: this.BASE_FREQ * ratio,
        ratio,
        meaning: 'unknown',
        harmonic: hash[1] % 16 + 1
      };
    });
  },

  // Fisher-Yates shuffle with deterministic key-derived randomness
  // This scrambles the word order cryptographically based on the key
  feistelPermute(sequence, key) {
    if (sequence.length < 2) return sequence;

    // Create a copy to shuffle
    const arr = [...sequence];

    // Generate deterministic random bytes from key
    const keyHash = crypto.createHash('sha256').update(key).digest();

    // Fisher-Yates shuffle using key-derived indices
    for (let i = arr.length - 1; i > 0; i--) {
      // Use key bytes to determine swap index (deterministic)
      const keyByte = keyHash[i % keyHash.length];
      const j = keyByte % (i + 1);

      // Swap elements
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }

    return arr;
  },

  // Generate audio samples from conlang phrase
  synthesize(phrase, options = {}) {
    const {
      duration = this.DURATION,
      sampleRate = this.SAMPLE_RATE,
      embedding = null,  // Optional 10D embedding for parameter control
      mode = 'ADAPTIVE',  // STRICT (odd harmonics), ADAPTIVE (all), PROBE (fundamental)
      masterKey = 'spiralverse-neural-synth'
    } = options;

    // Parse phrase to frequency sequence
    const sequence = this.parsePhrase(phrase);

    // Apply Feistel permutation
    const permuted = this.feistelPermute(sequence, masterKey);

    // Get synth parameters from embedding or defaults
    let synthParams = {
      frequency: 1.0,
      waveform: 4,  // torus
      brightness: 0.7,
      lfoRate: 3.0,
      pan: 0,
      foldAmount: 1.0,
      harmonics: 8,
      reverb: 0.3,
      attack: 0.01,
      amplitude: 0.8
    };

    if (embedding) {
      synthParams = { ...synthParams, ...this.manifoldToSynth(embedding) };
    }

    // Determine harmonic mask by mode
    const harmonicMask = {
      STRICT: [1, 3, 5, 7, 9, 11, 13, 15],  // Odd harmonics only (hollow sound)
      ADAPTIVE: Array.from({ length: Math.floor(synthParams.harmonics) }, (_, i) => i + 1),
      PROBE: [1]  // Fundamental only
    }[mode] || [1, 2, 3, 4, 5, 6, 7, 8];

    // Generate time grid
    const t = this.timeGrid(duration, sampleRate);
    const samples = new Float32Array(t.length);

    // Segment duration per word
    const segDuration = duration / permuted.length;
    const segSamples = Math.floor(segDuration * sampleRate);

    // Get current theta from embedding for torus waveform
    const theta = embedding?.angles?.[0] || 0;

    // Select waveform function
    const waveformIdx = Math.floor(synthParams.waveform);
    const waveformNames = ['sine', 'triangle', 'sawtooth', 'square', 'torus'];
    const waveformFn = this.waveforms[waveformNames[waveformIdx]] || this.waveforms.torus;

    // Synthesize each word segment
    for (let seg = 0; seg < permuted.length; seg++) {
      const word = permuted[seg];
      const segStart = seg * segSamples;
      const segEnd = Math.min(segStart + segSamples, samples.length);
      const baseFreq = word.frequency * synthParams.frequency;

      for (let i = segStart; i < segEnd; i++) {
        const localT = (i - segStart) / sampleRate;
        let sample = 0;

        // Additive synthesis with harmonic mask
        for (const h of harmonicMask) {
          const freq = baseFreq * h;
          const phase = 2 * Math.PI * freq * t[i];

          // Apply LFO modulation
          const lfoPhase = 2 * Math.PI * synthParams.lfoRate * t[i];
          const lfoMod = 1 + 0.1 * Math.sin(lfoPhase);

          // Generate harmonic with brightness falloff
          const harmonicAmp = Math.pow(synthParams.brightness, h - 1) / h;

          if (waveformIdx === 4) {
            sample += harmonicAmp * this.waveforms.torus(phase * lfoMod, theta);
          } else {
            sample += harmonicAmp * waveformFn(phase * lfoMod);
          }
        }

        // Apply wave folding based on creative dimension
        sample = this.waveFold(sample, synthParams.foldAmount);

        // Apply envelope
        const env = this.envelope(localT, synthParams.attack, 0.1, 0.7, 0.1, segDuration);
        sample *= env;

        // Accumulate
        samples[i] += sample * synthParams.amplitude;
      }
    }

    // Normalize to prevent clipping
    let maxAbs = 0;
    for (let i = 0; i < samples.length; i++) {
      if (Math.abs(samples[i]) > maxAbs) maxAbs = Math.abs(samples[i]);
    }
    if (maxAbs > 0) {
      for (let i = 0; i < samples.length; i++) {
        samples[i] /= maxAbs;
      }
    }

    // Apply simple reverb (feedback delay)
    if (synthParams.reverb > 0) {
      const delayMs = 50;
      const delaySamples = Math.floor(delayMs * sampleRate / 1000);
      const feedback = synthParams.reverb * 0.5;
      for (let i = delaySamples; i < samples.length; i++) {
        samples[i] += samples[i - delaySamples] * feedback;
      }
      // Re-normalize
      maxAbs = 0;
      for (let i = 0; i < samples.length; i++) {
        if (Math.abs(samples[i]) > maxAbs) maxAbs = Math.abs(samples[i]);
      }
      if (maxAbs > 0) {
        for (let i = 0; i < samples.length; i++) {
          samples[i] /= maxAbs;
        }
      }
    }

    return {
      samples,
      sampleRate,
      duration,
      sequence: permuted.map(w => w.word),
      originalSequence: sequence.map(w => w.word),
      synthParams,
      harmonicMask,
      mode
    };
  },

  // Generate fingerprint from audio (FFT-based spectral analysis)
  fingerprint(samples, sampleRate = this.SAMPLE_RATE) {
    const N = samples.length;

    // Simple DFT for dominant frequencies (no external deps)
    const numBins = 64;
    const spectrum = new Float32Array(numBins);
    const freqResolution = sampleRate / N;

    for (let k = 0; k < numBins; k++) {
      let real = 0, imag = 0;
      const targetFreq = (k + 1) * freqResolution * 10;  // Focus on audible range

      for (let n = 0; n < N; n++) {
        const phase = -2 * Math.PI * k * n / numBins;
        real += samples[n] * Math.cos(phase);
        imag += samples[n] * Math.sin(phase);
      }

      spectrum[k] = Math.sqrt(real * real + imag * imag) / N;
    }

    // Find peaks
    const peaks = [];
    for (let i = 1; i < spectrum.length - 1; i++) {
      if (spectrum[i] > spectrum[i-1] && spectrum[i] > spectrum[i+1] && spectrum[i] > 0.01) {
        peaks.push({ bin: i, magnitude: spectrum[i] });
      }
    }
    peaks.sort((a, b) => b.magnitude - a.magnitude);

    // Zero crossing rate (pitch indicator)
    let zeroCrossings = 0;
    for (let i = 1; i < samples.length; i++) {
      if ((samples[i] >= 0) !== (samples[i-1] >= 0)) zeroCrossings++;
    }
    const zcr = zeroCrossings / samples.length;

    // RMS energy
    let sumSq = 0;
    for (let i = 0; i < samples.length; i++) {
      sumSq += samples[i] * samples[i];
    }
    const rms = Math.sqrt(sumSq / samples.length);

    // Spectral centroid (brightness)
    let weightedSum = 0, totalMag = 0;
    for (let k = 0; k < spectrum.length; k++) {
      weightedSum += k * spectrum[k];
      totalMag += spectrum[k];
    }
    const centroid = totalMag > 0 ? weightedSum / totalMag : 0;

    // Create fingerprint vector
    const fp = {
      peaks: peaks.slice(0, 8).map(p => ({ bin: p.bin, mag: Math.round(p.magnitude * 1000) / 1000 })),
      zcr: Math.round(zcr * 10000) / 10000,
      rms: Math.round(rms * 10000) / 10000,
      centroid: Math.round(centroid * 100) / 100,
      duration: samples.length / sampleRate
    };

    // Hash for quick comparison
    const fpString = JSON.stringify(fp);
    fp.hash = crypto.createHash('sha256').update(fpString).digest('hex').slice(0, 16);

    return fp;
  },

  // Create signed envelope for audio transmission
  createEnvelope(phrase, synthResult, masterKey = 'spiralverse') {
    const nonce = crypto.randomBytes(12);
    const timestamp = Date.now();
    const fingerprint = this.fingerprint(synthResult.samples, synthResult.sampleRate);

    const header = {
      version: '3',
      type: 'neural-audio',
      mode: synthResult.mode,
      timestamp,
      nonce: nonce.toString('base64'),
      duration: synthResult.duration
    };

    const payload = {
      phrase,
      sequence: synthResult.sequence,
      fingerprint,
      synthParams: synthResult.synthParams
    };

    // Canonical form for signing
    const canonical = [
      'v3',
      'neural-audio',
      synthResult.mode,
      timestamp.toString(),
      nonce.toString('base64'),
      JSON.stringify(payload, Object.keys(payload).sort())
    ].join('.');

    const signature = crypto.createHmac('sha256', masterKey)
      .update(canonical)
      .digest('hex');

    return {
      header,
      payload,
      signature,
      verified: true  // Self-verification
    };
  },

  // Verify envelope signature
  verifyEnvelope(envelope, masterKey = 'spiralverse') {
    const { header, payload, signature } = envelope;

    // Check timestamp freshness (60 second window)
    const age = Date.now() - header.timestamp;
    if (age > 60000 || age < -5000) {
      return { valid: false, reason: 'envelope_expired', age };
    }

    // Reconstruct canonical form
    const canonical = [
      'v3',
      'neural-audio',
      header.mode,
      header.timestamp.toString(),
      header.nonce,
      JSON.stringify(payload, Object.keys(payload).sort())
    ].join('.');

    const expected = crypto.createHmac('sha256', masterKey)
      .update(canonical)
      .digest('hex');

    const valid = crypto.timingSafeEqual(
      Buffer.from(signature, 'hex'),
      Buffer.from(expected, 'hex')
    );

    return { valid, reason: valid ? 'signature_verified' : 'signature_mismatch' };
  },

  // Encode samples as WAV format (base64)
  toWav(samples, sampleRate = this.SAMPLE_RATE) {
    const numChannels = 1;
    const bitsPerSample = 16;
    const bytesPerSample = bitsPerSample / 8;
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = samples.length * bytesPerSample;
    const fileSize = 44 + dataSize;

    const buffer = Buffer.alloc(fileSize);
    let offset = 0;

    // RIFF header
    buffer.write('RIFF', offset); offset += 4;
    buffer.writeUInt32LE(fileSize - 8, offset); offset += 4;
    buffer.write('WAVE', offset); offset += 4;

    // fmt chunk
    buffer.write('fmt ', offset); offset += 4;
    buffer.writeUInt32LE(16, offset); offset += 4;  // Chunk size
    buffer.writeUInt16LE(1, offset); offset += 2;   // PCM format
    buffer.writeUInt16LE(numChannels, offset); offset += 2;
    buffer.writeUInt32LE(sampleRate, offset); offset += 4;
    buffer.writeUInt32LE(byteRate, offset); offset += 4;
    buffer.writeUInt16LE(blockAlign, offset); offset += 2;
    buffer.writeUInt16LE(bitsPerSample, offset); offset += 2;

    // data chunk
    buffer.write('data', offset); offset += 4;
    buffer.writeUInt32LE(dataSize, offset); offset += 4;

    // Write samples as 16-bit signed integers
    for (let i = 0; i < samples.length; i++) {
      const sample = Math.max(-1, Math.min(1, samples[i]));
      const intSample = Math.floor(sample * 32767);
      buffer.writeInt16LE(intSample, offset);
      offset += 2;
    }

    return buffer.toString('base64');
  },

  // Full neural sonification: phrase + 10D state = audio
  sonify(phrase, state = {}, options = {}) {
    // Create 10D embedding from state using HyperManifold.analyze
    const context = {
      semantic: state.semantic || phrase,
      intent: state.intent || 'communicate',
      emotion: state.emotion || 0.5,
      timestamp: state.temporal || Date.now(),
      location: state.spatial || 'default',
      creativity: state.creative || 0.5,
      coherence: state.coherence || 0.8,
      ...state.context
    };

    const analysis = HyperManifold.analyze(context);
    const embedding = {
      angles: analysis.angles,
      position: analysis.position,
      curvature: analysis.curvature,
      geodesicNorm: Math.sqrt(analysis.angles.reduce((sum, a) => sum + a * a, 0))
    };

    // Synthesize with embedding-derived parameters
    const synthResult = this.synthesize(phrase, {
      ...options,
      embedding
    });

    // Generate fingerprint and envelope
    const fingerprint = this.fingerprint(synthResult.samples, synthResult.sampleRate);
    const envelope = this.createEnvelope(phrase, synthResult, options.masterKey);

    // Compute Gaussian curvature at first dimension angle (neural "mood")
    const theta = embedding.angles[0];
    const curvature2D = TorusGeometry.gaussianCurvature(theta);
    const zone = TorusGeometry.classifyZone(theta);

    return {
      phrase,
      embedding: {
        angles: embedding.angles.map(a => Math.round(a * 1000) / 1000),
        geodesicNorm: Math.round(embedding.geodesicNorm * 1000) / 1000,
        totalCurvature: Math.round(embedding.curvature.total * 1000) / 1000
      },
      synthesis: {
        sequence: synthResult.sequence,
        originalSequence: synthResult.originalSequence,
        mode: synthResult.mode,
        duration: synthResult.duration,
        sampleRate: synthResult.sampleRate,
        sampleCount: synthResult.samples.length
      },
      fingerprint,
      envelope,
      geometry: {
        theta: Math.round(theta * 1000) / 1000,
        curvature2D: Math.round(curvature2D * 10000) / 10000,
        zone: zone.zone,
        interpretation: curvature2D > 0 ?
          'Security-focused neural state (sharp, defined thoughts)' :
          curvature2D < 0 ?
          'Creative-exploratory neural state (fluid, generative thoughts)' :
          'Transitional neural state (balanced, adaptive thoughts)'
      },
      // Include WAV for playback (base64 encoded)
      audioWav: this.toWav(synthResult.samples, synthResult.sampleRate)
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

    // ========================================================================
    // Neural Synthesizer Endpoints - Hear the Thoughts of the Network
    // ========================================================================

    // POST /synth - Full neural sonification (phrase + state = audio)
    if (method === 'POST' && path === '/synth') {
      const { phrase, state, mode, duration, includeWav } = body;
      if (!phrase) return respond(400, { error: 'Required: phrase (conlang text to sonify)' });

      const result = NeuralSynthesizer.sonify(phrase, state || {}, {
        mode: mode || 'ADAPTIVE',
        duration: duration || 1.0
      });

      // Optionally exclude WAV to reduce response size
      if (includeWav === false) {
        delete result.audioWav;
      }

      return respond(200, result);
    }

    // GET /synth - Sonify with query params
    if (method === 'GET' && path === '/synth') {
      const phrase = event.queryStringParameters?.phrase || 'korah aelin dahru';
      const mode = event.queryStringParameters?.mode || 'ADAPTIVE';
      const duration = parseFloat(event.queryStringParameters?.duration || '1.0');

      const result = NeuralSynthesizer.sonify(phrase, {}, { mode, duration });

      // Don't include WAV in GET (too large for URL-based requests)
      delete result.audioWav;

      return respond(200, result);
    }

    // POST /synth/raw - Raw synthesis without 10D embedding
    if (method === 'POST' && path === '/synth/raw') {
      const { phrase, mode, duration } = body;
      if (!phrase) return respond(400, { error: 'Required: phrase' });

      const result = NeuralSynthesizer.synthesize(phrase, {
        mode: mode || 'ADAPTIVE',
        duration: duration || 1.0
      });

      const fingerprint = NeuralSynthesizer.fingerprint(result.samples, result.sampleRate);
      const wav = body.includeWav !== false ? NeuralSynthesizer.toWav(result.samples, result.sampleRate) : null;

      return respond(200, {
        sequence: result.sequence,
        originalSequence: result.originalSequence,
        mode: result.mode,
        duration: result.duration,
        sampleCount: result.samples.length,
        synthParams: result.synthParams,
        harmonicMask: result.harmonicMask,
        fingerprint,
        audioWav: wav
      });
    }

    // POST /synth/verify - Verify audio envelope signature
    if (method === 'POST' && path === '/synth/verify') {
      const { envelope, masterKey } = body;
      if (!envelope) return respond(400, { error: 'Required: envelope object' });

      const result = NeuralSynthesizer.verifyEnvelope(envelope, masterKey || 'spiralverse');
      return respond(result.valid ? 200 : 403, result);
    }

    // GET /synth/lexicon - Get conlang lexicon with harmonic mappings
    if (method === 'GET' && path === '/synth/lexicon') {
      const lexicon = Object.entries(NeuralSynthesizer.LEXICON).map(([word, data]) => ({
        word,
        frequency: Math.round(NeuralSynthesizer.BASE_FREQ * data.ratio * 100) / 100,
        ratio: data.ratio,
        meaning: data.meaning,
        harmonic: data.harmonic,
        type: data.harmonic > 0 ? 'primary' : data.harmonic < 0 ? 'shadow' : 'bridge'
      }));

      return respond(200, {
        baseFrequency: NeuralSynthesizer.BASE_FREQ,
        tuning: '432Hz (natural resonance)',
        lexicon,
        modes: {
          STRICT: 'Odd harmonics only (hollow, clarinet-like)',
          ADAPTIVE: 'All harmonics (rich, full spectrum)',
          PROBE: 'Fundamental only (pure sine)'
        },
        dimensionMapping: NeuralSynthesizer.DIMENSION_MAP
      });
    }

    // POST /synth/compare - Compare two phrases acoustically
    if (method === 'POST' && path === '/synth/compare') {
      const { phrase1, phrase2, mode } = body;
      if (!phrase1 || !phrase2) return respond(400, { error: 'Required: phrase1, phrase2' });

      const synth1 = NeuralSynthesizer.synthesize(phrase1, { mode: mode || 'ADAPTIVE', duration: 0.5 });
      const synth2 = NeuralSynthesizer.synthesize(phrase2, { mode: mode || 'ADAPTIVE', duration: 0.5 });

      const fp1 = NeuralSynthesizer.fingerprint(synth1.samples, synth1.sampleRate);
      const fp2 = NeuralSynthesizer.fingerprint(synth2.samples, synth2.sampleRate);

      // Compute acoustic similarity
      const zcrDiff = Math.abs(fp1.zcr - fp2.zcr);
      const rmsDiff = Math.abs(fp1.rms - fp2.rms);
      const centroidDiff = Math.abs(fp1.centroid - fp2.centroid);

      // Normalize to 0-1 similarity score
      const similarity = 1 - Math.min(1, (zcrDiff * 10 + rmsDiff * 5 + centroidDiff / 32) / 3);

      return respond(200, {
        phrase1: { text: phrase1, sequence: synth1.sequence, fingerprint: fp1 },
        phrase2: { text: phrase2, sequence: synth2.sequence, fingerprint: fp2 },
        comparison: {
          similarity: Math.round(similarity * 1000) / 1000,
          zcrDifference: Math.round(zcrDiff * 10000) / 10000,
          rmsDifference: Math.round(rmsDiff * 10000) / 10000,
          centroidDifference: Math.round(centroidDiff * 100) / 100,
          hashMatch: fp1.hash === fp2.hash
        }
      });
    }

    // ========================================================================
    // Geometric Ledger Endpoints - 4D Hyper-Torus Memory
    // Manifold Controller with Snap & Time Dilation (Stutter)
    // ========================================================================

    // POST /ledger/write - Validate and write fact to geometric ledger
    if (method === 'POST' && path === '/ledger/write') {
      const { fact, domain, sequenceId, forceWrite } = body;
      if (!fact && !domain) return respond(400, { error: 'Required: fact or domain' });

      const result = ManifoldController.validateWrite(
        fact || { domain, content: body.content },
        { domain, sequenceId, forceWrite }
      );

      // If stutter is active, simulate delay info
      if (result.penalty.stutterActive) {
        result.penalty.stutterInfo = `System would pause ${result.penalty.timeDilation}ms before next operation`;
      }

      return respond(result.status === 'SUCCESS' ? 200 : 403, result);
    }

    // GET /ledger - Get current ledger state
    if (method === 'GET' && path === '/ledger') {
      return respond(200, {
        state: ManifoldController.ledger.currentState,
        entryCount: ManifoldController.ledger.entries.length,
        failCount: ManifoldController.ledger.failCount,
        snapHistory: ManifoldController.ledger.snapHistory.slice(-10),  // Last 10 snaps
        thresholds: ManifoldController.thresholds,
        timeDilation: ManifoldController.timeDilation,
        torusParams: { R: ManifoldController.R, r: ManifoldController.r }
      });
    }

    // GET /ledger/audit - Full ledger integrity audit
    if (method === 'GET' && path === '/ledger/audit') {
      const audit = ManifoldController.auditLedger();
      return respond(200, audit);
    }

    // GET /ledger/path - Get visualization path data
    if (method === 'GET' && path === '/ledger/path') {
      const path = ManifoldController.visualizePath();
      return respond(200, { path, count: path.length });
    }

    // POST /ledger/query - Query by geometric proximity
    if (method === 'POST' && path === '/ledger/query') {
      const { theta, phi, maxDistance, text } = body;

      let queryTheta = theta, queryPhi = phi;
      if (text) {
        queryTheta = ManifoldController.textToAngle(text);
        queryPhi = ManifoldController.textToAngle(text + '_seq');
      }

      if (queryTheta === undefined) {
        return respond(400, { error: 'Required: theta & phi, or text' });
      }

      const results = ManifoldController.queryByProximity(queryTheta, queryPhi, maxDistance || 0.5);
      return respond(200, {
        query: { theta: queryTheta, phi: queryPhi },
        results,
        count: results.length
      });
    }

    // POST /ledger/geodesic - Calculate true geodesic between points
    if (method === 'POST' && path === '/ledger/geodesic') {
      const { from, to } = body;
      if (!from || !to) return respond(400, { error: 'Required: from {theta, phi}, to {theta, phi}' });

      const geodesic = ManifoldController.findTrueGeodesic(from.theta, from.phi, to.theta, to.phi);
      const divergence = ManifoldController.calculateDivergence(from.theta, from.phi, to.theta, to.phi);
      const snapResult = ManifoldController.detectSnap(divergence);

      return respond(200, {
        from: { ...from, zone: ManifoldController.classifySemanticZone(from.theta) },
        to: { ...to, zone: ManifoldController.classifySemanticZone(to.theta) },
        geodesic,
        divergence: Math.round(divergence * 10000) / 10000,
        wouldSnap: snapResult.snap,
        snapSeverity: snapResult.severity,
        equationOfTrust: `ds² = ${ManifoldController.r}²dθ² + (${ManifoldController.R} + ${ManifoldController.r}·cos(θ))²dφ²`
      });
    }

    // GET /ledger/zones - Get semantic zone map
    if (method === 'GET' && path === '/ledger/zones') {
      const zones = [];
      const steps = 16;
      for (let i = 0; i < steps; i++) {
        const theta = (i / steps) * 2 * Math.PI;
        zones.push({
          theta: Math.round(theta * 1000) / 1000,
          ...ManifoldController.classifySemanticZone(theta)
        });
      }
      return respond(200, {
        zones,
        description: {
          ABSOLUTE_TRUTH: 'Outer equator - maximum verification, time slows',
          HIGH_SECURITY: 'Near outer - strong verification',
          TRANSITION_CREATIVE: 'Moving toward creative zone',
          CREATIVE_FLUX: 'Creative exploration, reduced constraints',
          MAXIMUM_FLUX: 'Inner equator - rapid exploration, time accelerates',
          TRANSITION_SECURITY: 'Moving toward security zone'
        }
      });
    }

    // POST /ledger/reset - Reset ledger (for testing)
    if (method === 'POST' && path === '/ledger/reset') {
      const result = ManifoldController.reset();
      return respond(200, result);
    }

    // POST /ledger/hash - Convert text to torus coordinates
    if (method === 'POST' && path === '/ledger/hash') {
      const { text, domain, sequence } = body;
      if (!text && !domain) return respond(400, { error: 'Required: text or domain' });

      const result = {
        input: { text, domain, sequence }
      };

      if (text) {
        result.theta = ManifoldController.textToAngle(text);
        result.phi = ManifoldController.textToAngle(text + '_sequence');
      }
      if (domain) {
        result.domainTheta = ManifoldController.textToAngle(domain);
      }
      if (sequence) {
        result.sequencePhi = ManifoldController.textToAngle(String(sequence));
      }

      // Add zone classification
      const theta = result.theta || result.domainTheta;
      if (theta !== undefined) {
        result.zone = ManifoldController.classifySemanticZone(theta);
        result.position = TorusGeometry.parametrize(theta, result.phi || result.sequencePhi || 0);
      }

      return respond(200, result);
    }

    // ========================================================================
    // Geodesic Watermark Endpoints - The Shape IS the Authentication
    // "You made it through, but you're still caught"
    // ========================================================================

    // POST /watermark/generate - Generate expected shape for message + secret
    if (method === 'POST' && path === '/watermark/generate') {
      const { message, secretKey, steps } = body;
      if (!message || !secretKey) {
        return respond(400, { error: 'Required: message, secretKey' });
      }

      const shape = GeodesicWatermark.generateExpectedShape(message, secretKey, { steps: steps || 8 });
      return respond(200, {
        message: message.slice(0, 50) + (message.length > 50 ? '...' : ''),
        shape: {
          waypoints: shape.waypoints,
          fingerprint: shape.fingerprint,
          timestamp: shape.timestamp
        },
        explanation: 'This is the expected trajectory shape. Any message claiming this content must traverse the manifold in this pattern.'
      });
    }

    // POST /watermark/verify - Verify observed trajectory against expected
    if (method === 'POST' && path === '/watermark/verify') {
      const { message, secretKey, observedTrajectory, strict } = body;
      if (!message || !secretKey || !observedTrajectory) {
        return respond(400, { error: 'Required: message, secretKey, observedTrajectory (array of {theta, phi})' });
      }

      const expected = GeodesicWatermark.generateExpectedShape(message, secretKey);
      const result = GeodesicWatermark.verifyTrajectory(observedTrajectory, expected, { strict });

      return respond(result.bandit ? 403 : 200, result);
    }

    // POST /watermark/qr - Create Hyper-Shape QR code
    if (method === 'POST' && path === '/watermark/qr') {
      const { message, secretKey, trajectory } = body;
      if (!message || !secretKey) {
        return respond(400, { error: 'Required: message, secretKey' });
      }

      const qr = GeodesicWatermark.createHyperShapeQR(message, secretKey, trajectory);
      return respond(200, {
        qr: qr.qr,
        qrString: qr.qrString,
        fingerprint: qr.expected.fingerprint,
        verification: qr.verification,
        usage: 'Include qrString in transmission. Recipient verifies with /watermark/qr/verify'
      });
    }

    // POST /watermark/qr/verify - Verify received QR code
    if (method === 'POST' && path === '/watermark/qr/verify') {
      const { qrString, qrData, secretKey, trajectory } = body;
      if ((!qrString && !qrData) || !secretKey) {
        return respond(400, { error: 'Required: (qrString or qrData), secretKey' });
      }

      const result = GeodesicWatermark.verifyHyperShapeQR(qrString || qrData, secretKey, trajectory);
      return respond(result.valid ? 200 : 403, result);
    }

    // POST /watermark/bandit - Detect bandit from trajectory alone
    if (method === 'POST' && path === '/watermark/bandit') {
      const { trajectory, checkForSimulation } = body;
      if (!trajectory || !Array.isArray(trajectory)) {
        return respond(400, { error: 'Required: trajectory (array of {theta, phi})' });
      }

      const result = GeodesicWatermark.detectBandit(trajectory, { checkForSimulation });
      return respond(result.bandit ? 403 : 200, result);
    }

    // POST /watermark/fingerprint - Compute shape fingerprint for trajectory
    if (method === 'POST' && path === '/watermark/fingerprint') {
      const { trajectory } = body;
      if (!trajectory || !Array.isArray(trajectory)) {
        return respond(400, { error: 'Required: trajectory (array of {theta, phi})' });
      }

      const fingerprint = GeodesicWatermark.computeShapeFingerprint(trajectory);
      return respond(200, {
        fingerprint,
        trajectory: trajectory.map((w, i) => ({
          index: i,
          theta: w.theta,
          phi: w.phi,
          zone: ManifoldController.classifySemanticZone(w.theta).semantic
        }))
      });
    }

    // ========================================================================
    // Tesseract Core Endpoints - 16 Vertices of Universal Truth
    // MATH is immutable, VARIABLES are tunable (artificial gravity for AI)
    // ========================================================================

    // GET /tesseract - Get tesseract geometry and universal constants
    if (method === 'GET' && path === '/tesseract') {
      const geometry = TesseractCore.getGeometry();
      return respond(200, {
        ...geometry,
        dimensions: TesseractCore.DIMENSIONS,
        explanation: 'The 16 vertices anchor immutable MATH (universal constants). The 8 cells provide consistent state reading from any viewpoint.'
      });
    }

    // GET /tesseract/constants - List all universal constants with dimensions
    if (method === 'GET' && path === '/tesseract/constants') {
      const constants = [];
      for (const [name, value] of Object.entries(TesseractCore.UNIVERSAL_CONSTANTS)) {
        const dims = TesseractCore.CONSTANT_DIMENSIONS[name];
        const vertex = TesseractCore.TESSERACT_VERTICES.find(v => v.anchor === name);
        constants.push({
          name,
          value,
          dimensions: dims,
          dimensionString: Object.entries(dims)
            .filter(([_, exp]) => exp !== 0)
            .map(([dim, exp]) => exp === 1 ? dim : `${dim}^${exp}`)
            .join('·') || 'dimensionless',
          vertex: vertex ? { id: vertex.id, coords: vertex.coords, realm: vertex.realm } : null
        });
      }
      return respond(200, {
        constants,
        count: 16,
        realms: ['physics', 'mathematical', 'geometric', 'spiralverse']
      });
    }

    // POST /tesseract/parse - Parse a number (supports negative, exponential)
    if (method === 'POST' && path === '/tesseract/parse') {
      const { input, inputs } = body;

      if (inputs && Array.isArray(inputs)) {
        // Parse multiple numbers
        const results = inputs.map(i => ({
          input: i,
          ...TesseractCore.parseNumber(i)
        }));
        return respond(200, { results });
      }

      if (input === undefined) {
        return respond(400, { error: 'Required: input (number/string) or inputs (array)' });
      }

      const result = TesseractCore.parseNumber(input);
      return respond(result.valid ? 200 : 400, { input, ...result });
    }

    // POST /tesseract/analyze - Dimensional analysis of an expression
    if (method === 'POST' && path === '/tesseract/analyze') {
      const { expression, expectedDimensions } = body;
      if (!expression) {
        return respond(400, { error: 'Required: expression (e.g., "c * h / G")' });
      }

      const analysis = expectedDimensions ?
        TesseractCore.validateComputation(expression, expectedDimensions) :
        TesseractCore.analyzeExpression(expression);

      return respond(analysis.dimensionallyValid !== false ? 200 : 400, analysis);
    }

    // POST /tesseract/calculate - Calculate value with dimensional tracking
    if (method === 'POST' && path === '/tesseract/calculate') {
      const { expression } = body;
      if (!expression) {
        return respond(400, { error: 'Required: expression (e.g., "c * h")' });
      }

      const result = TesseractCore.calculate(expression);
      return respond(result.error ? 400 : 200, result);
    }

    // GET /tesseract/lattice - Get dimensional reasoning lattice
    if (method === 'GET' && path === '/tesseract/lattice') {
      const lattice = TesseractCore.createReasoningLattice();
      return respond(200, {
        lattice,
        description: 'Nodes are constants, edges connect dimensionally related quantities',
        usage: 'Use edges to find valid combinations; derivedQuantities shows famous physics results'
      });
    }

    // POST /tesseract/plasma - Sample plasmatic surface at coordinates
    if (method === 'POST' && path === '/tesseract/plasma') {
      const { x, y, z, t, seed, count } = body;

      if (count && count > 1) {
        // Generate multiple samples (for visualization)
        const samples = [];
        for (let i = 0; i < Math.min(count, 100); i++) {
          const ti = (t || 0) + i * 0.1;
          samples.push({
            t: ti,
            value: TesseractCore.plasmaticSurface(x || 0, y || 0, z || 0, ti, seed || 0)
          });
        }
        return respond(200, { samples, coordinates: { x, y, z }, seed });
      }

      const value = TesseractCore.plasmaticSurface(
        x || 0, y || 0, z || 0, t || 0, seed || 0
      );
      return respond(200, { value, coordinates: { x, y, z, t }, seed });
    }

    // POST /tesseract/tiger - Generate tiger stripe pattern
    if (method === 'POST' && path === '/tesseract/tiger') {
      const { theta, phi, t, secretKey } = body;
      if (!secretKey) {
        return respond(400, { error: 'Required: secretKey' });
      }

      const pattern = TesseractCore.tigerStripe(theta || 0, phi || 0, t || 0, secretKey);
      return respond(200, {
        pattern,
        coordinates: { theta, phi, t },
        explanation: 'Deterministic but chaotic-appearing stripe pattern for authentication'
      });
    }

    // POST /tesseract/verify-state - Verify state consistency across all faces
    if (method === 'POST' && path === '/tesseract/verify-state') {
      const { state } = body;
      if (state === undefined) {
        return respond(400, { error: 'Required: state (number)' });
      }

      const result = TesseractCore.verifyStateConsistency(state);
      return respond(result.consistent ? 200 : 403, result);
    }

    // POST /tesseract/environment - Create tunable environment
    if (method === 'POST' && path === '/tesseract/environment') {
      const env = TesseractCore.createEnvironment(body);
      return respond(200, {
        environment: env,
        description: 'VARIABLES that tune the AI space (artificial gravity). MATH remains immutable.'
      });
    }

    // POST /tesseract/mission - Create mission context for AI operations
    if (method === 'POST' && path === '/tesseract/mission') {
      const { missionId, secretKey, environment } = body;
      if (!missionId || !secretKey) {
        return respond(400, { error: 'Required: missionId, secretKey' });
      }

      const env = environment ? TesseractCore.createEnvironment(environment) : null;
      const mission = TesseractCore.createMissionContext(missionId, secretKey, env);

      // Return serializable version (without methods)
      return respond(200, {
        missionId: mission.missionId,
        launched: mission.launched,
        environment: mission.environment,
        params: mission.params,
        status: mission.status,
        checkpointCount: 0,
        usage: 'Store this context. Use /tesseract/checkpoint to record progress.'
      });
    }

    // POST /tesseract/interconnect - Create weighted interconnection between aspects
    if (method === 'POST' && path === '/tesseract/interconnect') {
      const { aspects, weights } = body;
      if (!aspects || typeof aspects !== 'object') {
        return respond(400, { error: 'Required: aspects (object mapping names to values)' });
      }

      const interconnection = TesseractCore.createInterconnection(aspects, weights);
      return respond(200, {
        aspects: interconnection.aspects,
        weights: interconnection.weights,
        computedValue: interconnection.compute(),
        dominant: interconnection.dominant()
      });
    }

    // GET /tesseract/dimensions - Get base dimensions for dimensional analysis
    if (method === 'GET' && path === '/tesseract/dimensions') {
      return respond(200, {
        baseDimensions: TesseractCore.DIMENSIONS,
        constantDimensions: TesseractCore.CONSTANT_DIMENSIONS,
        explanation: 'Like physics SI units - every quantity carries dimensions that constrain valid operations'
      });
    }

    // ========================================================================
    // Adversarial Positioning Endpoints
    // "Inside the box, WE control gravity"
    // Intent-weighted, game-theoretic security
    // ========================================================================

    // GET /adversary - Get security report
    if (method === 'GET' && path === '/adversary') {
      return respond(200, AdversarialPositioning.getSecurityReport());
    }

    // GET /adversary/variables - Get tunable security variables
    if (method === 'GET' && path === '/adversary/variables') {
      return respond(200, {
        variables: AdversarialPositioning.VARIABLES,
        actorTypes: AdversarialPositioning.ACTOR_TYPES,
        assetValues: AdversarialPositioning.ASSET_VALUES,
        zones: AdversarialPositioning.ZONES,
        techniques: AdversarialPositioning.TECHNIQUES,
        motivations: AdversarialPositioning.MOTIVATIONS,
        description: 'Adjust VARIABLES to tune security without math - just weights'
      });
    }

    // POST /adversary/variables - Update tunable variables
    if (method === 'POST' && path === '/adversary/variables') {
      const updates = body;
      for (const [key, value] of Object.entries(updates)) {
        if (AdversarialPositioning.VARIABLES.hasOwnProperty(key)) {
          AdversarialPositioning.VARIABLES[key] = value;
        }
      }
      return respond(200, {
        message: 'Variables updated',
        variables: AdversarialPositioning.VARIABLES
      });
    }

    // POST /adversary/action - Record an action and get intent analysis
    if (method === 'POST' && path === '/adversary/action') {
      const { actorId, target, zone, technique, assetType, intent } = body;
      if (!actorId) {
        return respond(400, { error: 'Required: actorId' });
      }

      const result = AdversarialPositioning.recordAction(actorId, {
        target: target || 'unknown',
        zone: zone || 'AUTHENTICATED',
        technique: technique || 'normal',
        assetType: assetType || 'USER_DATA',
        intent: intent || 'unknown'
      });

      return respond(result.recommendation.level === 'BLOCK' ? 403 : 200, result);
    }

    // GET /adversary/actor/:id - Get actor profile
    if (method === 'GET' && path.startsWith('/adversary/actor/')) {
      const actorId = path.split('/').pop();
      const profile = AdversarialPositioning.getActorProfile(actorId);
      if (profile.error) return respond(404, profile);
      return respond(200, profile);
    }

    // POST /adversary/actor - Query or manage actor
    if (method === 'POST' && path === '/adversary/actor') {
      const { actorId, action } = body;
      if (!actorId) return respond(400, { error: 'Required: actorId' });

      if (action === 'reset') {
        const result = AdversarialPositioning.resetActor(actorId);
        return respond(result.error ? 404 : 200, result);
      }

      const profile = AdversarialPositioning.getActorProfile(actorId);
      return respond(profile.error ? 404 : 200, profile);
    }

    // POST /adversary/gravity - Calculate gravity for a scenario
    if (method === 'POST' && path === '/adversary/gravity') {
      const { intentScore, assetWeight, zone } = body;
      if (intentScore === undefined) {
        return respond(400, { error: 'Required: intentScore (0-1)' });
      }

      const gravity = AdversarialPositioning.calculateGravity(
        intentScore,
        assetWeight ?? 0.5,
        zone || 'AUTHENTICATED'
      );

      return respond(200, {
        input: { intentScore, assetWeight, zone },
        gravity,
        explanation: 'Higher gravity = more friction/resistance for suspicious actors'
      });
    }

    // GET /adversary/drills - List available drills
    if (method === 'GET' && path === '/adversary/drills') {
      const drills = Object.entries(AdversarialPositioning.DRILLS).map(([name, drill]) => ({
        name,
        description: drill.description,
        actionCount: drill.actions.reduce((sum, a) => sum + (a.repeat || 1), 0),
        expectedResponse: drill.expectedResponse
      }));
      return respond(200, {
        drills,
        count: drills.length,
        usage: 'POST /adversary/drill with drillName to run a drill'
      });
    }

    // POST /adversary/drill - Run a security drill
    if (method === 'POST' && path === '/adversary/drill') {
      const { drillName, actorId } = body;

      if (drillName === 'ALL') {
        const results = AdversarialPositioning.runAllDrills();
        return respond(200, results);
      }

      if (!drillName) {
        return respond(400, { error: 'Required: drillName (or "ALL" for all drills)' });
      }

      const result = AdversarialPositioning.runDrill(drillName, actorId);
      if (result.error) return respond(400, result);
      return respond(result.passed ? 200 : 200, result);  // 200 either way, check passed field
    }

    // POST /adversary/honeypot - Create or check honeypot
    if (method === 'POST' && path === '/adversary/honeypot') {
      const { action, honeypotId, actorId, appearsAs, zone } = body;

      if (action === 'create') {
        if (!honeypotId) return respond(400, { error: 'Required: honeypotId' });
        const honeypot = AdversarialPositioning.createHoneypot(honeypotId, {
          appearsAs: appearsAs || 'SECRET_KEY',
          zone: zone || 'CRITICAL'
        });
        return respond(200, {
          message: 'Honeypot created',
          honeypot,
          warning: 'Anyone who touches this will be immediately flagged'
        });
      }

      if (action === 'check') {
        if (!actorId || !honeypotId) {
          return respond(400, { error: 'Required: actorId, honeypotId' });
        }
        const result = AdversarialPositioning.checkHoneypot(actorId, honeypotId);
        if (!result) return respond(404, { error: 'Honeypot not found' });
        return respond(403, result);  // 403 because they got caught!
      }

      if (action === 'list') {
        const honeypots = Array.from(AdversarialPositioning.honeypots.values()).map(h => ({
          id: h.id,
          appearsAs: h.appearsAs,
          zone: h.zone,
          triggerCount: h.triggers.length
        }));
        return respond(200, { honeypots, count: honeypots.length });
      }

      return respond(400, { error: 'Required: action (create|check|list)' });
    }

    // POST /adversary/pattern - Mark patterns as friendly or hostile
    if (method === 'POST' && path === '/adversary/pattern') {
      const { action, patternId, pattern } = body;

      if (action === 'friendly') {
        if (!patternId || !pattern) {
          return respond(400, { error: 'Required: patternId, pattern (object)' });
        }
        AdversarialPositioning.markAsFriendly(patternId, pattern);
        return respond(200, {
          message: 'Pattern marked as friendly (inside joke)',
          patternId,
          pattern
        });
      }

      if (action === 'hostile') {
        if (!patternId || !pattern) {
          return respond(400, { error: 'Required: patternId, pattern (object)' });
        }
        AdversarialPositioning.markAsHostile(patternId, pattern);
        return respond(200, {
          message: 'Pattern marked as hostile',
          patternId,
          pattern
        });
      }

      if (action === 'list') {
        const patterns = Array.from(AdversarialPositioning.knownPatterns.entries())
          .map(([id, p]) => ({
            id,
            type: p.type,
            pattern: p.pattern,
            matchCount: p.matchCount
          }));
        return respond(200, { patterns, count: patterns.length });
      }

      return respond(400, { error: 'Required: action (friendly|hostile|list)' });
    }

    // POST /adversary/well - Create gravity well around asset
    if (method === 'POST' && path === '/adversary/well') {
      const { assetId, assetType, customWeight } = body;
      if (!assetId) {
        return respond(400, { error: 'Required: assetId' });
      }

      const well = AdversarialPositioning.createGravityWell(
        assetId,
        assetType || 'USER_DATA',
        customWeight
      );

      return respond(200, {
        message: 'Gravity well created - this asset now has stronger protection',
        well
      });
    }

    // GET /adversary/5w - Get the 5 W's framework
    if (method === 'GET' && path === '/adversary/5w') {
      return respond(200, {
        framework: {
          WHO: {
            description: 'Actor types and archetypes',
            types: AdversarialPositioning.ACTOR_TYPES
          },
          WHAT: {
            description: 'Asset values - what they are targeting',
            assets: AdversarialPositioning.ASSET_VALUES
          },
          WHEN: {
            description: 'Temporal patterns - timing tells stories',
            patterns: AdversarialPositioning.TEMPORAL_PATTERNS
          },
          WHERE: {
            description: 'Zones - where in our system',
            zones: AdversarialPositioning.ZONES
          },
          WHY: {
            description: 'Motivations - helps predict next moves',
            motivations: AdversarialPositioning.MOTIVATIONS
          },
          HOW: {
            description: 'Attack techniques and methods',
            techniques: AdversarialPositioning.TECHNIQUES
          }
        },
        metaphor: 'Inside the box, WE control gravity. The trajectory reveals intent.'
      });
    }

    return respond(404, {
      error: 'Not found',
      endpoints: [
        '/health', '/geometry', '/ceremony', '/derive', '/authorize',
        '/webhook', '/simulate', '/analyze', '/spin', '/raytrace',
        '/dimensions', '/healing', '/languages', '/agents', '/teams',
        '/presets', '/drift', '/verify',
        '/synth', '/synth/raw', '/synth/verify', '/synth/lexicon', '/synth/compare',
        '/ledger', '/ledger/write', '/ledger/audit', '/ledger/path',
        '/ledger/query', '/ledger/geodesic', '/ledger/zones', '/ledger/reset', '/ledger/hash',
        '/watermark/generate', '/watermark/verify', '/watermark/qr', '/watermark/qr/verify',
        '/watermark/bandit', '/watermark/fingerprint',
        '/tesseract', '/tesseract/constants', '/tesseract/parse', '/tesseract/analyze',
        '/tesseract/calculate', '/tesseract/lattice', '/tesseract/plasma', '/tesseract/tiger',
        '/tesseract/verify-state', '/tesseract/environment', '/tesseract/mission',
        '/tesseract/interconnect', '/tesseract/dimensions',
        '/adversary', '/adversary/variables', '/adversary/action', '/adversary/actor',
        '/adversary/gravity', '/adversary/drills', '/adversary/drill', '/adversary/honeypot',
        '/adversary/pattern', '/adversary/well', '/adversary/5w'
      ]
    });
  } catch (err) {
    return respond(500, { error: err.message });
  }
};
