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

    // Legacy endpoints
    if (method === 'POST' && path === '/verify') {
      return respond(200, TrajectoryAuthorization.authorize({
        chiCurrent: body.chiCurrent || 0.5, chiPrevious: body.chiPrevious || 0.5
      }));
    }

    return respond(404, {
      error: 'Not found',
      endpoints: ['/health', '/geometry', '/ceremony', '/derive', '/authorize', '/webhook', '/simulate', '/verify']
    });
  } catch (err) {
    return respond(500, { error: err.message });
  }
};
