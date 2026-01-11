/**
 * Spiralverse Protocol - Patent Claim Implementation
 * ML-KEM Integrated Dual-Lane Key Derivation
 *
 * Claim 1: Manifold-gated dual-lane key schedule
 * Claim 2: Trajectory-based coherence authorization
 *
 * Uses Node.js crypto (HKDF) + ML-KEM simulation layer
 */

const crypto = require('crypto');

// ============================================================================
// ML-KEM SIMULATION LAYER (swap for real ML-KEM in production)
// Simulates ML-KEM-768 parameter set structure
// ============================================================================

const MLKEM = {
  PARAMS: { name: 'ML-KEM-768', pkLen: 1184, skLen: 2400, ctLen: 1088, ssLen: 32 },

  // Generate keypair (simulated - uses crypto.randomBytes for structure)
  keyGen() {
    const seed = crypto.randomBytes(64);
    const pk = crypto.createHash('sha3-256').update(Buffer.concat([seed, Buffer.from('pk')])).digest();
    const sk = crypto.createHash('sha3-256').update(Buffer.concat([seed, Buffer.from('sk')])).digest();
    // Expand to realistic sizes for demonstration
    const pkFull = Buffer.concat([pk, crypto.randomBytes(this.PARAMS.pkLen - 32)]);
    const skFull = Buffer.concat([sk, crypto.randomBytes(this.PARAMS.skLen - 32)]);
    return { pk: pkFull.toString('base64'), sk: skFull.toString('base64'), pkHash: pk.toString('hex').slice(0, 16) };
  },

  // Encapsulate: pk → (ct, ss)
  encapsulate(pkBase64) {
    const pk = Buffer.from(pkBase64, 'base64');
    const ephemeral = crypto.randomBytes(32);
    const ss = crypto.createHash('sha3-256').update(Buffer.concat([pk.slice(0, 32), ephemeral])).digest();
    const ct = crypto.createHash('sha3-256').update(Buffer.concat([ephemeral, pk.slice(0, 32)])).digest();
    const ctFull = Buffer.concat([ct, crypto.randomBytes(this.PARAMS.ctLen - 32)]);
    return { ct: ctFull.toString('base64'), ss: ss.toString('hex'), ctHash: ct.toString('hex').slice(0, 16) };
  },

  // Decapsulate: (sk, ct) → ss
  decapsulate(skBase64, ctBase64) {
    const sk = Buffer.from(skBase64, 'base64');
    const ct = Buffer.from(ctBase64, 'base64');
    const ss = crypto.createHash('sha3-256').update(Buffer.concat([sk.slice(0, 32), ct.slice(0, 32)])).digest();
    return { ss: ss.toString('hex') };
  }
};

// ============================================================================
// CLAIM 1: Manifold-Gated Dual-Lane Key Schedule
// 5-tuple classifier: (χ, pk_in, ct_in, pk_out, ct_out) → lane bit L
// ============================================================================

const DualLaneKeySchedule = {
  PHI: 1.618033988749895,
  LANE_THRESHOLD: 0.5,

  // Extract geometric features from 5-tuple ceremony outputs
  extractManifoldCoordinates(chi, pkIn, ctIn, pkOut, ctOut) {
    // Hash each component to fixed-size coordinates
    const h = (b) => {
      const hash = crypto.createHash('sha256').update(Buffer.from(b, 'base64')).digest();
      return hash.readUInt32BE(0) / 0xFFFFFFFF; // Normalize to [0,1]
    };
    return {
      x1: typeof chi === 'number' ? chi : this.hashContext(chi),
      x2: h(pkIn),
      x3: h(ctIn),
      x4: h(pkOut),
      x5: h(ctOut)
    };
  },

  // Hash context object to scalar
  hashContext(ctx) {
    const str = JSON.stringify(ctx);
    const hash = crypto.createHash('sha256').update(str).digest();
    return hash.readUInt32BE(0) / 0xFFFFFFFF;
  },

  // Geometric classifier: project 5-tuple onto decision manifold
  computeLaneBit(coords) {
    const { x1, x2, x3, x4, x5 } = coords;
    // Spiral projection using golden ratio
    const r = Math.sqrt(x1*x1 + x2*x2 + x3*x3 + x4*x4 + x5*x5);
    const theta = Math.atan2(x2 - x4, x1 - x3) * this.PHI; // Inside vs outside asymmetry
    const psi = Math.atan2(x5, (x2 + x4) / 2); // Ciphertext contribution
    // Manifold projection
    const projection = Math.sin(theta + psi) * r / (1 + Math.abs(Math.cos(theta * this.PHI)));
    const normalized = (Math.tanh(projection) + 1) / 2;
    return {
      L: normalized >= this.LANE_THRESHOLD ? 1 : 0,
      confidence: Math.round(normalized * 1000) / 1000,
      coords
    };
  },

  // HKDF key derivation with lane-specific context
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
          info,
          salt: saltBuffer.toString('hex')
        });
      });
    });
  },

  // Full protocol: ENCAPSULATE → CLASSIFY → DERIVE
  async execute(ceremony) {
    const { chi, insideParty, outsideParty } = ceremony;

    // Step 1: ML-KEM encapsulations
    const insideEnc = MLKEM.encapsulate(insideParty.pk);
    const outsideEnc = MLKEM.encapsulate(outsideParty.pk);

    // Step 2: Combine shared secrets
    const combinedSS = crypto.createHash('sha256')
      .update(Buffer.from(insideEnc.ss, 'hex'))
      .update(Buffer.from(outsideEnc.ss, 'hex'))
      .digest('hex');

    // Step 3: Compute lane bit from 5-tuple
    const coords = this.extractManifoldCoordinates(chi, insideParty.pk, insideEnc.ct, outsideParty.pk, outsideEnc.ct);
    const classification = this.computeLaneBit(coords);

    // Step 4: Derive lane-specific key
    const derivedKey = await this.deriveKey(combinedSS, classification.L);

    return {
      ceremony: {
        chi: typeof chi === 'object' ? chi : { value: chi },
        inside: { pkHash: insideParty.pkHash, ctHash: insideEnc.ctHash },
        outside: { pkHash: outsideParty.pkHash, ctHash: outsideEnc.ctHash }
      },
      classification,
      derivedKey,
      nonUnilateral: true // No single party controls lane selection
    };
  }
};

// ============================================================================
// CLAIM 2: Trajectory-Based Coherence Authorization
// K(t) = f(χ(t), χ(t-1), Δχ, σ_drift, r_tube)
// ============================================================================

const TrajectoryAuthorization = {
  TUBE_RADIUS: 0.15,
  DECAY_FACTOR: 0.95,
  COHERENCE_THRESHOLD: 0.7,

  // Compute 5-variable kernel
  computeKernel(chiCurrent, chiPrevious, trajectory) {
    const deltaChi = Math.abs(chiCurrent - chiPrevious);
    const driftSigma = this.computeDriftSigma(trajectory);
    const tubeRadius = this.TUBE_RADIUS * Math.pow(this.DECAY_FACTOR, trajectory.length);

    return {
      chi_t: chiCurrent,
      chi_t1: chiPrevious,
      delta_chi: Math.round(deltaChi * 1000) / 1000,
      sigma_drift: Math.round(driftSigma * 1000) / 1000,
      r_tube: Math.round(tubeRadius * 1000) / 1000
    };
  },

  // Compute drift standard deviation from trajectory
  computeDriftSigma(trajectory) {
    if (trajectory.length < 2) return 0;
    const diffs = [];
    for (let i = 1; i < trajectory.length; i++) {
      diffs.push(trajectory[i] - trajectory[i - 1]);
    }
    const mean = diffs.reduce((a, b) => a + b, 0) / diffs.length;
    const variance = diffs.reduce((a, d) => a + (d - mean) ** 2, 0) / diffs.length;
    return Math.sqrt(variance);
  },

  // Tube test with drift amplification
  tubeTest(kernel) {
    const { sigma_drift, r_tube } = kernel;
    const withinTube = sigma_drift < r_tube;
    const amplification = withinTube ? 1.0 : 1.0 + (sigma_drift - r_tube) * 10;
    return { withinTube, amplification: Math.round(amplification * 100) / 100 };
  },

  // Coherence check across kernel variables
  computeCoherence(kernel) {
    const vars = [kernel.chi_t, kernel.chi_t1, kernel.delta_chi, kernel.sigma_drift, kernel.r_tube];
    const mean = vars.reduce((a, b) => a + b, 0) / vars.length;
    const variance = vars.reduce((a, v) => a + (v - mean) ** 2, 0) / vars.length;
    return Math.round((1 - Math.sqrt(variance)) * 1000) / 1000;
  },

  // Full authorization check
  authorize(request) {
    const { chiCurrent, chiPrevious, trajectory = [] } = request;
    const kernel = this.computeKernel(chiCurrent, chiPrevious, trajectory);
    const tubeResult = this.tubeTest(kernel);
    const coherence = this.computeCoherence(kernel);
    const authorized = tubeResult.withinTube && coherence >= this.COHERENCE_THRESHOLD;

    return {
      authorized,
      kernel,
      tubeTest: tubeResult,
      coherence,
      thresholds: { tube: this.TUBE_RADIUS, coherence: this.COHERENCE_THRESHOLD }
    };
  }
};

// ============================================================================
// Lambda Handler
// ============================================================================

exports.handler = async (event) => {
  const method = event.httpMethod || event.requestContext?.http?.method || 'GET';
  const path = event.path || event.rawPath || '/';
  const respond = (code, body) => ({ statusCode: code, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body, null, 2) });

  try {
    // GET /health
    if (method === 'GET' && path === '/health') {
      return respond(200, { status: 'healthy', claims: ['manifold-dual-lane-key-schedule', 'trajectory-coherence-auth'], mlkem: MLKEM.PARAMS.name });
    }

    const body = event.body ? JSON.parse(event.body) : {};

    // POST /ceremony - Initialize ceremony with keypairs
    if (method === 'POST' && path === '/ceremony') {
      const inside = MLKEM.keyGen();
      const outside = MLKEM.keyGen();
      return respond(200, { ceremony: { inside: { pk: inside.pk, pkHash: inside.pkHash }, outside: { pk: outside.pk, pkHash: outside.pkHash } }, note: 'Store sk securely, pass pk to /derive' });
    }

    // POST /derive - Execute dual-lane key derivation (Claim 1)
    if (method === 'POST' && path === '/derive') {
      const { chi, insidePk, outsidePk } = body;
      if (!chi || !insidePk || !outsidePk) {
        return respond(400, { error: 'Required: chi, insidePk, outsidePk' });
      }
      const result = await DualLaneKeySchedule.execute({
        chi,
        insideParty: { pk: insidePk, pkHash: crypto.createHash('sha256').update(Buffer.from(insidePk, 'base64')).digest('hex').slice(0, 16) },
        outsideParty: { pk: outsidePk, pkHash: crypto.createHash('sha256').update(Buffer.from(outsidePk, 'base64')).digest('hex').slice(0, 16) }
      });
      return respond(200, result);
    }

    // POST /authorize - Trajectory authorization (Claim 2)
    if (method === 'POST' && path === '/authorize') {
      const { chiCurrent, chiPrevious, trajectory } = body;
      if (chiCurrent === undefined || chiPrevious === undefined) {
        return respond(400, { error: 'Required: chiCurrent, chiPrevious. Optional: trajectory[]' });
      }
      const result = TrajectoryAuthorization.authorize({ chiCurrent, chiPrevious, trajectory });
      return respond(result.authorized ? 200 : 403, result);
    }

    // GET /health, POST /brain-lane, /oversight-lane, /verify - Legacy endpoints
    if (method === 'POST' && path === '/brain-lane') {
      const result = await DualLaneKeySchedule.execute({ chi: { ...body, forceLane: 'brain' }, insideParty: MLKEM.keyGen(), outsideParty: MLKEM.keyGen() });
      if (result.classification.L !== 0) return respond(403, { error: 'Classified for oversight', ...result });
      return respond(200, { lane: 'brain', ...result });
    }

    if (method === 'POST' && path === '/oversight-lane') {
      const result = await DualLaneKeySchedule.execute({ chi: { ...body, forceLane: 'oversight' }, insideParty: MLKEM.keyGen(), outsideParty: MLKEM.keyGen() });
      const auth = TrajectoryAuthorization.authorize({ chiCurrent: result.classification.coords.x1, chiPrevious: 0.5, trajectory: body.trajectory || [] });
      if (!auth.authorized) return respond(403, { error: 'Trajectory unauthorized', keyDerivation: result, authorization: auth });
      return respond(200, { lane: 'oversight', keyDerivation: result, authorization: auth });
    }

    if (method === 'POST' && path === '/verify') {
      return respond(200, TrajectoryAuthorization.authorize({ chiCurrent: body.chiCurrent || 0.5, chiPrevious: body.chiPrevious || 0.5, trajectory: body.trajectory || [] }));
    }

    return respond(404, { error: 'Not found', endpoints: ['/health', '/ceremony', '/derive', '/authorize', '/brain-lane', '/oversight-lane', '/verify'] });
  } catch (err) {
    return respond(500, { error: err.message, stack: err.stack });
  }
};
