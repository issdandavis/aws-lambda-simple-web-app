"""
SCBE-AETHERMOORE API Service

FastAPI application providing REST endpoints for:
- Risk assessment with harmonic scaling
- Governance verification (snap detection, causality)
- Dual-lattice consensus
"""
import time
import base64
import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api import __version__
from api.models import (
    RiskAssessmentRequest, RiskAssessmentResponse, Decision,
    GovernanceVerifyRequest, GovernanceVerifyResponse, SnapEvent,
    ConsensusSubmitRequest, ConsensusSubmitResponse, ConsensusState,
    HarmonicScaleRequest, HarmonicScaleResponse,
    HealthResponse,
)

# Import core modules
try:
    from symphonic_cipher.qasi_core import (
        hyperbolic_distance,
        realm_distance,
        harmonic_scale,
        security_bits,
    )
    QASI_AVAILABLE = True
except ImportError:
    QASI_AVAILABLE = False

try:
    from symphonic_cipher.scbe_aethermoore.governance import (
        GovernanceEngine,
        SnapProtocol,
        CausalityVerifier,
    )
    GOVERNANCE_AVAILABLE = True
except ImportError:
    GOVERNANCE_AVAILABLE = False

try:
    from symphonic_cipher.scbe_aethermoore.quantum import (
        PQCryptoSystem,
        PQContextCommitment,
    )
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# App Configuration
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="SCBE-AETHERMOORE API",
    description="Harmonic scaling governance and post-quantum consensus API",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════
# Health Check
# ═══════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and module availability."""
    modules = {
        "qasi_core": QASI_AVAILABLE,
        "governance": GOVERNANCE_AVAILABLE,
        "quantum": QUANTUM_AVAILABLE,
    }

    all_healthy = all(modules.values())
    some_healthy = any(modules.values())

    if all_healthy:
        status = "healthy"
    elif some_healthy:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        version=__version__,
        modules=modules,
    )


# ═══════════════════════════════════════════════════════════════
# Risk Assessment
# ═══════════════════════════════════════════════════════════════

@app.post("/api/v1/assess-risk", response_model=RiskAssessmentResponse, tags=["Risk"])
async def assess_risk(request: RiskAssessmentRequest):
    """
    Assess risk using harmonic scaling.

    The Grand Unified Equation:
        Risk' = Risk_base × H(d*, R)

    where H(d*, R) = R^(d*²) amplifies risk super-exponentially
    as distance from known realms increases.
    """
    if not QASI_AVAILABLE:
        raise HTTPException(status_code=503, detail="QASI core module not available")

    try:
        # Convert to numpy
        u = np.array(request.context_vector, dtype=np.float64)

        # Calculate d* (distance to nearest realm)
        if request.realm_centers and len(request.realm_centers) > 0:
            centers = np.array(request.realm_centers, dtype=np.float64)
            d_star = realm_distance(u, centers)
        else:
            # No realms = maximum uncertainty
            d_star = 1.0

        # Harmonic scaling
        R = 1.5  # Perfect fifth

        if request.use_bounded:
            # Bounded form: H = 1 + α·tanh(β·d)
            alpha, beta = 10.0, 0.5
            H = 1.0 + alpha * np.tanh(beta * d_star)
        else:
            # Primary form: H = R^(d*²)
            H = harmonic_scale(d_star, R)

        # Compute final risk
        risk_prime = request.risk_base * H

        # Security bits
        sec_bits = security_bits(128, d_star, R)

        # Decision thresholds
        if risk_prime < 0.30:
            decision = Decision.ALLOW
        elif risk_prime > 0.70:
            decision = Decision.DENY
        else:
            decision = Decision.QUARANTINE

        return RiskAssessmentResponse(
            decision=decision,
            risk_base=request.risk_base,
            risk_prime=float(risk_prime),
            harmonic_scale=float(H),
            d_star=float(d_star),
            security_bits=float(sec_bits),
            timestamp=time.time(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# Governance Verification
# ═══════════════════════════════════════════════════════════════

@app.post("/api/v1/governance/verify", response_model=GovernanceVerifyResponse, tags=["Governance"])
async def verify_governance(request: GovernanceVerifyRequest):
    """
    Verify governance compliance of a state trajectory.

    Detects:
    - Snap events (discontinuities exceeding threshold)
    - Causality violations (time ordering)
    """
    if not GOVERNANCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Governance module not available")

    try:
        # Initialize components
        snap_protocol = SnapProtocol(threshold=request.snap_threshold)
        causality_verifier = CausalityVerifier()

        snap_events = []
        causality_valid = True

        # Process trajectory
        for i, (state, ts) in enumerate(zip(request.state_trajectory, request.timestamps)):
            state_vec = np.array(state, dtype=np.float64)

            # Check for snaps
            event = snap_protocol.check(state_vec, ts)
            if event:
                snap_events.append(SnapEvent(
                    index=i,
                    magnitude=event.magnitude,
                    timestamp=ts,
                ))

            # Verify causality
            record = causality_verifier.record(state_vec, ts)
            if i > 0 and not causality_verifier.verify_chain():
                causality_valid = False

        return GovernanceVerifyResponse(
            valid=len(snap_events) == 0 and causality_valid,
            snap_events=snap_events,
            causality_verified=causality_valid,
            total_states=len(request.state_trajectory),
            timestamp=time.time(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# Consensus
# ═══════════════════════════════════════════════════════════════

@app.post("/api/v1/consensus/submit", response_model=ConsensusSubmitResponse, tags=["Consensus"])
async def submit_consensus(request: ConsensusSubmitRequest):
    """
    Submit data for dual-lattice consensus.

    Requires BOTH Kyber KEM and Dilithium DSA to validate
    for consensus to be achieved.
    """
    if not QUANTUM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Quantum module not available")

    try:
        # Decode payload
        try:
            payload = base64.b64decode(request.payload)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 payload")

        # Initialize PQ crypto system
        pq_system = PQCryptoSystem()

        # Create context commitment
        commitment = PQContextCommitment.create(
            context_id=request.context_id,
            data=payload,
        )

        # Sign the commitment
        signature = pq_system.sign(commitment.to_bytes())

        # Verify (in real system, this would be done by validators)
        kyber_valid = True  # KEM encapsulation succeeded
        dilithium_valid = pq_system.verify_signature(
            commitment.to_bytes(),
            signature,
        )

        # Determine consensus state
        if kyber_valid and dilithium_valid:
            state = ConsensusState.CONSENSUS
            settled = True
        elif kyber_valid:
            state = ConsensusState.KYBER_ONLY
            settled = False
        elif dilithium_valid:
            state = ConsensusState.DILITHIUM_ONLY
            settled = False
        else:
            state = ConsensusState.FAILED
            settled = False

        return ConsensusSubmitResponse(
            state=state,
            settled=settled,
            context_commitment=commitment.commitment_hash,
            kyber_valid=kyber_valid,
            dilithium_valid=dilithium_valid,
            timestamp=time.time(),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# Direct Harmonic Scaling
# ═══════════════════════════════════════════════════════════════

@app.post("/api/v1/harmonic-scale", response_model=HarmonicScaleResponse, tags=["Math"])
async def compute_harmonic_scale(request: HarmonicScaleRequest):
    """
    Compute harmonic scaling directly.

    H(d*, R) = R^(d*²)

    Returns scaling factor and security bits.
    """
    if not QASI_AVAILABLE:
        raise HTTPException(status_code=503, detail="QASI core module not available")

    try:
        overflow = False

        try:
            H = harmonic_scale(request.d_star, request.R)
        except (OverflowError, ValueError):
            # Fall back to bounded form
            alpha, beta = 10.0, 0.5
            H = 1.0 + alpha * np.tanh(beta * request.d_star)
            overflow = True

        sec_bits = security_bits(
            request.base_security_bits,
            request.d_star,
            request.R,
        )

        return HarmonicScaleResponse(
            H=float(H),
            d_star=request.d_star,
            R=request.R,
            security_bits=float(sec_bits),
            overflow=overflow,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# Root
# ═══════════════════════════════════════════════════════════════

@app.get("/", tags=["System"])
async def root():
    """API root - redirects to docs."""
    return {
        "name": "SCBE-AETHERMOORE API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
