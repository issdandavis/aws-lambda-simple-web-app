"""
SCBE-AETHERMOORE REST API
=========================
FastAPI server for cross-device access (phone, tablet, laptop, Kindle).

Run:
    uvicorn scbe_production.api:app --host 0.0.0.0 --port 8000

Or:
    python -m scbe_production.api
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

# SCBE imports
try:
    from .service import SCBEProductionService, AccessRequest as SCBEAccessRequest
    from .config import get_config
    from .logging import get_logger
except ImportError:
    # Running as script
    from service import SCBEProductionService, AccessRequest as SCBEAccessRequest
    from config import get_config
    from logging import get_logger

# Sacred Tongues
try:
    from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
        SacredTongueTokenizer,
        SacredTongue,
        format_ss1_blob,
        encode_to_spelltext,
        TONGUES,
    )
    TONGUES_AVAILABLE = True
except ImportError:
    TONGUES_AVAILABLE = False

# Initialize
app = FastAPI(
    title="SCBE-AETHERMOORE API",
    description="Hyperbolic Geometry + Sacred Tongues + Post-Quantum Cryptography",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for cross-device access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service instance
service = SCBEProductionService()
logger = get_logger()


# =============================================================================
# Request/Response Models
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    components: Dict[str, Any]


class SealRequest(BaseModel):
    data: str = Field(..., description="Data to seal (text)")
    agent_id: str = Field(..., description="Agent identifier")
    topic: str = Field(default="default", description="Topic classification")
    position: List[int] = Field(default=[1, 2, 3, 5, 8, 13], description="Fibonacci position")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags")


class SealResponse(BaseModel):
    shard_id: str
    agent_id: str
    topic: str
    position: List[int]
    timestamp: str
    sealed_data_length: int
    success: bool


class AccessRequestModel(BaseModel):
    agent_id: str = Field(..., description="Requesting agent ID")
    message: str = Field(default="", description="Access justification")
    trust_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Trust level 0-1")
    position: List[int] = Field(default=[1, 2, 3, 5, 8, 13], description="Target position")
    intent: str = Field(default="read", description="Intent: read, write, admin, delete")


class AccessResponseModel(BaseModel):
    decision: str
    risk_score: float
    reason: str
    geometric_path: str
    time_dilation: float
    success: bool


class EncodeRequest(BaseModel):
    data: str = Field(..., description="Text to encode")
    tongue: str = Field(default="ko", description="Tongue code: ko, av, ru, ca, um, dr")


class EncodeResponse(BaseModel):
    tongue: str
    tongue_name: str
    spelltext: str
    token_count: int
    success: bool


class VerifyRequest(BaseModel):
    agent_id: str
    intent: str = Field(default="read")
    trust_level: float = Field(default=0.5, ge=0.0, le=1.0)
    position: List[int] = Field(default=[1, 2, 3, 5, 8, 13])


class VerifyResponse(BaseModel):
    decision: str
    risk_score: float
    hyperbolic_distance: float
    harmonic_factor: float
    geoseal_path: str
    time_dilation: float
    position_valid: bool
    success: bool


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web demo."""
    web_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "index.html")
    if os.path.exists(web_path):
        with open(web_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="""
    <html>
        <head><title>SCBE-AETHERMOORE</title></head>
        <body style="background:#0a0a0f;color:#e2e8f0;font-family:sans-serif;padding:2rem;text-align:center;">
            <h1>SCBE-AETHERMOORE API</h1>
            <p>API is running. Visit <a href="/docs" style="color:#6366f1;">/docs</a> for documentation.</p>
        </body>
    </html>
    """)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    result = service.health_check()
    return HealthResponse(**result)


@app.post("/seal", response_model=SealResponse)
async def seal_memory(request: SealRequest):
    """Seal data with SCBE protection."""
    try:
        shard = service.seal_memory(
            plaintext=request.data.encode('utf-8'),
            agent_id=request.agent_id,
            topic=request.topic,
            position=tuple(request.position),
            tags=request.tags,
        )
        return SealResponse(
            shard_id=shard.shard_id,
            agent_id=shard.agent_id,
            topic=shard.topic,
            position=list(shard.position),
            timestamp=shard.timestamp,
            sealed_data_length=len(shard.sealed_data),
            success=True,
        )
    except Exception as e:
        logger.error(f"Seal failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/access", response_model=AccessResponseModel)
async def access_memory(request: AccessRequestModel):
    """Request access with governance check."""
    try:
        # Map intent to risk features
        intent_risks = {"read": 0.1, "write": 0.3, "admin": 0.6, "delete": 0.8}
        intent_risk = intent_risks.get(request.intent, 0.5)

        scbe_request = SCBEAccessRequest(
            agent_id=request.agent_id,
            message=request.message,
            features={
                "trust_level": request.trust_level,
                "intent_risk": intent_risk,
            },
            position=tuple(request.position),
        )

        response = service.access_memory(scbe_request)

        return AccessResponseModel(
            decision=response.decision,
            risk_score=response.risk_score,
            reason=response.reason,
            geometric_path=response.geometric_path,
            time_dilation=response.time_dilation,
            success=True,
        )
    except Exception as e:
        logger.error(f"Access check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify", response_model=VerifyResponse)
async def verify_authorization(request: VerifyRequest):
    """Quick verification endpoint (simplified)."""
    import numpy as np
    import hashlib

    try:
        # Hash agent ID to get user point
        hash_bytes = hashlib.sha256(request.agent_id.encode()).digest()
        user_point = np.array([
            (hash_bytes[0] / 255.0) * 0.8 - 0.4,
            (hash_bytes[1] / 255.0) * 0.8 - 0.4
        ])
        trusted_center = np.array([0.0, 0.0])

        # Hyperbolic distance
        eps = 1e-6
        u, v = user_point, trusted_center
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
        if u_norm >= 1.0 - eps:
            u = u * ((1.0 - eps) / u_norm)
        diff_norm_sq = np.linalg.norm(u - v) ** 2
        denom = (1 - np.linalg.norm(u)**2) * (1 - np.linalg.norm(v)**2)
        d_H = float(np.arccosh(max(1.0, 1 + 2 * diff_norm_sq / denom)))

        # Harmonic scaling
        H = 1 + 10 * np.tanh(0.5 * d_H)

        # Intent risk
        intent_risks = {"read": 0.1, "write": 0.3, "admin": 0.6, "delete": 0.8}
        intent_risk = intent_risks.get(request.intent, 0.5)

        # Position validation (Fibonacci check)
        pos = request.position
        is_fibonacci = len(pos) >= 3 and all(
            abs(pos[i] - (pos[i-2] + pos[i-1])) < 0.01
            for i in range(2, len(pos))
        )
        position_penalty = 0 if is_fibonacci else 0.3

        # GeoSeal path
        geo_distance = (1 - request.trust_level) * 0.5 + intent_risk * 0.3
        geo_path = "interior" if geo_distance < 0.5 else "exterior"
        time_dilation = float(np.exp(-2 * geo_distance))

        # Final risk
        risk = (intent_risk + (1 - request.trust_level) * 0.5 + position_penalty) * H
        risk = risk * (1.5 if geo_path == "exterior" else 1.0)
        normalized_risk = min(1.0, risk / 5)

        # Decision
        if normalized_risk >= 0.8:
            decision = "SNAP"
        elif normalized_risk >= 0.4:
            decision = "DENY"
        elif normalized_risk >= 0.2:
            decision = "QUARANTINE"
        else:
            decision = "ALLOW"

        return VerifyResponse(
            decision=decision,
            risk_score=round(normalized_risk, 4),
            hyperbolic_distance=round(d_H, 4),
            harmonic_factor=round(H, 4),
            geoseal_path=geo_path,
            time_dilation=round(time_dilation, 4),
            position_valid=is_fibonacci,
            success=True,
        )
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/encode", response_model=EncodeResponse)
async def encode_to_tongue(request: EncodeRequest):
    """Encode text to Sacred Tongue spell-text."""
    if not TONGUES_AVAILABLE:
        raise HTTPException(status_code=501, detail="Sacred Tongues not available")

    try:
        tongue_code = request.tongue.lower()
        if tongue_code not in TONGUES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tongue. Valid: {list(TONGUES.keys())}"
            )

        tokenizer = SacredTongueTokenizer(tongue_code)
        data_bytes = request.data.encode('utf-8')

        # Encode to spell-text
        spelltext = tokenizer.encode_to_string(data_bytes, separator=" ")

        return EncodeResponse(
            tongue=tongue_code,
            tongue_name=TONGUES[tongue_code].name,
            spelltext=spelltext,
            token_count=len(data_bytes),
            success=True,
        )
    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tongues")
async def list_tongues():
    """List available Sacred Tongues."""
    if not TONGUES_AVAILABLE:
        return {"error": "Sacred Tongues not available", "tongues": []}

    return {
        "tongues": [
            {
                "code": code,
                "name": spec.name,
                "domain": spec.domain,
                "prefix_count": len(spec.prefixes),
                "suffix_count": len(spec.suffixes),
            }
            for code, spec in TONGUES.items()
        ]
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the API server."""
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting SCBE-AETHERMOORE API on http://{host}:{port}")
    print(f"Documentation: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
