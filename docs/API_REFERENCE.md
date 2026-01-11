# Spiralverse Protocol - API Reference

> **16 Vertices of Universal Truth - Geometric AI Operating System**

## Base URL

```
https://{api-id}.execute-api.{region}.amazonaws.com/{stage}
```

## Authentication

API requests can optionally include an API key:
```
X-Api-Key: your-api-key
```

---

## Core Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "protocol": "spiralverse"
}
```

### GET /geometry
Get torus geometry parameters.

**Response:**
```json
{
  "R": 3.0,
  "r": 1.0,
  "dimensions": 10,
  "topology": "T^10 (10-torus)"
}
```

---

## Tesseract Core

The tesseract (4D hypercube) anchors 16 universal constants that form the immutable MATH layer.

### GET /tesseract
Get tesseract geometry and structure.

**Response:**
```json
{
  "vertices": 16,
  "edges": 32,
  "faces": 24,
  "cells": 8,
  "realms": {
    "physics": [0, 1, 2, 3],
    "mathematical": [4, 5, 6, 7],
    "geometric": [8, 9, 10, 11],
    "spiralverse": [12, 13, 14, 15]
  },
  "constants": { "c": 299792458, "h": 6.62607015e-34, ... }
}
```

### GET /tesseract/constants
List all 16 universal constants with their dimensions.

**Response:**
```json
{
  "constants": [
    {
      "name": "c",
      "value": 299792458,
      "dimensions": { "L": 1, "T": -1 },
      "dimensionString": "L·T^-1",
      "vertex": { "id": 0, "coords": [-1,-1,-1,-1], "realm": "physics" }
    }
    // ... 15 more
  ],
  "count": 16,
  "realms": ["physics", "mathematical", "geometric", "spiralverse"]
}
```

### POST /tesseract/parse
Parse numbers in any format (integer, float, negative, exponential).

**Request:**
```json
{
  "input": "6.62607015e-34"
}
```

Or parse multiple:
```json
{
  "inputs": ["6.62607015e-34", "-3.5E+10", "42", "-17.5"]
}
```

**Response:**
```json
{
  "input": "6.62607015e-34",
  "value": 6.62607015e-34,
  "valid": true,
  "parsed": { "mantissa": 6.62607015, "exponent": -34, "notation": "exponential" }
}
```

### POST /tesseract/analyze
Perform dimensional analysis on an expression.

**Request:**
```json
{
  "expression": "c * h / G"
}
```

**Response:**
```json
{
  "expression": "c * h / G",
  "terms": [
    { "token": "c", "value": 299792458, "dimensions": { "L": 1, "T": -1 } },
    { "token": "h", "value": 6.62607015e-34, "dimensions": { "M": 1, "L": 2, "T": -1 } },
    { "token": "G", "value": 6.6743e-11, "dimensions": { "L": 3, "M": -1, "T": -2 } }
  ],
  "dimensionallyValid": true,
  "resultDimensions": { "M": 2, "T": 2 },
  "resultDimensionString": "M^2·T^2"
}
```

### POST /tesseract/calculate
Calculate value with dimensional tracking.

**Request:**
```json
{
  "expression": "pi * phi"
}
```

**Response:**
```json
{
  "value": 5.083203692...,
  "dimensions": {},
  "dimensionString": "dimensionless",
  "terms": ["pi", "phi"]
}
```

### GET /tesseract/lattice
Get the dimensional reasoning lattice.

**Response:**
```json
{
  "lattice": {
    "nodes": [...],  // 16 constants as nodes
    "edges": [...],  // Connections between dimensionally-related constants
    "derivedQuantities": [
      { "name": "Planck length", "expression": "sqrt(h * G / c^3)", "value": 1.616e-35 },
      { "name": "Planck time", "expression": "sqrt(h * G / c^5)", "value": 5.391e-44 },
      { "name": "Planck mass", "expression": "sqrt(h * c / G)", "value": 2.176e-8 }
    ]
  }
}
```

### POST /tesseract/plasma
Sample the plasmatic surface (deterministic chaos).

**Request:**
```json
{
  "x": 1.5,
  "y": 2.0,
  "z": 0.5,
  "t": 0,
  "seed": 0,
  "count": 10
}
```

**Response:**
```json
{
  "samples": [
    { "t": 0, "value": 0.6137 },
    { "t": 0.1, "value": 0.6014 },
    ...
  ],
  "coordinates": { "x": 1.5, "y": 2.0, "z": 0.5 }
}
```

### POST /tesseract/tiger
Generate tiger stripe pattern (deterministic chaos for authentication).

**Request:**
```json
{
  "theta": 1.5,
  "phi": 0.8,
  "t": 0,
  "secretKey": "my-secret"
}
```

**Response:**
```json
{
  "pattern": {
    "value": 0.72,
    "stripe": "LIGHT",
    "intensity": 0.4745
  }
}
```

### POST /tesseract/verify-state
Verify state consistency across all 8 cubic faces.

**Request:**
```json
{
  "state": 12345678
}
```

**Response:**
```json
{
  "consistent": true,
  "readings": [3847234, 2938472, ...],
  "avgBitDeviation": 12.875,
  "verdict": "State is geometrically consistent across all tesseract faces"
}
```

### POST /tesseract/environment
Create a tunable environment (VARIABLES, not MATH).

**Request:**
```json
{
  "semanticWeight": 1.5,
  "viscosity": 0.2,
  "temperature": 1.2,
  "snapThreshold": 0.6
}
```

**Response:**
```json
{
  "environment": {
    "gravity": { "semantic": 1.5, "temporal": 1.0, ... },
    "atmosphere": { "viscosity": 0.2, "temperature": 1.2, ... },
    "shields": { "snap": 0.6, ... },
    "mission": { "maxDuration": 300000, ... }
  },
  "description": "VARIABLES that tune the AI space. MATH remains immutable."
}
```

---

## Geometric Ledger

The ledger enforces geometric integrity - facts must follow geodesic paths on the torus.

### GET /ledger
Get current ledger state.

**Response:**
```json
{
  "state": { "theta": 0.5, "phi": 0.3 },
  "entryCount": 42,
  "failCount": 3,
  "snapHistory": [...],
  "torusParams": { "R": 3.0, "r": 1.0 }
}
```

### POST /ledger/write
Write a fact to the ledger (validates geometric integrity).

**Request:**
```json
{
  "fact": {
    "domain": "physics",
    "content": "Energy equals mass times speed of light squared"
  },
  "domain": "physics"
}
```

**Response (Success):**
```json
{
  "status": "SUCCESS",
  "entry": { "theta": 0.72, "phi": 0.31, "content": "..." },
  "geodesic": { "length": 0.234, "steps": 5 },
  "penalty": { "stutterActive": false }
}
```

**Response (SNAP - Geometric Violation):**
```json
{
  "status": "SNAP_DETECTED",
  "severity": "critical",
  "divergence": 1.234,
  "penalty": {
    "stutterActive": true,
    "timeDilation": 2500
  }
}
```

### GET /ledger/zones
Get semantic zone map.

**Response:**
```json
{
  "zones": [
    { "theta": 0, "semantic": "ABSOLUTE_TRUTH" },
    { "theta": 0.785, "semantic": "HIGH_SECURITY" },
    { "theta": 1.571, "semantic": "TRANSITION_CREATIVE" },
    { "theta": 2.356, "semantic": "CREATIVE_FLUX" },
    { "theta": 3.142, "semantic": "MAXIMUM_FLUX" },
    ...
  ],
  "description": {
    "ABSOLUTE_TRUTH": "Outer equator - maximum verification, time slows",
    "MAXIMUM_FLUX": "Inner equator - rapid exploration, time accelerates"
  }
}
```

### POST /ledger/geodesic
Calculate true geodesic path between two points.

**Request:**
```json
{
  "from": { "theta": 0, "phi": 0 },
  "to": { "theta": 1.0, "phi": 0.5 }
}
```

**Response:**
```json
{
  "from": { "theta": 0, "phi": 0, "zone": "ABSOLUTE_TRUTH" },
  "to": { "theta": 1.0, "phi": 0.5, "zone": "HIGH_SECURITY" },
  "geodesic": {
    "length": 0.567,
    "steps": 8,
    "waypoints": [...]
  },
  "divergence": 0.234,
  "wouldSnap": false,
  "snapSeverity": "none"
}
```

---

## Geodesic Watermark

Authentication where the trajectory shape IS the signature.

### POST /watermark/generate
Generate expected trajectory shape for a message.

**Request:**
```json
{
  "message": "This is protected content",
  "secretKey": "my-secret-key",
  "steps": 8
}
```

**Response:**
```json
{
  "shape": {
    "waypoints": [
      { "theta": 0.12, "phi": 0.34, "zone": "HIGH_SECURITY" },
      { "theta": 0.45, "phi": 0.67, "zone": "TRANSITION" },
      ...
    ],
    "fingerprint": {
      "curvatureProfile": [0.12, 0.15, 0.18, ...],
      "zoneTransitions": ["HIGH_SECURITY->TRANSITION", ...],
      "windingNumbers": { "theta": 0.234, "phi": 0.156 }
    }
  }
}
```

### POST /watermark/verify
Verify observed trajectory against expected.

**Request:**
```json
{
  "message": "This is protected content",
  "secretKey": "my-secret-key",
  "observedTrajectory": [
    { "theta": 0.12, "phi": 0.34 },
    { "theta": 0.45, "phi": 0.67 },
    ...
  ]
}
```

**Response:**
```json
{
  "verified": true,
  "score": 0.95,
  "bandit": false,
  "violations": []
}
```

### POST /watermark/bandit
Detect bandit (imposter) from trajectory alone.

**Request:**
```json
{
  "trajectory": [
    { "theta": 0, "phi": 0 },
    { "theta": 3.14, "phi": 3.14 }  // Impossible jump
  ]
}
```

**Response:**
```json
{
  "bandit": true,
  "violations": [
    { "type": "IMPOSSIBLE_ZONE_JUMP", "severity": "critical" },
    { "type": "DISCONTINUOUS_JUMP", "distance": 4.44 }
  ]
}
```

---

## Neural Synthesizer

Convert cognitive state to audio - hear the AI's thoughts.

### GET /synth/lexicon
Get the conlang lexicon with harmonic mappings.

**Response:**
```json
{
  "baseFrequency": 432,
  "tuning": "432Hz (natural resonance)",
  "lexicon": [
    { "word": "korah", "frequency": 432, "ratio": "1/1", "meaning": "origin" },
    { "word": "aelin", "frequency": 486, "ratio": "9/8", "meaning": "flow" },
    ...
  ],
  "modes": {
    "STRICT": "Odd harmonics only (hollow, clarinet-like)",
    "ADAPTIVE": "All harmonics (rich, full spectrum)",
    "PROBE": "Fundamental only (pure sine)"
  }
}
```

### POST /synth
Full neural sonification.

**Request:**
```json
{
  "phrase": "korah aelin dahru veleth",
  "state": { "emotion": "contemplative", "energy": 0.7 },
  "mode": "ADAPTIVE",
  "duration": 1.0,
  "includeWav": true
}
```

**Response:**
```json
{
  "sequence": ["korah", "aelin", "dahru", "veleth"],
  "fingerprint": {
    "zcr": 0.123,
    "rms": 0.456,
    "centroid": 789.0,
    "hash": "a1b2c3d4"
  },
  "manifoldEmbedding": {
    "angles": [0.12, 0.34, ...],
    "curvature": 0.567
  },
  "audioWav": "data:audio/wav;base64,..."
}
```

### POST /synth/compare
Compare two phrases acoustically.

**Request:**
```json
{
  "phrase1": "korah aelin",
  "phrase2": "dahru veleth"
}
```

**Response:**
```json
{
  "similarity": 0.78,
  "phrase1": { "fingerprint": {...} },
  "phrase2": { "fingerprint": {...} },
  "comparison": {
    "zcrDifference": 0.05,
    "centroidDifference": 45.6
  }
}
```

---

## Languages, Agents & Teams

### GET /languages
Get the six-language codex.

**Response:**
```json
{
  "codex": [
    { "id": "directive", "name": "Directive", "emoji": "...", "allowedZones": ["ABSOLUTE_TRUTH", "HIGH_SECURITY"] },
    { "id": "query", "name": "Query", "emoji": "...", "allowedZones": ["ALL"] },
    { "id": "declarative", "name": "Declarative", "emoji": "...", "allowedZones": ["HIGH_SECURITY", "TRANSITION"] },
    { "id": "speculative", "name": "Speculative", "emoji": "...", "allowedZones": ["CREATIVE_FLUX", "MAXIMUM_FLUX"] },
    { "id": "reflective", "name": "Reflective", "emoji": "...", "allowedZones": ["TRANSITION", "CREATIVE_FLUX"] },
    { "id": "emergent", "name": "Emergent", "emoji": "...", "allowedZones": ["MAXIMUM_FLUX"] }
  ]
}
```

### GET /agents
Get agent archetypes.

**Response:**
```json
{
  "roles": [
    { "id": "guardian", "name": "Guardian", "description": "Enforces semantic integrity", "driftTolerance": 0.1 },
    { "id": "explorer", "name": "Explorer", "description": "Discovers new patterns", "driftTolerance": 0.8 },
    ...
  ]
}
```

### GET /presets
Get team presets.

**Response:**
```json
{
  "presets": [
    { "name": "Security Council", "roles": ["guardian", "validator", "auditor"] },
    { "name": "Creative Workshop", "roles": ["explorer", "synthesizer", "critic"] },
    ...
  ]
}
```

---

## Security

### POST /simulate
Run security simulation.

**Request:**
```json
{
  "participants": 100,
  "breachAttempts": 10
}
```

**Response:**
```json
{
  "participants": 100,
  "breachAttempts": 10,
  "breachFraction": 0.02,
  "resistance": 0.98,
  "geometricIntegrity": "maintained"
}
```

### POST /spin
Quantum spin-based operations.

**Request:**
```json
{
  "action": "generate",
  "strength": 256
}
```

**Response:**
```json
{
  "secret": "...",
  "pair": { "alice": "...", "bob": "..." }
}
```

---

## Error Responses

All error responses follow this format:

```json
{
  "error": "Error description",
  "code": "ERROR_CODE",
  "details": { ... }
}
```

### Common HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 403 | Forbidden (geometric violation, SNAP, bandit detected) |
| 404 | Not Found |
| 500 | Internal Server Error |

---

## Rate Limits

Default Lambda concurrency: 1000 concurrent executions
API Gateway: 10,000 requests/second

---

## Versioning

Current version: **1.0.0**

The API follows semantic versioning. Breaking changes will increment the major version.
