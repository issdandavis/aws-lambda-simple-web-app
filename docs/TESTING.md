# Spiralverse Protocol - Testing Guide

> Comprehensive testing for geometric AI integrity

## Quick Start

```bash
# Set API URL
export API_URL=https://your-api.execute-api.us-west-2.amazonaws.com/prod

# Run all tests
./test-all.sh

# Or for local testing
./test-all.sh local
```

---

## Test Categories

### 1. Core Health Checks

```bash
# Health endpoint
curl $API_URL/health

# Expected: {"status":"healthy","protocol":"spiralverse"}

# Geometry info
curl $API_URL/geometry
```

### 2. Tesseract Core Tests

#### Universal Constants
```bash
# Get tesseract structure
curl $API_URL/tesseract

# Get all 16 constants with dimensions
curl $API_URL/tesseract/constants

# Get reasoning lattice
curl $API_URL/tesseract/lattice
```

#### Number Parsing
```bash
# Parse exponential notation
curl -X POST $API_URL/tesseract/parse \
  -H "Content-Type: application/json" \
  -d '{"input": "6.62607015e-34"}'

# Parse negative numbers
curl -X POST $API_URL/tesseract/parse \
  -H "Content-Type: application/json" \
  -d '{"input": "-3.5E+10"}'

# Parse array of numbers
curl -X POST $API_URL/tesseract/parse \
  -H "Content-Type: application/json" \
  -d '{"inputs": [42, -17.5, "1e-10", "299792458"]}'
```

#### Dimensional Analysis
```bash
# Analyze velocity (c)
curl -X POST $API_URL/tesseract/analyze \
  -H "Content-Type: application/json" \
  -d '{"expression": "c"}'
# Expected dimensions: L·T^-1

# Analyze c * h
curl -X POST $API_URL/tesseract/analyze \
  -H "Content-Type: application/json" \
  -d '{"expression": "c * h"}'
# Expected: L^3·M·T^-2

# Calculate pi * phi
curl -X POST $API_URL/tesseract/calculate \
  -H "Content-Type: application/json" \
  -d '{"expression": "pi * phi"}'
```

### 3. Geometric Ledger Tests

#### Valid Operations
```bash
# Get ledger state
curl $API_URL/ledger

# Write a valid fact
curl -X POST $API_URL/ledger/write \
  -H "Content-Type: application/json" \
  -d '{
    "fact": {"domain": "physics", "content": "E=mc²"},
    "domain": "physics"
  }'

# Get semantic zones
curl $API_URL/ledger/zones
```

#### SNAP Detection (Intentional Violation)
```bash
# This should trigger SNAP detection (large geometric jump)
curl -X POST $API_URL/ledger/geodesic \
  -H "Content-Type: application/json" \
  -d '{
    "from": {"theta": 0, "phi": 0},
    "to": {"theta": 3.14159, "phi": 3.14159}
  }'

# Expected: "wouldSnap": true, "snapSeverity": "critical" or "catastrophic"
```

### 4. Watermark Authentication Tests

#### Generate Watermark
```bash
curl -X POST $API_URL/watermark/generate \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Protected content from Spiralverse",
    "secretKey": "my-secret-key-123",
    "steps": 8
  }'
```

#### Verify Valid Trajectory
```bash
# First generate expected shape, then verify with matching trajectory
curl -X POST $API_URL/watermark/verify \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Protected content",
    "secretKey": "my-secret",
    "observedTrajectory": [
      {"theta": 0.12, "phi": 0.34},
      {"theta": 0.45, "phi": 0.67},
      {"theta": 0.78, "phi": 0.90}
    ]
  }'
```

#### Bandit Detection (Imposter)
```bash
# Impossible trajectory (should detect bandit)
curl -X POST $API_URL/watermark/bandit \
  -H "Content-Type: application/json" \
  -d '{
    "trajectory": [
      {"theta": 0, "phi": 0},
      {"theta": 3.14, "phi": 3.14}
    ]
  }'
# Expected: "bandit": true, "violations": [...]
```

### 5. Neural Synthesizer Tests

```bash
# Get conlang lexicon
curl $API_URL/synth/lexicon

# Sonify a phrase (without WAV to reduce size)
curl -X POST $API_URL/synth \
  -H "Content-Type: application/json" \
  -d '{
    "phrase": "korah aelin dahru veleth",
    "mode": "ADAPTIVE",
    "duration": 0.5,
    "includeWav": false
  }'

# Compare two phrases
curl -X POST $API_URL/synth/compare \
  -H "Content-Type: application/json" \
  -d '{
    "phrase1": "korah aelin",
    "phrase2": "dahru veleth"
  }'
```

### 6. Languages, Agents, Teams

```bash
# Six languages
curl $API_URL/languages

# Agent archetypes
curl $API_URL/agents

# Team presets
curl $API_URL/presets

# Healing status
curl $API_URL/healing
```

### 7. Security Simulation

```bash
# Run security simulation
curl -X POST $API_URL/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "participants": 100,
    "breachAttempts": 10
  }'
# Expected: "breachFraction" should be very low (< 0.05)
```

---

## Expected Test Results

| Test | Expected Outcome |
|------|------------------|
| Health check | `{"status": "healthy"}` |
| Tesseract vertices | 16 vertices |
| Tesseract edges | 32 edges |
| Tesseract cells | 8 cells |
| Parse exponential | Valid, correct value |
| Dimensional analysis (c) | L·T^-1 |
| Dimensional analysis (c*h) | L^3·M·T^-2 |
| Ledger write (valid) | SUCCESS |
| Ledger geodesic (large jump) | SNAP detected |
| Watermark generate | Shape + fingerprint |
| Bandit detection (impossible) | Bandit = true |
| Synth lexicon | 432Hz base, word list |
| Security simulation | breachFraction < 0.05 |

---

## Integration Testing

### Test Scenario: Complete Workflow

```bash
#!/bin/bash
# test-workflow.sh - Complete integration test

API_URL=${API_URL:-http://localhost:3000}

echo "=== Spiralverse Integration Test ==="

# 1. Check health
echo -n "1. Health check... "
health=$(curl -s $API_URL/health)
if echo "$health" | grep -q "healthy"; then
    echo "OK"
else
    echo "FAILED"
    exit 1
fi

# 2. Create mission environment
echo -n "2. Creating environment... "
env=$(curl -s -X POST $API_URL/tesseract/environment \
    -H "Content-Type: application/json" \
    -d '{"semanticWeight": 1.2, "viscosity": 0.1}')
if echo "$env" | grep -q "environment"; then
    echo "OK"
else
    echo "FAILED"
    exit 1
fi

# 3. Write fact to ledger
echo -n "3. Writing to ledger... "
write=$(curl -s -X POST $API_URL/ledger/write \
    -H "Content-Type: application/json" \
    -d '{"fact": {"domain": "test", "content": "Integration test"}}')
if echo "$write" | grep -q "SUCCESS\|entry"; then
    echo "OK"
else
    echo "FAILED (but may be expected if SNAP)"
fi

# 4. Generate watermark
echo -n "4. Generating watermark... "
wm=$(curl -s -X POST $API_URL/watermark/generate \
    -H "Content-Type: application/json" \
    -d '{"message": "Test", "secretKey": "key", "steps": 4}')
if echo "$wm" | grep -q "fingerprint"; then
    echo "OK"
else
    echo "FAILED"
    exit 1
fi

# 5. Sonify phrase
echo -n "5. Sonifying phrase... "
synth=$(curl -s -X POST $API_URL/synth \
    -H "Content-Type: application/json" \
    -d '{"phrase": "korah aelin", "includeWav": false}')
if echo "$synth" | grep -q "sequence"; then
    echo "OK"
else
    echo "FAILED"
    exit 1
fi

echo ""
echo "=== Integration Test PASSED ==="
```

---

## Performance Testing

### Using Apache Benchmark (ab)

```bash
# Install ab
sudo apt-get install apache2-utils

# Test health endpoint (1000 requests, 10 concurrent)
ab -n 1000 -c 10 $API_URL/health

# Test POST endpoint
ab -n 100 -c 5 -p payload.json -T "application/json" $API_URL/tesseract/parse
```

### Using wrk

```bash
# Install wrk
sudo apt-get install wrk

# Benchmark for 30 seconds with 10 connections
wrk -t2 -c10 -d30s $API_URL/health
```

---

## Troubleshooting Failed Tests

### Test Fails with 403
- Check for SNAP detection (geometric violation)
- Check for bandit detection
- Review the error message for details

### Test Fails with 500
- Check CloudWatch logs for Lambda errors
- Verify environment variables are set
- Check for JavaScript errors in the code

### Test Fails with CORS Error
- Ensure API Gateway has CORS enabled
- Check browser console for specific error
- Verify allowed origins include your test interface

### Timeout Errors
- Increase Lambda timeout (30s recommended)
- Check for infinite loops in code
- Monitor Lambda duration in CloudWatch

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Spiralverse Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: npm install
      - run: npm test
      - run: |
          export API_URL=${{ secrets.API_URL }}
          ./test-all.sh
```
