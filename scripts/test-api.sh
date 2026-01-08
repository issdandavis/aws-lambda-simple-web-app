#!/bin/bash
set -e

# Physics Simulation API Test Script
# Usage: ./scripts/test-api.sh <api-endpoint> <api-key>

API_ENDPOINT=${1:-""}
API_KEY=${2:-""}

if [ -z "$API_ENDPOINT" ] || [ -z "$API_KEY" ]; then
    echo "Usage: ./scripts/test-api.sh <api-endpoint> <api-key>"
    echo ""
    echo "Example:"
    echo "  ./scripts/test-api.sh https://abc123.execute-api.us-west-2.amazonaws.com/dev your-api-key"
    exit 1
fi

# Remove trailing slash from endpoint
API_ENDPOINT="${API_ENDPOINT%/}"

echo "================================================"
echo "Physics Simulation API Tests"
echo "Endpoint: $API_ENDPOINT"
echo "================================================"

# Health check
echo ""
echo "1. Health Check"
echo "---------------"
curl -s "${API_ENDPOINT}/health" | jq .

# Quantum: Photon properties
echo ""
echo "2. Quantum: Photon Properties (550nm green light)"
echo "--------------------------------------------------"
curl -s -X POST "${API_ENDPOINT}/simulate" \
    -H "Content-Type: application/json" \
    -H "x-api-key: $API_KEY" \
    -d '{
        "simulationType": "quantum",
        "operation": "photon_properties",
        "parameters": {
            "wavelengthNm": 550
        }
    }' | jq .

# Quantum: Hydrogen energy levels
echo ""
echo "3. Quantum: Hydrogen Energy Levels"
echo "-----------------------------------"
curl -s -X POST "${API_ENDPOINT}/simulate" \
    -H "Content-Type: application/json" \
    -H "x-api-key: $API_KEY" \
    -d '{
        "simulationType": "quantum",
        "operation": "hydrogen_transition",
        "parameters": {
            "nInitial": 3,
            "nFinal": 2
        }
    }' | jq .

# Quantum: Uncertainty principle
echo ""
echo "4. Quantum: Heisenberg Uncertainty Principle"
echo "---------------------------------------------"
curl -s -X POST "${API_ENDPOINT}/simulate" \
    -H "Content-Type: application/json" \
    -H "x-api-key: $API_KEY" \
    -d '{
        "simulationType": "quantum",
        "operation": "uncertainty",
        "parameters": {
            "deltaX": 1e-10
        }
    }' | jq .

# Quantum: Tunneling
echo ""
echo "5. Quantum: Tunneling Through Barrier"
echo "--------------------------------------"
curl -s -X POST "${API_ENDPOINT}/simulate" \
    -H "Content-Type: application/json" \
    -H "x-api-key: $API_KEY" \
    -d '{
        "simulationType": "quantum",
        "operation": "tunneling",
        "parameters": {
            "particleMass": 9.1093837015e-31,
            "particleEnergy": 1e-19,
            "barrierHeight": 2e-19,
            "barrierWidth": 1e-10
        }
    }' | jq .

# Particle: Relativistic properties
echo ""
echo "6. Particle: Relativistic Properties (0.5c)"
echo "--------------------------------------------"
curl -s -X POST "${API_ENDPOINT}/simulate" \
    -H "Content-Type: application/json" \
    -H "x-api-key: $API_KEY" \
    -d '{
        "simulationType": "particle",
        "operation": "relativistic",
        "parameters": {
            "restMass": 9.1093837015e-31,
            "velocity": 149896229
        }
    }' | jq .

# Particle: Earth-Moon gravitational force
echo ""
echo "7. Particle: Earth-Moon Gravitational Force"
echo "--------------------------------------------"
curl -s -X POST "${API_ENDPOINT}/simulate" \
    -H "Content-Type: application/json" \
    -H "x-api-key: $API_KEY" \
    -d '{
        "simulationType": "particle",
        "operation": "gravitational_force",
        "parameters": {
            "m1": 5.972e24,
            "m2": 7.342e22,
            "r": {"x": 3.844e8, "y": 0, "z": 0}
        }
    }' | jq .

# Wave: Blackbody radiation (Sun temperature)
echo ""
echo "8. Wave: Blackbody Radiation (Sun, 5778K)"
echo "------------------------------------------"
curl -s -X POST "${API_ENDPOINT}/simulate" \
    -H "Content-Type: application/json" \
    -H "x-api-key: $API_KEY" \
    -d '{
        "simulationType": "wave",
        "operation": "blackbody",
        "parameters": {
            "temperature": 5778
        }
    }' | jq '.result | {peakWavelength, peakFrequency, totalPower}'

# Wave: Relativistic Doppler
echo ""
echo "9. Wave: Relativistic Doppler Effect"
echo "--------------------------------------"
curl -s -X POST "${API_ENDPOINT}/simulate" \
    -H "Content-Type: application/json" \
    -H "x-api-key: $API_KEY" \
    -d '{
        "simulationType": "wave",
        "operation": "doppler_relativistic",
        "parameters": {
            "sourceFrequency": 5e14,
            "relativeVelocity": 1e7
        }
    }' | jq .

# Constants: Get all
echo ""
echo "10. Constants: Get All Physical Constants"
echo "------------------------------------------"
curl -s -X POST "${API_ENDPOINT}/simulate" \
    -H "Content-Type: application/json" \
    -H "x-api-key: $API_KEY" \
    -d '{
        "simulationType": "constants",
        "operation": "get_all_constants",
        "parameters": {}
    }' | jq '.result | {c, h, G, e, me, mp, kB, NA}'

echo ""
echo "================================================"
echo "All tests completed!"
echo "================================================"
