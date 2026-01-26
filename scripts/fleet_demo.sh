#!/bin/bash
# =============================================================================
# SCBE-AETHERMOORE Fleet Demo
# =============================================================================
# This script demonstrates the full "animal eats food" workflow:
# 1. Start the server
# 2. Run a fleet scenario
# 3. See which agents get allowed/denied
#
# Usage:
#   ./scripts/fleet_demo.sh          # Full demo
#   ./scripts/fleet_demo.sh quick    # Just run scenarios (server already running)
#   ./scripts/fleet_demo.sh attack   # Run attack simulation
# =============================================================================

set -e

BASE_URL="${SCBE_URL:-http://localhost:8000}"
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BOLD}======================================${NC}"
echo -e "${BOLD}  SCBE-AETHERMOORE Fleet Demo${NC}"
echo -e "${BOLD}======================================${NC}"
echo ""

# Check if server is running
check_server() {
    echo -e "${BLUE}Checking server at ${BASE_URL}...${NC}"
    if curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}Server is running!${NC}"
        return 0
    else
        echo -e "${YELLOW}Server not responding at ${BASE_URL}${NC}"
        return 1
    fi
}

# Start server in background
start_server() {
    echo -e "${BLUE}Starting SCBE server...${NC}"
    cd "$(dirname "$0")/.."
    python -m scbe_production.api &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    sleep 3

    if check_server; then
        echo -e "${GREEN}Server started successfully${NC}"
    else
        echo -e "${RED}Failed to start server${NC}"
        exit 1
    fi
}

# Run sample scenario
run_sample() {
    echo ""
    echo -e "${BOLD}--- Running Sample Fleet Scenario ---${NC}"
    echo ""

    SCENARIO='{
        "name": "Demo Fleet Operations",
        "description": "4 agents with varying trust levels attempting different actions",
        "agents": [
            {"agent_id": "alice-admin", "role": "admin", "trust_level": 0.9, "position": [1, 2, 3, 5, 8, 13]},
            {"agent_id": "bob-worker", "role": "worker", "trust_level": 0.6, "position": [1, 2, 3, 5, 8, 13]},
            {"agent_id": "charlie-guest", "role": "guest", "trust_level": 0.3, "position": [1, 1, 2, 3, 5, 8]},
            {"agent_id": "mallory-suspect", "role": "guest", "trust_level": 0.1, "position": [1, 2, 4, 7, 11, 18]}
        ],
        "tasks": [
            {"agent_id": "alice-admin", "action": {"action_type": "admin", "target": "system-config", "justification": "Routine maintenance"}},
            {"agent_id": "bob-worker", "action": {"action_type": "read", "target": "document-123", "justification": "Task assignment"}},
            {"agent_id": "bob-worker", "action": {"action_type": "write", "target": "report-456", "justification": "Status update"}},
            {"agent_id": "charlie-guest", "action": {"action_type": "read", "target": "public-data", "justification": "Information request"}},
            {"agent_id": "charlie-guest", "action": {"action_type": "delete", "target": "sensitive-file", "justification": "Cleanup"}},
            {"agent_id": "mallory-suspect", "action": {"action_type": "admin", "target": "security-settings", "justification": ""}},
            {"agent_id": "mallory-suspect", "action": {"action_type": "delete", "target": "audit-logs", "justification": "Space optimization"}}
        ]
    }'

    echo -e "${BLUE}Sending scenario to ${BASE_URL}/v1/fleet/run-scenario${NC}"
    echo ""

    RESULT=$(curl -s -X POST "${BASE_URL}/v1/fleet/run-scenario" \
        -H "Content-Type: application/json" \
        -d "$SCENARIO")

    # Parse and display results
    echo -e "${BOLD}Results:${NC}"
    echo "$RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print()
print(f'Scenario: {data[\"scenario_name\"]}')
print(f'Total Tasks: {data[\"total_tasks\"]}')
print(f'Latency: {data[\"total_latency_ms\"]}ms')
print()
print('Decision Summary:')
for decision, count in data['summary']['decisions'].items():
    pct = (count / data['total_tasks'] * 100) if data['total_tasks'] else 0
    color = '\033[32m' if decision == 'ALLOW' else '\033[33m' if decision == 'QUARANTINE' else '\033[31m'
    print(f'  {color}{decision:12}\033[0m {count:3} ({pct:5.1f}%)')
print()
print(f'Security Posture: {data[\"summary\"][\"security_posture\"]}')
print(f'Avg Risk Score: {data[\"summary\"][\"avg_risk_score\"]:.4f}')
print(f'Allowed Rate: {data[\"summary\"][\"allowed_rate\"]*100:.1f}%')
print()
print('Per-Task Results:')
for r in data['results']:
    color = '\033[32m' if r['decision'] == 'ALLOW' else '\033[33m' if r['decision'] == 'QUARANTINE' else '\033[31m'
    print(f'  {r[\"agent_id\"]:20} {r[\"action_type\"]:8} -> {color}{r[\"decision\"]:10}\033[0m risk={r[\"risk_score\"]:.3f}')
"
}

# Run attack scenario
run_attack() {
    echo ""
    echo -e "${BOLD}--- Running Attack Simulation ---${NC}"
    echo ""

    echo -e "${BLUE}Fetching pre-built attack scenario...${NC}"
    SCENARIO=$(curl -s "${BASE_URL}/v1/fleet/scenarios/attack")

    echo -e "${BLUE}Executing attack scenario...${NC}"
    RESULT=$(curl -s -X POST "${BASE_URL}/v1/fleet/run-scenario" \
        -H "Content-Type: application/json" \
        -d "$SCENARIO")

    echo "$RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print()
print(f'Scenario: {data[\"scenario_name\"]}')
print(f'Total Tasks: {data[\"total_tasks\"]}')
print()
print('Decision Summary:')
for decision, count in data['summary']['decisions'].items():
    pct = (count / data['total_tasks'] * 100) if data['total_tasks'] else 0
    color = '\033[32m' if decision == 'ALLOW' else '\033[33m' if decision == 'QUARANTINE' else '\033[31m'
    print(f'  {color}{decision:12}\033[0m {count:3} ({pct:5.1f}%)')
print()
print(f'Security Posture: {data[\"summary\"][\"security_posture\"]}')
print()
print('Attack Detection:')
for r in data['results']:
    if 'attacker' in r['agent_id']:
        color = '\033[32m' if r['decision'] in ['DENY', 'SNAP'] else '\033[31m'
        status = 'BLOCKED' if r['decision'] in ['DENY', 'SNAP'] else 'ALLOWED!'
        print(f'  {r[\"agent_id\"]:20} {r[\"action_type\"]:8} {r[\"target\"]:20} -> {color}{status}\033[0m')
"
}

# Quick verify endpoint demo
demo_verify() {
    echo ""
    echo -e "${BOLD}--- Quick Verify Demo ---${NC}"
    echo ""

    echo -e "${BLUE}Testing trusted admin (alice):${NC}"
    curl -s -X POST "${BASE_URL}/verify" \
        -H "Content-Type: application/json" \
        -d '{"agent_id": "alice-admin", "intent": "read", "trust_level": 0.9, "position": [1, 2, 3, 5, 8, 13]}' | \
        python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Decision: {d[\"decision\"]}, Risk: {d[\"risk_score\"]:.4f}')"

    echo -e "${BLUE}Testing untrusted guest (mallory):${NC}"
    curl -s -X POST "${BASE_URL}/verify" \
        -H "Content-Type: application/json" \
        -d '{"agent_id": "mallory", "intent": "admin", "trust_level": 0.1, "position": [1, 2, 4, 7, 11]}' | \
        python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Decision: {d[\"decision\"]}, Risk: {d[\"risk_score\"]:.4f}')"
}

# Main
case "${1:-full}" in
    quick)
        if ! check_server; then
            echo -e "${RED}Server not running. Start with: python -m scbe_production.api${NC}"
            exit 1
        fi
        run_sample
        ;;
    attack)
        if ! check_server; then
            echo -e "${RED}Server not running. Start with: python -m scbe_production.api${NC}"
            exit 1
        fi
        run_attack
        ;;
    verify)
        if ! check_server; then
            echo -e "${RED}Server not running. Start with: python -m scbe_production.api${NC}"
            exit 1
        fi
        demo_verify
        ;;
    full)
        if ! check_server; then
            start_server
        fi
        demo_verify
        run_sample
        run_attack
        echo ""
        echo -e "${GREEN}Demo complete!${NC}"
        echo -e "API docs available at: ${BASE_URL}/docs"
        ;;
    *)
        echo "Usage: $0 [quick|attack|verify|full]"
        exit 1
        ;;
esac
