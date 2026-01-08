#!/bin/bash
set -e

# Physics Simulation Engine Destroy Script
# Usage: ./scripts/destroy.sh [dev|staging|prod]

STAGE=${1:-dev}

echo "================================================"
echo "Physics Simulation Engine - DESTROY"
echo "Stage: $STAGE"
echo "================================================"

# Validate stage
if [[ ! "$STAGE" =~ ^(dev|staging|prod)$ ]]; then
    echo "Error: Invalid stage '$STAGE'. Must be one of: dev, staging, prod"
    exit 1
fi

# Production safety check
if [ "$STAGE" == "prod" ]; then
    echo ""
    echo "WARNING: You are about to destroy PRODUCTION resources!"
    echo "This action cannot be undone."
    echo ""
    read -p "Type 'destroy-production' to confirm: " CONFIRM
    if [ "$CONFIRM" != "destroy-production" ]; then
        echo "Aborted."
        exit 1
    fi
fi

echo ""
echo "Destroying stack..."
npx cdk destroy --context stage=$STAGE --force

echo ""
echo "================================================"
echo "Stack destroyed successfully"
echo "================================================"
