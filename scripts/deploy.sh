#!/bin/bash
set -e

# Physics Simulation Engine Deployment Script
# Usage: ./scripts/deploy.sh [dev|staging|prod]

STAGE=${1:-dev}

echo "================================================"
echo "Physics Simulation Engine - Deployment"
echo "Stage: $STAGE"
echo "================================================"

# Validate stage
if [[ ! "$STAGE" =~ ^(dev|staging|prod)$ ]]; then
    echo "Error: Invalid stage '$STAGE'. Must be one of: dev, staging, prod"
    exit 1
fi

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed"
    exit 1
fi

if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "Error: AWS credentials not configured or invalid"
    exit 1
fi

echo "Prerequisites OK"

# Install dependencies
echo ""
echo "Installing dependencies..."
npm ci

# Run tests
echo ""
echo "Running tests..."
npm test

# Build TypeScript
echo ""
echo "Building TypeScript..."
npm run build

# Bootstrap CDK (if needed)
echo ""
echo "Bootstrapping CDK..."
npx cdk bootstrap --context stage=$STAGE || true

# Synthesize CloudFormation
echo ""
echo "Synthesizing CloudFormation template..."
npx cdk synth --context stage=$STAGE

# Deploy
echo ""
echo "Deploying to AWS..."
npx cdk deploy --context stage=$STAGE --require-approval never

echo ""
echo "================================================"
echo "Deployment Complete!"
echo "================================================"

# Get outputs
echo ""
echo "Stack Outputs:"
aws cloudformation describe-stacks \
    --stack-name PhysicsSimulationStack-$STAGE \
    --query 'Stacks[0].Outputs' \
    --output table

echo ""
echo "To retrieve your API key value, run:"
echo "  aws apigateway get-api-key --api-key <ApiKeyId> --include-value"
