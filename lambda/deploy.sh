#!/bin/bash
# SCBE-AETHERMOORE Lambda Deployment Script
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - SAM CLI installed (pip install aws-sam-cli)
#
# Usage:
#   ./deploy.sh                    # Deploy to us-east-1
#   ./deploy.sh us-west-2          # Deploy to specific region

set -e

REGION=${1:-us-east-1}
STACK_NAME="scbe-governance-pipeline"

echo "=== SCBE-AETHERMOORE Lambda Deployment ==="
echo "Region: $REGION"
echo ""

# Build
echo "Building Lambda package..."
sam build --use-container

# Deploy
echo "Deploying to AWS..."
sam deploy \
    --stack-name $STACK_NAME \
    --region $REGION \
    --capabilities CAPABILITY_IAM \
    --resolve-s3 \
    --no-confirm-changeset \
    --no-fail-on-empty-changeset

# Get endpoint URL
echo ""
echo "=== Deployment Complete ==="
aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $REGION \
    --query 'Stacks[0].Outputs[?OutputKey==`SCBEApi`].OutputValue' \
    --output text

echo ""
echo "Test with:"
echo "  curl -X POST <endpoint-url> -H 'Content-Type: application/json' -d '{\"text\": \"test message\"}'"
