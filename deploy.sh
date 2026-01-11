#!/bin/bash
# =============================================================================
# Spiralverse Protocol - AWS Deployment Script
# =============================================================================
# This script deploys the complete Spiralverse system to AWS
#
# Prerequisites:
#   1. AWS CLI installed: https://aws.amazon.com/cli/
#   2. AWS SAM CLI installed: https://aws.amazon.com/serverless/sam/
#   3. AWS credentials configured: aws configure
#
# Usage:
#   ./deploy.sh [environment] [region]
#
# Examples:
#   ./deploy.sh              # Deploy to prod in us-west-2
#   ./deploy.sh dev          # Deploy to dev in us-west-2
#   ./deploy.sh prod us-east-1  # Deploy to prod in us-east-1
# =============================================================================

set -e  # Exit on error

# Configuration
ENVIRONMENT="${1:-prod}"
REGION="${2:-us-west-2}"
STACK_NAME="spiralverse-${ENVIRONMENT}"
S3_BUCKET="spiralverse-deploy-${ENVIRONMENT}-$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo 'unknown')"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           SPIRALVERSE PROTOCOL - AWS DEPLOYMENT                  ║${NC}"
echo -e "${BLUE}║       16 Vertices of Universal Truth in the Cloud               ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}[1/8] Checking prerequisites...${NC}"

if ! command -v aws &> /dev/null; then
    echo -e "${RED}ERROR: AWS CLI not installed${NC}"
    echo "Install: https://aws.amazon.com/cli/"
    exit 1
fi

if ! command -v sam &> /dev/null; then
    echo -e "${RED}ERROR: AWS SAM CLI not installed${NC}"
    echo "Install: https://aws.amazon.com/serverless/sam/"
    echo ""
    echo "Quick install (if you have pip):"
    echo "  pip install aws-sam-cli"
    exit 1
fi

# Verify AWS credentials
echo -e "${YELLOW}[2/8] Verifying AWS credentials...${NC}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
if [ -z "$ACCOUNT_ID" ]; then
    echo -e "${RED}ERROR: AWS credentials not configured${NC}"
    echo "Run: aws configure"
    exit 1
fi
echo -e "${GREEN}  Account: ${ACCOUNT_ID}${NC}"
echo -e "${GREEN}  Region: ${REGION}${NC}"
echo -e "${GREEN}  Environment: ${ENVIRONMENT}${NC}"

# Generate master key if not provided
echo -e "${YELLOW}[3/8] Generating secure master key...${NC}"
if [ -z "$MASTER_KEY" ]; then
    MASTER_KEY=$(openssl rand -base64 32 | tr -d '/+=')
    echo -e "${GREEN}  Generated new master key (save this securely!)${NC}"
    echo -e "${BLUE}  MASTER_KEY=${MASTER_KEY}${NC}"
else
    echo -e "${GREEN}  Using provided MASTER_KEY${NC}"
fi

# Create S3 bucket for deployment artifacts
echo -e "${YELLOW}[4/8] Creating deployment bucket...${NC}"
S3_BUCKET="spiralverse-deploy-${ACCOUNT_ID}-${REGION}"
if aws s3 ls "s3://${S3_BUCKET}" 2>&1 | grep -q 'NoSuchBucket'; then
    aws s3 mb "s3://${S3_BUCKET}" --region "${REGION}"
    echo -e "${GREEN}  Created bucket: ${S3_BUCKET}${NC}"
else
    echo -e "${GREEN}  Bucket exists: ${S3_BUCKET}${NC}"
fi

# Build and package
echo -e "${YELLOW}[5/8] Building and packaging Lambda function...${NC}"
sam build --template template.yaml

# Deploy with SAM
echo -e "${YELLOW}[6/8] Deploying to AWS (this may take 2-3 minutes)...${NC}"
sam deploy \
    --stack-name "${STACK_NAME}" \
    --s3-bucket "${S3_BUCKET}" \
    --region "${REGION}" \
    --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND \
    --parameter-overrides \
        Environment="${ENVIRONMENT}" \
        MasterKey="${MASTER_KEY}" \
    --no-confirm-changeset \
    --no-fail-on-empty-changeset

# Get outputs
echo -e "${YELLOW}[7/8] Retrieving deployment outputs...${NC}"
API_URL=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
    --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" \
    --output text)

TEST_UI_URL=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
    --query "Stacks[0].Outputs[?OutputKey=='TestInterfaceUrl'].OutputValue" \
    --output text)

DASHBOARD_URL=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
    --query "Stacks[0].Outputs[?OutputKey=='DashboardUrl'].OutputValue" \
    --output text)

# Upload test interface to S3
echo -e "${YELLOW}[8/8] Uploading test interface...${NC}"
if [ -f "test-interface.html" ]; then
    # Update API URL in test interface
    sed "s|API_URL_PLACEHOLDER|${API_URL}|g" test-interface.html > /tmp/index.html

    TEST_BUCKET="spiralverse-test-ui-${ACCOUNT_ID}-${ENVIRONMENT}"
    aws s3 cp /tmp/index.html "s3://${TEST_BUCKET}/index.html" --content-type "text/html"
    aws s3 cp error.html "s3://${TEST_BUCKET}/error.html" --content-type "text/html" 2>/dev/null || true
    echo -e "${GREEN}  Test interface uploaded${NC}"
fi

# Print summary
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    DEPLOYMENT SUCCESSFUL!                        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}API Endpoint:${NC}"
echo "  ${API_URL}"
echo ""
echo -e "${BLUE}Test Interface:${NC}"
echo "  ${TEST_UI_URL}"
echo ""
echo -e "${BLUE}CloudWatch Dashboard:${NC}"
echo "  ${DASHBOARD_URL}"
echo ""
echo -e "${BLUE}Quick Test:${NC}"
echo "  curl ${API_URL}/health"
echo "  curl ${API_URL}/tesseract"
echo ""
echo -e "${YELLOW}Master Key (SAVE THIS SECURELY):${NC}"
echo "  ${MASTER_KEY}"
echo ""
echo -e "${BLUE}To run comprehensive tests:${NC}"
echo "  API_URL=${API_URL} ./test-all.sh"
echo ""

# Save deployment info
cat > deployment-info.json << EOF
{
  "stackName": "${STACK_NAME}",
  "environment": "${ENVIRONMENT}",
  "region": "${REGION}",
  "accountId": "${ACCOUNT_ID}",
  "apiUrl": "${API_URL}",
  "testInterfaceUrl": "${TEST_UI_URL}",
  "dashboardUrl": "${DASHBOARD_URL}",
  "deployedAt": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
echo -e "${GREEN}Deployment info saved to: deployment-info.json${NC}"
