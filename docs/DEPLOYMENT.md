# Spiralverse Protocol - Deployment Guide

> Deploy the complete Spiralverse system to AWS

## Prerequisites

1. **AWS Account** with permissions for:
   - Lambda
   - API Gateway
   - S3
   - CloudWatch
   - IAM

2. **AWS CLI** installed and configured:
   ```bash
   # Install AWS CLI
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip
   sudo ./aws/install

   # Configure credentials
   aws configure
   # Enter: Access Key, Secret Key, Region (us-west-2), Output (json)
   ```

3. **AWS SAM CLI** installed:
   ```bash
   pip install aws-sam-cli
   # Or on Mac: brew install aws-sam-cli
   ```

---

## Quick Deploy (Automated)

```bash
# Clone the repository
git clone https://github.com/your-repo/spiralverse-protocol.git
cd spiralverse-protocol

# Run deployment script
./deploy.sh prod us-west-2

# The script will:
# 1. Validate prerequisites
# 2. Generate a secure master key
# 3. Create deployment bucket
# 4. Build and package Lambda
# 5. Deploy CloudFormation stack
# 6. Upload test interface to S3
# 7. Print all URLs
```

---

## Manual Deploy (Step by Step)

### Step 1: Create S3 Bucket for Deployment

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-west-2
BUCKET_NAME="spiralverse-deploy-${ACCOUNT_ID}-${REGION}"

aws s3 mb "s3://${BUCKET_NAME}" --region ${REGION}
```

### Step 2: Build the Lambda Package

```bash
# Using SAM
sam build --template template.yaml

# Or manually create a zip
zip -r function.zip index.js package.json node_modules/
```

### Step 3: Deploy CloudFormation Stack

```bash
# Generate a secure master key
MASTER_KEY=$(openssl rand -base64 32 | tr -d '/+=')
echo "Master Key: ${MASTER_KEY}"
# SAVE THIS KEY SECURELY!

# Deploy with SAM
sam deploy \
    --stack-name spiralverse-prod \
    --s3-bucket "${BUCKET_NAME}" \
    --region ${REGION} \
    --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND \
    --parameter-overrides \
        Environment=prod \
        MasterKey="${MASTER_KEY}" \
    --no-confirm-changeset
```

### Step 4: Get Deployment Outputs

```bash
aws cloudformation describe-stacks \
    --stack-name spiralverse-prod \
    --region ${REGION} \
    --query "Stacks[0].Outputs" \
    --output table
```

### Step 5: Upload Test Interface

```bash
# Get API URL from outputs
API_URL=$(aws cloudformation describe-stacks \
    --stack-name spiralverse-prod \
    --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" \
    --output text)

# Update test interface with API URL
sed "s|API_URL_PLACEHOLDER|${API_URL}|g" test-interface.html > /tmp/index.html

# Upload to S3
TEST_BUCKET="spiralverse-test-ui-${ACCOUNT_ID}-prod"
aws s3 cp /tmp/index.html "s3://${TEST_BUCKET}/index.html" --content-type "text/html"
```

---

## AWS Console Deployment (No CLI)

### 1. Create Lambda Function

1. Go to **AWS Lambda** → **Create function**
2. Settings:
   - Name: `spiralverse-protocol-prod`
   - Runtime: Node.js 18.x
   - Architecture: x86_64
3. Click **Create function**
4. In the code editor, upload `index.js`
5. Go to **Configuration** → **General**:
   - Memory: 512 MB
   - Timeout: 30 seconds
6. Go to **Configuration** → **Environment variables**:
   - Add `MASTER_KEY` with a secure random value
   - Add `ENVIRONMENT` = `prod`

### 2. Create API Gateway

1. Go to **API Gateway** → **Create API**
2. Choose **REST API** → **Build**
3. API name: `spiralverse-api-prod`
4. For each endpoint:
   - Click **Actions** → **Create Resource**
   - Resource name: e.g., `health`, `tesseract`, `ledger`, etc.
   - Click **Actions** → **Create Method** → Choose GET or POST
   - Integration type: Lambda Function
   - Lambda Function: `spiralverse-protocol-prod`

5. Enable CORS:
   - Select each resource
   - Click **Actions** → **Enable CORS**
   - Accept defaults

6. Deploy:
   - Click **Actions** → **Deploy API**
   - Stage name: `prod`

### 3. Create CloudWatch Dashboard

1. Go to **CloudWatch** → **Dashboards** → **Create dashboard**
2. Name: `Spiralverse-prod`
3. Add widgets:
   - **Line chart**: API Gateway Count metric
   - **Line chart**: API Gateway Latency metric
   - **Number**: Lambda Errors
   - **Logs table**: Filter for "SNAP" events

### 4. Upload Test Interface to S3

1. Go to **S3** → **Create bucket**
2. Name: `spiralverse-test-ui-{account-id}-prod`
3. Uncheck "Block all public access"
4. Enable static website hosting
5. Upload `test-interface.html` as `index.html`
6. Make the object public

---

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MASTER_KEY` | Cryptographic master key | Required |
| `ENVIRONMENT` | dev/staging/prod | prod |

### Lambda Settings

| Setting | Recommended Value |
|---------|-------------------|
| Memory | 512 MB |
| Timeout | 30 seconds |
| Tracing | Active (X-Ray) |

### API Gateway Settings

| Setting | Recommended Value |
|---------|-------------------|
| Throttling | 10,000 req/sec |
| CORS | Enabled |
| API Key | Optional |

---

## Verify Deployment

```bash
# Quick health check
curl https://your-api-id.execute-api.us-west-2.amazonaws.com/prod/health

# Expected response:
# {"status":"healthy","protocol":"spiralverse","version":"1.0.0"}

# Run comprehensive tests
API_URL=https://your-api-id.execute-api.us-west-2.amazonaws.com/prod ./test-all.sh
```

---

## Troubleshooting

### Lambda Timeout
- Increase timeout to 30 seconds
- Check CloudWatch logs for slow operations

### CORS Errors
- Ensure API Gateway has CORS enabled on all resources
- Check that OPTIONS method returns correct headers

### 502 Bad Gateway
- Check Lambda execution role has required permissions
- Look at CloudWatch logs for error details

### "Internal Server Error"
- Check Lambda logs in CloudWatch
- Verify environment variables are set

---

## Cleanup

To remove all resources:

```bash
# Delete CloudFormation stack
aws cloudformation delete-stack --stack-name spiralverse-prod --region us-west-2

# Wait for deletion
aws cloudformation wait stack-delete-complete --stack-name spiralverse-prod

# Delete S3 buckets (must be empty first)
aws s3 rm s3://spiralverse-test-ui-xxx-prod --recursive
aws s3 rb s3://spiralverse-test-ui-xxx-prod
```

---

## Next Steps

After deployment:

1. **Test all endpoints** using the test interface or `test-all.sh`
2. **Monitor** the CloudWatch dashboard for errors and latency
3. **Set up alerts** for high error rates or latency
4. **Document your API key** if you enabled authentication
5. **Consider** adding a custom domain via Route 53

See `API_REFERENCE.md` for complete endpoint documentation.
