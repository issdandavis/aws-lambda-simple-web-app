# SCBE-AETHERMOORE AWS Lambda

Serverless deployment of the 14-layer governance pipeline.

## Quick Deploy (2 options)

### Option A: SAM CLI (Recommended)

1. Install SAM CLI:
   ```bash
   pip install aws-sam-cli
   ```

2. Configure AWS credentials:
   ```bash
   aws configure
   # Enter your Access Key ID, Secret Key, region (us-east-1)
   ```

3. Deploy:
   ```bash
   cd lambda
   ./deploy.sh
   ```

### Option B: AWS Console (No CLI needed)

1. **Create deployment package:**
   ```bash
   cd lambda
   pip install numpy -t .
   zip -r scbe-lambda.zip scbe_handler.py numpy*
   ```

2. **Upload to AWS:**
   - Go to [AWS Lambda Console](https://console.aws.amazon.com/lambda/)
   - Click "Create function"
   - Name: `scbe-governance-pipeline`
   - Runtime: Python 3.11
   - Click "Create function"
   - Upload the `scbe-lambda.zip` file
   - Set Handler: `scbe_handler.lambda_handler`

3. **Add API Gateway trigger:**
   - Click "Add trigger"
   - Select "API Gateway"
   - Create new REST API
   - Security: Open (or add API key)
   - Click "Add"

4. **Test:**
   - Copy the API endpoint URL
   - Test with: `curl -X POST <url> -d '{"text":"hello"}'`

## API Usage

**POST /analyze**
```json
{
  "text": "Content to analyze"
}
```

**Response:**
```json
{
  "decision": "ALLOW|QUARANTINE|DENY",
  "risk_score": 0.2847,
  "consensus": "ALLOW",
  "votes": ["ALLOW", "ALLOW", "QUARANTINE"],
  "harmonic_scale": 11390.625,
  "hyperbolic_distance": 1.4523,
  "embedding_norm": 0.6821,
  "processing_ms": 12.34
}
```

## Zapier Integration

Once deployed, use your API Gateway URL in Zapier:
1. Add "Webhooks by Zapier" action
2. Method: POST
3. URL: Your API Gateway endpoint
4. Data: `{"text": "{{trigger_data}}"}`
