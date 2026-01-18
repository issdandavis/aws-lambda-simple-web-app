# SCBE-AETHERMOORE Python SDK - Deployment Guide

Complete guide for deploying SCBE-AETHERMOORE Python SDK in production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [PyPI Installation](#pypi-installation)
- [Docker Deployment](#docker-deployment)
- [AWS Lambda Deployment](#aws-lambda-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Environment Configuration](#environment-configuration)
- [Production Checklist](#production-checklist)
- [Monitoring & Logging](#monitoring--logging)
- [Security Hardening](#security-hardening)

---

## Prerequisites

- Python >= 3.10
- pip >= 23.0
- Docker (for containerized deployments)
- kubectl (for Kubernetes)
- AWS CLI (for Lambda deployments)

## PyPI Installation

### From PyPI (when published)

```bash
pip install scbe-aethermoore
```

### From GitHub

```bash
pip install git+https://github.com/issdandavis/aws-lambda-simple-web-app.git
```

### From Source

```bash
git clone https://github.com/issdandavis/aws-lambda-simple-web-app.git
cd aws-lambda-simple-web-app
pip install -e .
```

### Verify Installation

```python
from scbe_production import __version__
print(f"SCBE v{__version__}")

from scbe_production.service import SCBEProductionService
service = SCBEProductionService()
print(service.health_check())
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY symphonic_cipher/ ./symphonic_cipher/
COPY scbe_production/ ./scbe_production/
COPY pyproject.toml .

# Install package
RUN pip install -e .

# Environment
ENV SCBE_ENVIRONMENT=production
ENV SCBE_LOG_LEVEL=INFO
ENV SCBE_AUDIT_ENABLED=true

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from scbe_production.service import SCBEProductionService; SCBEProductionService().health_check()"

CMD ["python", "-m", "uvicorn", "scbe_production.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run

```bash
# Build
docker build -t scbe-aethermoore-python:3.0.0 .

# Run
docker run -d \
  --name scbe-python \
  -p 8000:8000 \
  -e SCBE_ENVIRONMENT=production \
  -e SCBE_PQC_KEM_LEVEL=768 \
  scbe-aethermoore-python:3.0.0

# View logs
docker logs -f scbe-python
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  scbe:
    build: .
    image: scbe-aethermoore-python:3.0.0
    ports:
      - "8000:8000"
    environment:
      SCBE_ENVIRONMENT: production
      SCBE_LOG_LEVEL: INFO
      SCBE_AUDIT_ENABLED: "true"
      SCBE_PQC_KEM_LEVEL: "768"
      SCBE_PQC_DSA_LEVEL: "65"
    volumes:
      - scbe-data:/app/data
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  scbe-data:
```

---

## AWS Lambda Deployment

### Lambda Handler

```python
# lambda_handler.py
import json
from scbe_production.service import SCBEProductionService, AccessRequest
import numpy as np

service = SCBEProductionService()

def handler(event, context):
    """AWS Lambda handler for SCBE operations."""

    operation = event.get('operation', 'health')

    if operation == 'health':
        return {
            'statusCode': 200,
            'body': json.dumps(service.health_check())
        }

    elif operation == 'seal':
        shard = service.seal_memory(
            plaintext=event['data'].encode(),
            agent_id=event['agent_id'],
            topic=event.get('topic', 'default'),
            position=tuple(event.get('position', [1, 2, 3, 5, 8, 13]))
        )
        return {
            'statusCode': 200,
            'body': json.dumps(shard.to_dict())
        }

    elif operation == 'access':
        request = AccessRequest(
            agent_id=event['agent_id'],
            message=event.get('message', ''),
            features=event.get('features', {}),
            position=tuple(event['position'])
        )
        response = service.access_memory(request)
        return {
            'statusCode': 200 if response.decision == 'ALLOW' else 403,
            'body': json.dumps(response.to_dict())
        }

    return {
        'statusCode': 400,
        'body': json.dumps({'error': f'Unknown operation: {operation}'})
    }
```

### SAM Template

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Globals:
  Function:
    Timeout: 30
    Runtime: python3.11
    MemorySize: 512

Resources:
  SCBEFunction:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: scbe-aethermoore
      CodeUri: ./
      Handler: lambda_handler.handler
      Environment:
        Variables:
          SCBE_ENVIRONMENT: production
          SCBE_LOG_LEVEL: INFO
      Events:
        Api:
          Type: Api
          Properties:
            Path: /{proxy+}
            Method: ANY

Outputs:
  ApiEndpoint:
    Value: !Sub "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/"
```

### Deploy

```bash
# Build and deploy
sam build
sam deploy --guided
```

---

## Kubernetes Deployment

### Deployment Manifest

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: scbe-aethermoore
  labels:
    app: scbe
spec:
  replicas: 3
  selector:
    matchLabels:
      app: scbe
  template:
    metadata:
      labels:
        app: scbe
    spec:
      containers:
      - name: scbe
        image: ghcr.io/issdandavis/scbe-aethermoore-python:3.0.0
        ports:
        - containerPort: 8000
        env:
        - name: SCBE_ENVIRONMENT
          value: "production"
        - name: SCBE_LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: scbe-service
spec:
  selector:
    app: scbe
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Deploy to K8s

```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods -l app=scbe
kubectl logs -l app=scbe -f
```

---

## Environment Configuration

### Required Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCBE_ENVIRONMENT` | `production` | Environment (development/staging/production) |
| `SCBE_LOG_LEVEL` | `INFO` | Logging level |
| `SCBE_AUDIT_ENABLED` | `true` | Enable audit logging |

### PQC Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SCBE_PQC_KEM_LEVEL` | `768` | ML-KEM security level (512/768/1024) |
| `SCBE_PQC_DSA_LEVEL` | `65` | ML-DSA security level (44/65/87) |
| `SCBE_PQC_HYBRID` | `true` | Enable hybrid mode |

### Governance Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SCBE_GOV_ALLOW_THRESHOLD` | `0.20` | Risk threshold for ALLOW |
| `SCBE_GOV_QUARANTINE_THRESHOLD` | `0.40` | Risk threshold for QUARANTINE |
| `SCBE_GEOSEAL_INTERIOR_THRESHOLD` | `0.5` | Interior path threshold |

---

## Production Checklist

- [ ] Set `SCBE_ENVIRONMENT=production`
- [ ] Configure PQC security levels (recommend 768/65 minimum)
- [ ] Enable audit logging
- [ ] Set up log aggregation (CloudWatch, ELK, etc.)
- [ ] Configure health check endpoints
- [ ] Set resource limits (memory, CPU)
- [ ] Enable HTTPS/TLS
- [ ] Configure network policies
- [ ] Set up monitoring dashboards
- [ ] Test failover scenarios

---

## Monitoring & Logging

### Structured Logs

All logs are JSON-formatted with audit trail:

```json
{
  "event_id": "uuid",
  "timestamp": "2026-01-18T19:00:00Z",
  "event_type": "governance.allow",
  "service": "scbe-production",
  "risk_score": 0.15,
  "decision": "ALLOW"
}
```

### Metrics to Monitor

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `governance.deny_rate` | Percentage of DENY decisions | > 50% |
| `geoseal.exterior_rate` | Exterior path detections | > 30% |
| `pqc.failures` | PQC operation failures | Any |
| `lattice.timeout_rate` | Consensus timeouts | > 5% |
| `request.latency_p99` | 99th percentile latency | > 2000ms |

---

## Security Hardening

### Network Security

```bash
# Allow only HTTPS
iptables -A INPUT -p tcp --dport 8000 -j DROP
iptables -A INPUT -p tcp --dport 443 -j ACCEPT
```

### Secrets Management

```python
# Use environment variables or secrets manager
import os
password = os.environ.get('SCBE_MASTER_PASSWORD', '').encode()
service = SCBEProductionService(password=password)
```

### Rate Limiting

Implement rate limiting at the API gateway level:
- 100 requests/minute for seal operations
- 1000 requests/minute for access operations

---

## Troubleshooting

### Common Issues

**PQC Backend Warning**
```
PQC Backend: MOCK
```
Install `pqcrypto` for real PQC: `pip install pqcrypto`

**Import Errors**
Ensure all dependencies: `pip install -r requirements.txt`

**Memory Issues**
Increase container memory limits to 512Mi minimum.

---

*For support, open an issue at https://github.com/issdandavis/aws-lambda-simple-web-app/issues*
