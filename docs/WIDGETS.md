# Spiralverse Protocol - CloudWatch Dashboard Guide

> Monitor your geometric AI operating system

## Dashboard Overview

The Spiralverse CloudWatch dashboard provides real-time visibility into:
- API request patterns
- Latency and performance
- Error rates and types
- SNAP events (geometric violations)
- Lambda resource usage

---

## Dashboard Widgets

### 1. Request Count by Endpoint

**Type:** Line Chart
**Metric:** AWS/ApiGateway → Count

Shows the number of requests over time. Use this to:
- Identify traffic patterns
- Detect unexpected spikes
- Plan capacity

**CloudWatch Query:**
```
{
  "metrics": [
    ["AWS/ApiGateway", "Count", "ApiName", "spiralverse-api-prod", {"stat": "Sum", "period": 300}]
  ],
  "title": "Request Count"
}
```

### 2. Average Latency

**Type:** Line Chart
**Metrics:**
- Latency (total response time)
- IntegrationLatency (Lambda execution time)

Shows how fast your API responds. Key insights:
- Latency spike = potential issue
- IntegrationLatency > Latency difference = API Gateway overhead
- Normal: < 500ms for most operations

**CloudWatch Query:**
```
{
  "metrics": [
    ["AWS/ApiGateway", "Latency", "ApiName", "spiralverse-api-prod", {"stat": "Average"}],
    ["AWS/ApiGateway", "IntegrationLatency", "ApiName", "spiralverse-api-prod", {"stat": "Average"}]
  ],
  "title": "Latency"
}
```

### 3. Error Rates

**Type:** Line Chart
**Metrics:**
- 4XXError (client errors - bad requests)
- 5XXError (server errors - bugs/crashes)

**Interpretation:**
- 4XX = Client sending bad data (expected occasionally)
- 5XX = Server-side issue (should be near zero)
- Spike in 5XX = immediate investigation needed

**Thresholds:**
- 4XX: Alert if > 10% of requests
- 5XX: Alert if > 1% of requests

### 4. Lambda Duration

**Type:** Line Chart
**Metrics:**
- Duration (Average)
- Duration (Maximum)

Shows Lambda execution time:
- Average should be < 1000ms
- Maximum spikes indicate complex operations
- Consistent high maximum = potential timeout risk

### 5. Lambda Invocations & Errors

**Type:** Line Chart
**Metrics:**
- Invocations (total calls)
- Errors (failed executions)

**Key Insights:**
- Error/Invocation ratio = error rate
- Spike in errors = code issue or resource problem
- Normal: Error rate < 0.1%

### 6. SNAP Events Log

**Type:** Log Table
**Source:** Lambda logs

Shows geometric violations detected:
```
FIELDS @timestamp, @message
| FILTER @message LIKE /SNAP|snap|Snap/
| SORT @timestamp DESC
| LIMIT 50
```

**SNAP Severity Levels:**
- `warning`: Minor deviation (0.3-0.5)
- `snap`: Moderate violation (0.5-0.8)
- `critical`: Severe violation (0.8-1.5)
- `catastrophic`: Complete breakdown (>1.5)

---

## Creating Custom Widgets

### SNAP Counter Widget

```json
{
  "type": "log",
  "properties": {
    "query": "SOURCE '/aws/lambda/spiralverse-protocol-prod' | stats count(*) as snap_count by bin(1h) | filter @message like /SNAP/",
    "title": "SNAP Events per Hour"
  }
}
```

### Latency by Endpoint

```json
{
  "type": "metric",
  "properties": {
    "metrics": [
      ["AWS/ApiGateway", "Latency", "ApiName", "spiralverse-api-prod", "Resource", "/tesseract"],
      ["AWS/ApiGateway", "Latency", "ApiName", "spiralverse-api-prod", "Resource", "/ledger"],
      ["AWS/ApiGateway", "Latency", "ApiName", "spiralverse-api-prod", "Resource", "/synth"]
    ],
    "title": "Latency by Endpoint"
  }
}
```

### Semantic Zone Distribution

Track which zones are most accessed:
```json
{
  "type": "log",
  "properties": {
    "query": "SOURCE '/aws/lambda/spiralverse-protocol-prod' | parse @message /zone.*\"([A-Z_]+)\"/ as zone | stats count(*) by zone",
    "title": "Requests by Semantic Zone"
  }
}
```

### Bandit Detection Rate

```json
{
  "type": "log",
  "properties": {
    "query": "SOURCE '/aws/lambda/spiralverse-protocol-prod' | filter @message like /bandit.*true/ | stats count(*) as bandits by bin(1h)",
    "title": "Bandit Detections per Hour"
  }
}
```

---

## Setting Up Alarms

### High Error Rate Alarm

```bash
aws cloudwatch put-metric-alarm \
    --alarm-name "Spiralverse-HighErrors" \
    --alarm-description "High error rate in API" \
    --metric-name "5XXError" \
    --namespace "AWS/ApiGateway" \
    --statistic Sum \
    --period 300 \
    --threshold 10 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --dimensions Name=ApiName,Value=spiralverse-api-prod \
    --alarm-actions arn:aws:sns:us-west-2:ACCOUNT:alerts
```

### High Latency Alarm

```bash
aws cloudwatch put-metric-alarm \
    --alarm-name "Spiralverse-HighLatency" \
    --alarm-description "API latency exceeds threshold" \
    --metric-name "Latency" \
    --namespace "AWS/ApiGateway" \
    --statistic Average \
    --period 300 \
    --threshold 5000 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --dimensions Name=ApiName,Value=spiralverse-api-prod \
    --alarm-actions arn:aws:sns:us-west-2:ACCOUNT:alerts
```

### SNAP Event Alarm

Create a metric filter first:
```bash
aws logs put-metric-filter \
    --log-group-name "/aws/lambda/spiralverse-protocol-prod" \
    --filter-name "SnapEvents" \
    --filter-pattern "SNAP" \
    --metric-transformations \
        metricName=SnapEvents,metricNamespace=Spiralverse,metricValue=1

aws cloudwatch put-metric-alarm \
    --alarm-name "Spiralverse-SnapEvents" \
    --alarm-description "Geometric violations detected" \
    --metric-name "SnapEvents" \
    --namespace "Spiralverse" \
    --statistic Sum \
    --period 300 \
    --threshold 5 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 1 \
    --alarm-actions arn:aws:sns:us-west-2:ACCOUNT:alerts
```

---

## X-Ray Tracing

X-Ray provides distributed tracing for debugging:

### Enable X-Ray

Already enabled in the SAM template with `Tracing: Active`.

### View Traces

1. Go to **X-Ray** in AWS Console
2. Click **Traces**
3. Filter by:
   - Service: `spiralverse-protocol-prod`
   - Time range
   - Status (Error/Fault)

### Trace Annotations

The Lambda adds annotations for:
- Endpoint path
- Request method
- SNAP events
- Semantic zone

---

## Lambda Insights

Enhanced monitoring for Lambda:

### Enable Lambda Insights

1. Go to **Lambda** → Your function
2. **Configuration** → **Monitoring tools**
3. Enable **Lambda Insights**

### Insights Metrics

- Memory utilization
- CPU time
- Network throughput
- Cold start frequency
- Init duration

---

## Cost Monitoring

### Estimated Costs

| Component | Cost Factor |
|-----------|-------------|
| Lambda | $0.20 per 1M requests + duration |
| API Gateway | $3.50 per 1M requests |
| CloudWatch | $0.30 per GB logs |
| X-Ray | $5 per 1M traces |

### Cost Alarm

```bash
aws cloudwatch put-metric-alarm \
    --alarm-name "Spiralverse-CostAlert" \
    --alarm-description "Spending exceeds threshold" \
    --metric-name "EstimatedCharges" \
    --namespace "AWS/Billing" \
    --statistic Maximum \
    --period 86400 \
    --threshold 50 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 1 \
    --dimensions Name=Currency,Value=USD
```

---

## Dashboard JSON (Complete)

Import this into CloudWatch:

```json
{
  "widgets": [
    {
      "type": "metric",
      "x": 0, "y": 0, "width": 12, "height": 6,
      "properties": {
        "title": "Request Count",
        "metrics": [["AWS/ApiGateway", "Count", "ApiName", "spiralverse-api-prod", {"stat": "Sum"}]],
        "period": 300
      }
    },
    {
      "type": "metric",
      "x": 12, "y": 0, "width": 12, "height": 6,
      "properties": {
        "title": "Latency",
        "metrics": [
          ["AWS/ApiGateway", "Latency", "ApiName", "spiralverse-api-prod", {"stat": "Average"}],
          ["AWS/ApiGateway", "IntegrationLatency", "ApiName", "spiralverse-api-prod", {"stat": "Average"}]
        ],
        "period": 300
      }
    },
    {
      "type": "metric",
      "x": 0, "y": 6, "width": 8, "height": 6,
      "properties": {
        "title": "Error Rates",
        "metrics": [
          ["AWS/ApiGateway", "4XXError", "ApiName", "spiralverse-api-prod", {"stat": "Sum", "color": "#ff7f0e"}],
          ["AWS/ApiGateway", "5XXError", "ApiName", "spiralverse-api-prod", {"stat": "Sum", "color": "#d62728"}]
        ],
        "period": 300
      }
    },
    {
      "type": "metric",
      "x": 8, "y": 6, "width": 8, "height": 6,
      "properties": {
        "title": "Lambda Duration",
        "metrics": [
          ["AWS/Lambda", "Duration", "FunctionName", "spiralverse-protocol-prod", {"stat": "Average"}],
          ["AWS/Lambda", "Duration", "FunctionName", "spiralverse-protocol-prod", {"stat": "Maximum"}]
        ],
        "period": 300
      }
    },
    {
      "type": "metric",
      "x": 16, "y": 6, "width": 8, "height": 6,
      "properties": {
        "title": "Lambda Invocations & Errors",
        "metrics": [
          ["AWS/Lambda", "Invocations", "FunctionName", "spiralverse-protocol-prod", {"stat": "Sum"}],
          ["AWS/Lambda", "Errors", "FunctionName", "spiralverse-protocol-prod", {"stat": "Sum", "color": "#d62728"}]
        ],
        "period": 300
      }
    },
    {
      "type": "log",
      "x": 0, "y": 12, "width": 24, "height": 6,
      "properties": {
        "title": "SNAP Events (Geometric Violations)",
        "query": "SOURCE '/aws/lambda/spiralverse-protocol-prod' | fields @timestamp, @message | filter @message like /SNAP|snap/ | sort @timestamp desc | limit 50"
      }
    }
  ]
}
```

---

## Monitoring Best Practices

1. **Check dashboard daily** during initial deployment
2. **Set up SNS alerts** for critical issues
3. **Review SNAP events** to understand user behavior
4. **Monitor cold starts** if latency is critical
5. **Track costs weekly** to avoid surprises
6. **Archive old logs** after 30 days to reduce costs
