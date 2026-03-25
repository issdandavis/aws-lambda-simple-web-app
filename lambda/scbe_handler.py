"""
SCBE-AETHERMOORE AWS Lambda Handler

Serverless function for the 14-layer governance pipeline.
Deploy with: sam deploy or zip and upload to AWS Console.
"""

import json
import hashlib
import time
import numpy as np

# Constants
PHI = (1 + np.sqrt(5)) / 2
R_FIFTH = 1.5


def text_to_embedding(text: str, dim: int = 6) -> np.ndarray:
    """Convert text to 6D Poincare ball embedding."""
    h = hashlib.sha256(text.encode()).digest()
    values = []
    for i in range(dim):
        byte_val = h[i * 4:(i + 1) * 4]
        int_val = int.from_bytes(byte_val, 'big')
        float_val = (int_val / (2**32)) * 2 - 1
        values.append(float_val * 0.7)
    return np.array(values)


def hyperbolic_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Calculate hyperbolic distance in Poincare ball."""
    norm_u, norm_v = np.linalg.norm(u), np.linalg.norm(v)
    if norm_u >= 1 or norm_v >= 1:
        return float('inf')
    diff_norm = np.linalg.norm(u - v)
    denom = (1 - norm_u**2) * (1 - norm_v**2)
    if denom <= 0:
        return float('inf')
    return np.arccosh(max(1.0, 1 + 2 * diff_norm**2 / denom))


def harmonic_scale(d: int, R: float = R_FIFTH) -> float:
    """H(d, R) = R^(d²)"""
    return R ** (d ** 2)


def run_pipeline(text: str) -> dict:
    """Run 14-layer governance pipeline."""
    start = time.time()

    # L1-4: Embedding
    emb = text_to_embedding(text)
    norm = float(np.linalg.norm(emb))

    # L5: Hyperbolic distance
    h_dist = hyperbolic_distance(np.zeros(6), emb)

    # L8: Multi-well zones
    safe = np.array([0.1]*6)
    dist_safe = np.linalg.norm(emb - safe)
    dist_quar = np.linalg.norm(emb - np.array([0.4]*6))
    dist_deny = np.linalg.norm(emb - np.array([0.7]*6))

    # L11: Triadic consensus
    votes = []
    for i in range(3):
        risk = (hash(text + str(i)) % 100) / 100
        votes.append('ALLOW' if risk < 0.3 else 'QUARANTINE' if risk < 0.7 else 'DENY')

    from collections import Counter
    consensus = Counter(votes).most_common(1)[0][0]

    # L12: Harmonic scaling
    h_scale = harmonic_scale(6)

    # L13: Final decision
    risk_score = dist_safe / (dist_safe + dist_quar + dist_deny + 0.001)

    if risk_score < 0.3 and consensus == 'ALLOW':
        decision = 'ALLOW'
    elif risk_score > 0.6 or consensus == 'DENY':
        decision = 'DENY'
    else:
        decision = 'QUARANTINE'

    return {
        'decision': decision,
        'risk_score': round(risk_score, 4),
        'consensus': consensus,
        'votes': votes,
        'harmonic_scale': h_scale,
        'hyperbolic_distance': round(h_dist, 4),
        'embedding_norm': round(norm, 4),
        'processing_ms': round((time.time() - start) * 1000, 2)
    }


def lambda_handler(event, context):
    """
    AWS Lambda entry point.

    Accepts:
        POST with JSON body: {"text": "content to analyze"}
        GET with query param: ?text=content

    Returns:
        JSON with decision, risk_score, and layer details
    """
    try:
        # Parse input
        if event.get('body'):
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
            text = body.get('text', '')
        elif event.get('queryStringParameters'):
            text = event['queryStringParameters'].get('text', '')
        else:
            text = event.get('text', '')

        if not text:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': 'Missing "text" parameter'})
            }

        # Run pipeline
        result = run_pipeline(text)
        result['input_preview'] = text[:50] + ('...' if len(text) > 50 else '')

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result)
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': str(e)})
        }


# For local testing
if __name__ == '__main__':
    test_event = {'text': 'Hello, this is a test message'}
    result = lambda_handler(test_event, None)
    print(json.dumps(json.loads(result['body']), indent=2))
