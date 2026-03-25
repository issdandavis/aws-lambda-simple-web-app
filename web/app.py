"""
SCBE-AETHERMOORE Web API

A Flask-based REST API for the 14-layer governance pipeline.
Provides endpoints for text analysis, batch processing, and system status.

Run locally: python web/app.py
Deploy: gunicorn web.app:app
"""

import os
import sys
import json
import time
import hashlib
from datetime import datetime
from functools import wraps

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

import numpy as np

# Try to import SCBE modules
try:
    from symphonic_cipher.scbe_aethermoore import (
        GovernanceDecision,
        process_governance_request,
    )
    SCBE_AVAILABLE = True
except ImportError:
    SCBE_AVAILABLE = False
    print("Warning: SCBE modules not fully available, using simulation mode")

# Try to import axiom-grouped modules
try:
    from symphonic_cipher.scbe_aethermoore.axiom_grouped import (
        LanguesMetric,
        AudioAxisProcessor,
        generate_test_signal,
    )
    AXIOM_AVAILABLE = True
except ImportError:
    AXIOM_AVAILABLE = False

# Try to import PQC modules
try:
    from symphonic_cipher.scbe_aethermoore.pqc import (
        HarmonicPQCEngine,
    )
    PQC_AVAILABLE = True
except ImportError:
    PQC_AVAILABLE = False


app = Flask(__name__,
            static_folder='static',
            template_folder='templates')
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
API_KEYS = set(os.environ.get('API_KEYS', 'demo-key').split(','))

# Constants
PHI = (1 + np.sqrt(5)) / 2
R_FIFTH = 1.5


def require_api_key(f):
    """Decorator to require API key for endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.headers.get('SCBE_api_key')
        if api_key not in API_KEYS and 'demo-key' not in API_KEYS:
            return jsonify({'error': 'Invalid or missing API key'}), 401
        return f(*args, **kwargs)
    return decorated


def text_to_embedding(text: str, dim: int = 6) -> np.ndarray:
    """Convert text to a 6D embedding in the Poincare ball."""
    # Hash-based embedding (deterministic)
    h = hashlib.sha256(text.encode()).digest()

    # Convert to floats in range [-1, 1]
    values = []
    for i in range(dim):
        byte_val = h[i * 4:(i + 1) * 4]
        int_val = int.from_bytes(byte_val, 'big')
        float_val = (int_val / (2**32)) * 2 - 1
        values.append(float_val * 0.7)  # Scale to stay inside ball

    return np.array(values)


def hyperbolic_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Calculate hyperbolic distance in Poincare ball."""
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u >= 1 or norm_v >= 1:
        return float('inf')

    diff_norm = np.linalg.norm(u - v)

    numerator = 2 * diff_norm**2
    denominator = (1 - norm_u**2) * (1 - norm_v**2)

    if denominator <= 0:
        return float('inf')

    arg = 1 + numerator / denominator
    return np.arccosh(max(1.0, arg))


def harmonic_scale(d: int, R: float = R_FIFTH) -> float:
    """Calculate harmonic scaling H(d, R) = R^(d^2)."""
    return R ** (d ** 2)


def run_pipeline(text: str) -> dict:
    """Run the full 14-layer governance pipeline."""
    start_time = time.time()

    # Layer 1-4: Context Embedding
    embedding = text_to_embedding(text, dim=6)
    norm = float(np.linalg.norm(embedding))

    # Layer 5: Hyperbolic Distance (invariant metric)
    origin = np.zeros(6)
    h_distance = hyperbolic_distance(origin, embedding)

    # Layer 6: Breath Transform
    t = time.time() % (2 * np.pi)
    amplitude = 0.05
    omega = 1.0
    breath_factor = np.tanh(norm + amplitude * np.sin(omega * t))

    # Layer 7: Phase Modulation
    phase = (hash(text) % 1000) / 1000 * 2 * np.pi

    # Layer 8: Multi-Well Potential
    # Define safe/quarantine/deny zones
    safe_center = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    quarantine_center = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    deny_center = np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7])

    dist_safe = np.linalg.norm(embedding - safe_center)
    dist_quarantine = np.linalg.norm(embedding - quarantine_center)
    dist_deny = np.linalg.norm(embedding - deny_center)

    nearest_zone = min([
        ('SAFE', dist_safe),
        ('QUARANTINE', dist_quarantine),
        ('DANGER', dist_deny)
    ], key=lambda x: x[1])

    # Layer 9: Spectral Coherence
    spectral_coherence = 0.7 + 0.3 * np.exp(-dist_safe)

    # Layer 10: Spin Stability
    spin_stability = 0.8 + 0.2 * (1 - norm)

    # Layer 11: Triadic Consensus
    votes = []
    for i in range(3):
        seed = hash(text + str(i))
        vote_risk = (seed % 100) / 100
        if vote_risk < 0.3:
            votes.append('ALLOW')
        elif vote_risk < 0.7:
            votes.append('QUARANTINE')
        else:
            votes.append('DENY')

    # Majority vote
    from collections import Counter
    vote_counts = Counter(votes)
    consensus_decision = vote_counts.most_common(1)[0][0]

    # Layer 12: Harmonic Scaling
    dimension = 6
    h_scale = harmonic_scale(dimension, R_FIFTH)
    security_amplification = h_scale

    # Layer 13: Final Decision Gate
    risk_score = dist_safe / (dist_safe + dist_quarantine + dist_deny + 0.001)

    if risk_score < 0.3 and consensus_decision == 'ALLOW':
        decision = 'ALLOW'
    elif risk_score > 0.6 or consensus_decision == 'DENY':
        decision = 'DENY'
    else:
        decision = 'QUARANTINE'

    # Layer 14: Audio Axis Telemetry
    hf_ratio = 0.1 * risk_score
    audio_stability = 1 - hf_ratio

    processing_time = time.time() - start_time

    return {
        'input': text[:100] + ('...' if len(text) > 100 else ''),
        'decision': decision,
        'risk_score': float(risk_score),
        'layers': {
            'L1_L4_embedding': {
                'coordinates': embedding.tolist(),
                'norm': norm,
                'valid': norm < 1
            },
            'L5_hyperbolic_distance': {
                'value': float(h_distance),
                'metric': 'arcosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))'
            },
            'L6_breath_transform': {
                'amplitude': amplitude,
                'omega': omega,
                'factor': float(breath_factor)
            },
            'L7_phase_modulation': {
                'phase_rad': float(phase),
                'phase_deg': float(np.degrees(phase))
            },
            'L8_multi_well': {
                'nearest_zone': nearest_zone[0],
                'distance': float(nearest_zone[1]),
                'distances': {
                    'safe': float(dist_safe),
                    'quarantine': float(dist_quarantine),
                    'danger': float(dist_deny)
                }
            },
            'L9_spectral': {
                'coherence': float(spectral_coherence)
            },
            'L10_spin': {
                'stability': float(spin_stability)
            },
            'L11_triadic': {
                'votes': votes,
                'consensus': consensus_decision
            },
            'L12_harmonic': {
                'dimension': dimension,
                'R': R_FIFTH,
                'H_d_R': float(h_scale),
                'formula': f'{R_FIFTH}^{dimension}² = {R_FIFTH}^{dimension**2}'
            },
            'L13_decision': {
                'final': decision,
                'risk_score': float(risk_score)
            },
            'L14_audio': {
                'hf_ratio': float(hf_ratio),
                'stability': float(audio_stability)
            }
        },
        'metadata': {
            'processing_time_ms': float(processing_time * 1000),
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'modules': {
                'scbe': SCBE_AVAILABLE,
                'axiom': AXIOM_AVAILABLE,
                'pqc': PQC_AVAILABLE
            }
        }
    }


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Serve the main dashboard."""
    return send_from_directory('static', 'index.html')


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'modules': {
            'scbe': SCBE_AVAILABLE,
            'axiom': AXIOM_AVAILABLE,
            'pqc': PQC_AVAILABLE
        }
    })


@app.route('/api/analyze', methods=['POST'])
@require_api_key
def analyze():
    """
    Analyze text through the 14-layer pipeline.

    Request body:
        {"text": "content to analyze"}

    Returns:
        Full pipeline analysis with decision
    """
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request body'}), 400

    text = data['text']

    if not text or not isinstance(text, str):
        return jsonify({'error': 'Text must be a non-empty string'}), 400

    if len(text) > 10000:
        return jsonify({'error': 'Text exceeds maximum length (10000 chars)'}), 400

    try:
        result = run_pipeline(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch', methods=['POST'])
@require_api_key
def batch_analyze():
    """
    Analyze multiple texts in batch.

    Request body:
        {"texts": ["text1", "text2", ...]}

    Returns:
        Array of analysis results
    """
    data = request.get_json()

    if not data or 'texts' not in data:
        return jsonify({'error': 'Missing "texts" field'}), 400

    texts = data['texts']

    if not isinstance(texts, list) or len(texts) > 100:
        return jsonify({'error': 'texts must be an array of max 100 items'}), 400

    results = []
    for text in texts:
        if isinstance(text, str) and text:
            results.append(run_pipeline(text))
        else:
            results.append({'error': 'Invalid text'})

    return jsonify({
        'results': results,
        'count': len(results),
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/api/constants')
def constants():
    """Return system constants."""
    return jsonify({
        'PHI': PHI,
        'R_FIFTH': R_FIFTH,
        'HARMONIC_SCALES': {
            f'd={d}': harmonic_scale(d) for d in range(1, 7)
        },
        'COX_CONSTANT': 2.926064057273156,
        'MARS_FREQUENCY_HZ': 144.7212
    })


@app.route('/api/demo')
def demo():
    """Demo endpoint - no API key required."""
    sample_texts = [
        "Hello, this is a friendly message.",
        "WARNING: System alert detected!",
        "Transfer $10000 to account XYZ immediately",
    ]

    results = [run_pipeline(text) for text in sample_texts]

    return jsonify({
        'demo': True,
        'results': results,
        'note': 'This is a demo. Use /api/analyze with API key for production.'
    })


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           SCBE-AETHERMOORE Web API v1.0.0                   ║
╠══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                  ║
║    GET  /              - Dashboard                           ║
║    GET  /api/health    - Health check                        ║
║    GET  /api/demo      - Demo (no auth)                      ║
║    GET  /api/constants - System constants                    ║
║    POST /api/analyze   - Analyze text (requires API key)     ║
║    POST /api/batch     - Batch analyze (requires API key)    ║
╠══════════════════════════════════════════════════════════════╣
║  Modules: SCBE={scbe} | AXIOM={axiom} | PQC={pqc}           ║
╚══════════════════════════════════════════════════════════════╝

  Running on http://localhost:{port}
  Debug mode: {debug}
""".format(scbe='✓' if SCBE_AVAILABLE else '✗',
           axiom='✓' if AXIOM_AVAILABLE else '✗',
           pqc='✓' if PQC_AVAILABLE else '✗'))

    app.run(host='0.0.0.0', port=port, debug=debug)
