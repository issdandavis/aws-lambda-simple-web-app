"""
AWS Lambda Handler for SCBE-AETHERMOORE

Deploy this to AWS Lambda for cloud-based geometric verification.

Endpoints:
  POST /classify - Classify context geometrically
  POST /verify   - Verify with dual lattice
"""

import json
import numpy as np
from scbe_aethermoore.hypercube_brain import (
    hypercube_brain_classify,
    SignatureMode
)


def lambda_handler(event, context):
    """
    Main Lambda entry point.

    Request body:
    {
        "action": "classify" | "verify",
        "context": [timestamp, device_id, threat, entropy, load, velocity],
        "intent": "protect" | "seek" | "create" | ...
        "policy": {"tier": 0.8, "intent": 0.9, ...}
    }
    """
    try:
        # Parse request
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event)

        action = body.get('action', 'classify')
        context_list = body.get('context', [])
        intent = body.get('intent', 'unknown')

        # Convert to numpy
        context = np.array(context_list, dtype=float)

        # Ensure 6D context
        if len(context) < 6:
            context = np.pad(context, (0, 6 - len(context)), constant_values=0.5)

        # Classify geometrically
        state = hypercube_brain_classify(context)

        # Build response
        response = {
            "authorized": bool(state.signature_mode != SignatureMode.TRAPDOOR_FROZEN),
            "mode": state.signature_mode.value,
            "gamma": float(round(state.gamma, 4)),
            "risk": float(round(state.risk_factor, 4)),
            "inside_sphere": bool(state.is_inside),
            "expansion": float(round(state.expansion_factor, 4))
        }

        # Add trap warning
        if state.signature_mode == SignatureMode.TRAPDOOR_FROZEN:
            response["warning"] = "Time dilation trap activated"
            response["authorized"] = False

        # Add risk warning
        if state.risk_factor > 0.8:
            response["warning"] = "High risk context"
            response["authorized"] = False

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response)
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }


# Local testing
if __name__ == "__main__":
    # Test normal request
    test_event = {
        "body": json.dumps({
            "action": "classify",
            "context": [1704700000.0, 101.0, 3.0, 0.45, 12.0, 0.3],
            "intent": "protect"
        })
    }

    result = lambda_handler(test_event, None)
    print("Normal request:")
    print(json.dumps(json.loads(result['body']), indent=2))

    # Test attack (high velocity)
    attack_event = {
        "body": json.dumps({
            "action": "classify",
            "context": [1704700000.0, 101.0, 3.0, 0.45, 12.0, 0.95],
            "intent": "unknown"
        })
    }

    result = lambda_handler(attack_event, None)
    print("\nAttack request (high velocity):")
    print(json.dumps(json.loads(result['body']), indent=2))
