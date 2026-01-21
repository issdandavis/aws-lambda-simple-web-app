# SCBE-AETHERMOORE Quick Start Guide

Get running in 5 minutes.

---

## Step 1: Install

```bash
git clone <this-repo>
cd aws-lambda-simple-web-app
pip install numpy
```

That's it. No other dependencies for the demo.

---

## Step 2: Run Your First Demo

```bash
python demo_quantum_resistance.py
```

You should see legitimate operations passing and attacks being blocked.

---

## Step 3: Understand the Core Concept

### Traditional Encryption
```
key = random_bytes(32)
ciphertext = encrypt(plaintext, key)
# Anyone with key can decrypt
```

### SCBE-AETHERMOORE
```python
context = [timestamp, device_id, threat, entropy, load, velocity]
key = derive_from_geometry(context)
ciphertext = encrypt(plaintext, key)
# Only correct context produces correct key
# Wrong context → wrong key → noise output
```

---

## Step 4: Try the Python API

```python
from scbe_aethermoore.hypercube_brain import hypercube_brain_classify
import numpy as np

# Create a context
context = np.array([
    1704700000.0,  # timestamp
    101.0,         # device_id
    3.0,           # threat level
    0.45,          # entropy
    12.0,          # load
    0.4            # velocity
])

# Classify it
state = hypercube_brain_classify(context)

# See the result
print(f"Signature mode: {state.signature_mode}")
print(f"Is inside sphere: {state.is_inside}")
print(f"Time dilation γ: {state.gamma}")
print(f"Risk factor: {state.risk_factor}")
```

---

## Step 5: Understand the Three Outcomes

```python
from scbe_aethermoore.hypercube_brain import SignatureMode

if state.signature_mode == SignatureMode.KYBER_INTERNAL:
    # Inside the brain sphere
    # Use Kyber (faster, internal operations)
    print("Internal operation - trusted")

elif state.signature_mode == SignatureMode.DILITHIUM_EXTERNAL:
    # Outside the brain sphere
    # Use Dilithium (stronger, external operations)
    print("External operation - verified")

elif state.signature_mode == SignatureMode.TRAPDOOR_FROZEN:
    # Velocity too high - attacker detected
    print("TRAPPED - time dilation activated")
```

---

## Step 6: Try Different Contexts

### Safe Context (Low Velocity)
```python
safe_context = np.array([1704700000.0, 101.0, 3.0, 0.45, 12.0, 0.3])
state = hypercube_brain_classify(safe_context)
# γ ≈ 1.05 - normal operation
```

### Dangerous Context (High Velocity)
```python
dangerous_context = np.array([1704700000.0, 101.0, 3.0, 0.45, 12.0, 0.95])
state = hypercube_brain_classify(dangerous_context)
# γ ≈ 3.2 - TRAPDOOR activated
```

### High Risk Context
```python
high_risk = np.array([1704700000.0, 101.0, 9.0, 0.45, 12.0, 0.4])
state = hypercube_brain_classify(high_risk)
# Hypercube expands - tighter policy boundaries
```

---

## Step 7: Full Dual Lattice Verification

```python
from demo_quantum_resistance import dual_lattice_verify
import numpy as np

context = np.array([1704700000.0, 101.0, 3.0, 0.45, 12.0, 0.4])
policy = {"tier": 0.8, "intent": 0.9, "data_class": 0.7, "safety": 0.95}
intent = "protect"
master_seed = b"your_secret_key_here_32_bytes!!"

success, results = dual_lattice_verify(context, policy, intent, master_seed)

print(f"Authorized: {success}")
for detail in results["details"]:
    print(f"  {detail}")
```

---

## Step 8: Run All Demos

```bash
# Quantum resistance (main demo)
python demo_quantum_resistance.py

# Geometry visualization
python -m scbe_aethermoore.hypercube_brain

# Time crystallization
python demo_temporal_lattice.py

# 17D dimensional folding
python demo_dimensional_fold.py

# Concentric trust rings
python demo_rings.py

# Full GeoSeal
python demo_geoseal.py
```

---

## Key Concepts Cheat Sheet

| Concept | What It Is | Why It Matters |
|---------|------------|----------------|
| **Context** | 6D vector describing the request | Binds crypto to situation |
| **Sphere** | Behavioral manifold | Represents "who you are" |
| **Hypercube** | Policy space | Represents "what's allowed" |
| **Intersection** | Where sphere meets hypercube | Determines your key |
| **γ (gamma)** | Time dilation factor | Traps fast attackers |
| **Kyber** | Post-quantum KEM | Key encapsulation |
| **Dilithium** | Post-quantum DSA | Signatures |
| **Crystallization** | Equations stabilizing | Time becomes security axis |

---

## Common Issues

### "ModuleNotFoundError: No module named 'numpy'"
```bash
pip install numpy
```

### "ModuleNotFoundError: No module named 'scbe_aethermoore'"
```bash
# Make sure you're in the right directory
cd aws-lambda-simple-web-app
python demo_quantum_resistance.py
```

### Tests Failing
```bash
# Run from repo root
python -m pytest tests/ -v
```

---

## Next Steps

1. Read the [UNIFIED_SPECIFICATION.md](scbe_aethermoore/UNIFIED_SPECIFICATION.md) for full math
2. Look at `examples/` for practical use cases
3. Explore `scbe_aethermoore/` source code
4. Try integrating into your own project

---

*Questions? See the main README or contact Isaac Davis.*
