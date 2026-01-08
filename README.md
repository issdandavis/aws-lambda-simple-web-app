# Physics Simulation Engine

A production-ready physics simulation engine built with AWS CDK, featuring real quantum mechanics, particle dynamics, and wave simulations.

## Features

### Physics Simulations

#### Quantum Mechanics
- **Photon Properties**: Calculate energy, frequency, wavelength, and momentum from wavelength
- **Hydrogen Atom**: Energy levels, spectral transitions (Lyman, Balmer, Paschen series)
- **Uncertainty Principle**: Heisenberg's ΔxΔp ≥ ħ/2
- **Quantum Tunneling**: Transmission coefficients through barriers
- **Harmonic Oscillator**: Quantum energy levels, zero-point energy
- **Particle in a Box**: Quantized energy levels
- **de Broglie Wavelength**: Matter wave calculations
- **Spin-Orbit Coupling**: Fine structure corrections
- **Wavefunctions**: Hydrogen radial wavefunctions with Laguerre polynomials

#### Particle Dynamics
- **Gravitational Force**: Newton's law of gravitation (N-body ready)
- **Electrostatic Force**: Coulomb's law
- **Lorentz Force**: Charged particle in E and B fields
- **Orbital Mechanics**: Kepler elements, periods, eccentricity
- **N-Body Simulation**: Multi-particle gravitational simulation
- **Collisions**: Elastic and inelastic with coefficient of restitution
- **Relativistic Mechanics**: Lorentz factor, time dilation, E=mc²
- **Black Holes**: Schwarzschild radius calculations
- **Terminal Velocity**: Drag force calculations

#### Wave Physics
- **Interference**: Two-source interference patterns, fringe spacing
- **Diffraction**: Single slit, diffraction gratings
- **Doppler Effect**: Sound and relativistic light
- **Standing Waves**: Modes, harmonics, nodes/antinodes
- **Blackbody Radiation**: Planck's law, Wien's law, Stefan-Boltzmann
- **Electromagnetic Waves**: Poynting vector, radiation pressure
- **Refraction**: Snell's law, total internal reflection, Fresnel equations

#### Physical Constants (CODATA 2018)
- All fundamental constants with exact SI values
- Uncertainty information for measured constants
- Derived quantities and mathematical constants

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway                               │
│                   (REST API with API Key)                       │
├─────────────────────────────────────────────────────────────────┤
│                     POST /simulate                               │
│                     GET /health                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Lambda Function                               │
│              (Physics Simulation Engine)                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Quantum    │  │   Particle   │  │    Wave      │          │
│  │   Module     │  │   Module     │  │   Module     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │   DynamoDB   │ │      S3      │ │  CloudWatch  │
    │  (Metadata)  │ │  (Results)   │ │ (Monitoring) │
    └──────────────┘ └──────────────┘ └──────────────┘
```

## Prerequisites

- Node.js 18+
- AWS CLI configured with credentials
- AWS CDK CLI (`npm install -g aws-cdk`)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd physics-simulation-engine

# Install dependencies
npm install

# Build
npm run build

# Run tests
npm test
```

## Deployment

### Quick Deploy

```bash
# Deploy to dev environment
./scripts/deploy.sh dev

# Deploy to staging
./scripts/deploy.sh staging

# Deploy to production
./scripts/deploy.sh prod
```

### Manual Deployment

```bash
# Bootstrap CDK (first time only)
npx cdk bootstrap

# Deploy
npx cdk deploy --context stage=dev

# View outputs
aws cloudformation describe-stacks --stack-name PhysicsSimulationStack-dev --query 'Stacks[0].Outputs'
```

## API Usage

### Authentication

All simulation endpoints require an API key in the `x-api-key` header.

```bash
# Get your API key value
aws apigateway get-api-key --api-key <ApiKeyId> --include-value
```

### Request Format

```json
{
  "simulationType": "quantum" | "particle" | "wave" | "constants",
  "operation": "<operation-name>",
  "parameters": { ... },
  "options": {
    "saveToS3": false,
    "includeMetadata": true,
    "precision": "standard" | "high"
  }
}
```

### Examples

#### Photon Properties
```bash
curl -X POST https://your-api.amazonaws.com/dev/simulate \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "simulationType": "quantum",
    "operation": "photon_properties",
    "parameters": {
      "wavelengthNm": 550
    }
  }'
```

Response:
```json
{
  "success": true,
  "simulationType": "quantum",
  "operation": "photon_properties",
  "result": {
    "energy": 3.61e-19,
    "frequency": 5.45e14,
    "wavelength": 5.5e-7,
    "momentum": 1.20e-27,
    "angularMomentum": 1.05e-34
  },
  "metadata": {
    "simulationId": "abc123",
    "timestamp": "2024-01-01T00:00:00.000Z",
    "executionTimeMs": 5,
    "constantsUsed": ["h", "c", "hbar"]
  }
}
```

#### Hydrogen Transition (Balmer Series)
```bash
curl -X POST https://your-api.amazonaws.com/dev/simulate \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "simulationType": "quantum",
    "operation": "hydrogen_transition",
    "parameters": {
      "nInitial": 3,
      "nFinal": 2
    }
  }'
```

#### Relativistic Particle
```bash
curl -X POST https://your-api.amazonaws.com/dev/simulate \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "simulationType": "particle",
    "operation": "relativistic",
    "parameters": {
      "restMass": 9.1093837015e-31,
      "velocity": 149896229
    }
  }'
```

#### Blackbody Radiation
```bash
curl -X POST https://your-api.amazonaws.com/dev/simulate \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "simulationType": "wave",
    "operation": "blackbody",
    "parameters": {
      "temperature": 5778
    }
  }'
```

#### N-Body Simulation
```bash
curl -X POST https://your-api.amazonaws.com/dev/simulate \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "simulationType": "particle",
    "operation": "n_body_simulation",
    "parameters": {
      "particles": [
        {"mass": 1.989e30, "position": {"x": 0, "y": 0, "z": 0}, "velocity": {"x": 0, "y": 0, "z": 0}},
        {"mass": 5.972e24, "position": {"x": 1.496e11, "y": 0, "z": 0}, "velocity": {"x": 0, "y": 29780, "z": 0}}
      ],
      "dt": 86400,
      "steps": 365
    }
  }'
```

## Available Operations

### Quantum Operations
| Operation | Parameters | Description |
|-----------|------------|-------------|
| `photon_properties` | `wavelengthNm` | Calculate photon energy, frequency, momentum |
| `hydrogen_energy` | `n` | Hydrogen atom energy level |
| `hydrogen_transition` | `nInitial, nFinal` | Spectral transition wavelength |
| `uncertainty` | `deltaX` | Heisenberg uncertainty |
| `tunneling` | `particleMass, particleEnergy, barrierHeight, barrierWidth` | Quantum tunneling |
| `harmonic_oscillator` | `mass, angularFrequency, n` | QHO energy levels |
| `particle_in_box` | `mass, boxLength, n` | Infinite well |
| `de_broglie` | `mass, velocity` | Matter wavelength |
| `spin_orbit` | `n, l, j` | Fine structure |
| `hydrogen_wavefunction` | `n, l, rValues` | Radial wavefunction |

### Particle Operations
| Operation | Parameters | Description |
|-----------|------------|-------------|
| `gravitational_force` | `m1, m2, r` | Newton's gravity |
| `electrostatic_force` | `q1, q2, r` | Coulomb force |
| `lorentz_force` | `charge, velocity, electricField, magneticField` | EM force |
| `orbital_elements` | `centralMass, orbiterMass, position, velocity` | Kepler elements |
| `n_body_simulation` | `particles, dt, steps` | Multi-body gravity |
| `elastic_collision` | `m1, v1, m2, v2` | Elastic collision |
| `inelastic_collision` | `m1, v1, m2, v2, restitution` | Inelastic collision |
| `relativistic` | `restMass, velocity` | Special relativity |
| `escape_velocity` | `centralMass, radius` | Escape velocity |
| `schwarzschild_radius` | `mass` | Event horizon |
| `pendulum` | `length, gravity?` | Simple pendulum |
| `terminal_velocity` | `mass, gravity, fluidDensity, crossSectionalArea, dragCoefficient` | Drag |

### Wave Operations
| Operation | Parameters | Description |
|-----------|------------|-------------|
| `wave_parameters` | `amplitude, frequency, mediumVelocity?, phase?` | Wave properties |
| `sinusoidal_wave` | `params, xMin, xMax, time, numPoints?` | Wave displacement |
| `interference` | `wavelength, sourceSpacing, screenDistance, screenWidth` | Double slit |
| `single_slit_diffraction` | `wavelength, slitWidth, screenDistance` | Diffraction |
| `diffraction_grating` | `wavelength, gratingSpacing, numSlits` | Grating |
| `doppler_sound` | `sourceFrequency, soundSpeed, sourceVelocity, observerVelocity` | Sound Doppler |
| `doppler_relativistic` | `sourceFrequency, relativeVelocity` | Light Doppler |
| `standing_waves` | `stringLength, tension, linearDensity` | Standing waves |
| `blackbody` | `temperature` | Planck radiation |
| `em_wave_properties` | `electricFieldAmplitude, frequency` | EM wave |
| `refraction` | `incidentAngle, n1, n2` | Snell's law |

### Constants Operations
| Operation | Parameters | Description |
|-----------|------------|-------------|
| `get_constant` | `name` | Get single constant |
| `get_all_constants` | - | Get all constants |
| `get_constant_with_uncertainty` | `name` | Get with uncertainty |

## Monitoring

The stack includes:
- CloudWatch Dashboard with Lambda metrics
- Alarms for errors, duration, and throttling
- API Gateway metrics and logging
- X-Ray tracing enabled

Access the dashboard:
```bash
aws cloudwatch get-dashboard --dashboard-name physics-simulation-dev
```

## Testing

```bash
# Run all tests
npm test

# Run with coverage
npm test -- --coverage

# Run specific test file
npm test -- tests/unit/physics/quantum.test.ts

# Test the deployed API
./scripts/test-api.sh https://your-api.amazonaws.com/dev your-api-key
```

## Cleanup

```bash
# Destroy dev stack
./scripts/destroy.sh dev

# Destroy production (with confirmation)
./scripts/destroy.sh prod
```

## Project Structure

```
├── bin/
│   └── app.ts                 # CDK app entry point
├── lib/
│   └── physics-simulation-stack.ts  # CDK infrastructure
├── src/
│   ├── physics/
│   │   ├── constants.ts       # CODATA 2018 constants
│   │   ├── quantum.ts         # Quantum mechanics
│   │   ├── particles.ts       # Particle dynamics
│   │   ├── waves.ts           # Wave simulations
│   │   └── index.ts           # Physics exports
│   └── lambda/
│       ├── handler.ts         # Lambda handler
│       ├── simulation-engine.ts # Simulation executor
│       ├── validator.ts       # Input validation
│       ├── types.ts           # TypeScript types
│       └── index.ts           # Lambda exports
├── tests/
│   ├── setup.ts               # Jest setup
│   └── unit/
│       ├── physics/           # Physics module tests
│       └── lambda/            # Lambda tests
├── scripts/
│   ├── deploy.sh              # Deployment script
│   ├── test-api.sh            # API testing script
│   └── destroy.sh             # Cleanup script
├── package.json
├── tsconfig.json
├── cdk.json
└── jest.config.js
```

## License

MIT
