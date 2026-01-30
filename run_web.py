#!/usr/bin/env python3
"""
Quick start script for SCBE-AETHERMOORE Web API.

Usage:
    python run_web.py          # Run on port 5000
    python run_web.py 8080     # Run on port 8080
"""

import os
import sys

def main():
    # Get port from command line or default to 5000
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

    # Set environment variables
    os.environ['PORT'] = str(port)
    os.environ['DEBUG'] = 'true'
    os.environ['API_KEYS'] = 'demo-key'

    # Import and run app
    from web.app import app
    print(f"\n  Starting SCBE-AETHERMOORE Web API on http://localhost:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=True)


if __name__ == '__main__':
    main()
