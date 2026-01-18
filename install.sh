#!/bin/bash
#
# SCBE-AETHERMOORE Quick Install Script
# =====================================
# Works on: Linux, macOS, Windows (Git Bash/WSL)
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/issdandavis/aws-lambda-simple-web-app/main/install.sh | bash
#
# Or:
#   ./install.sh
#

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           SCBE-AETHERMOORE Quick Installer                 ║"
echo "║     Hyperbolic Geometry + Sacred Tongues + PQC             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON=python3
    elif command -v python &> /dev/null; then
        PYTHON=python
    else
        echo -e "${RED}Error: Python 3.10+ is required but not found.${NC}"
        echo "Please install Python from https://python.org"
        exit 1
    fi

    # Check version
    PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
    PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

    if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
        echo -e "${RED}Error: Python 3.10+ required, found $PY_VERSION${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓${NC} Found Python $PY_VERSION"
}

# Check pip
check_pip() {
    if ! $PYTHON -m pip --version &> /dev/null; then
        echo -e "${YELLOW}Installing pip...${NC}"
        curl -sSL https://bootstrap.pypa.io/get-pip.py | $PYTHON
    fi
    echo -e "${GREEN}✓${NC} pip available"
}

# Create virtual environment
create_venv() {
    if [ -d "venv" ]; then
        echo -e "${YELLOW}Virtual environment already exists${NC}"
    else
        echo -e "${BLUE}Creating virtual environment...${NC}"
        $PYTHON -m venv venv
    fi

    # Activate
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    fi
    echo -e "${GREEN}✓${NC} Virtual environment ready"
}

# Install from GitHub
install_package() {
    echo -e "${BLUE}Installing SCBE-AETHERMOORE...${NC}"

    # If we're in the repo directory
    if [ -f "pyproject.toml" ]; then
        pip install -e ".[dev]" --quiet
    else
        # Clone and install
        if [ -d "scbe-aethermoore" ]; then
            cd scbe-aethermoore
            git pull origin main
        else
            git clone https://github.com/issdandavis/aws-lambda-simple-web-app.git scbe-aethermoore
            cd scbe-aethermoore
        fi
        pip install -e ".[dev]" --quiet
    fi

    echo -e "${GREEN}✓${NC} Package installed"
}

# Install API dependencies
install_api_deps() {
    echo -e "${BLUE}Installing API dependencies...${NC}"
    pip install fastapi uvicorn httpx --quiet
    echo -e "${GREEN}✓${NC} API dependencies ready"
}

# Test installation
test_install() {
    echo -e "${BLUE}Testing installation...${NC}"

    $PYTHON -c "
from scbe_production import __version__
from scbe_production.service import SCBEProductionService
from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import SacredTongueTokenizer

print(f'  SCBE Version: {__version__}')

service = SCBEProductionService()
health = service.health_check()
print(f'  Service Status: {health[\"status\"]}')

tokenizer = SacredTongueTokenizer('ko')
tokens = tokenizer.encode_to_string(b'test', separator=' ')
print(f'  Sacred Tongues: Working')
print(f'  Sample: {tokens}')
"
    echo -e "${GREEN}✓${NC} All tests passed"
}

# Print usage
print_usage() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                 Installation Complete!                      ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${GREEN}Quick Start:${NC}"
    echo ""
    echo "  # Activate environment"
    echo "  source venv/bin/activate"
    echo ""
    echo "  # Run the demo"
    echo "  python demo.py"
    echo ""
    echo "  # Start API server (for phone/tablet access)"
    echo "  python -m scbe_production.api"
    echo ""
    echo "  # Open web demo"
    echo "  open web/index.html"
    echo ""
    echo -e "${BLUE}API Endpoints (when server running):${NC}"
    echo "  http://localhost:8000/       - Web Demo"
    echo "  http://localhost:8000/docs   - API Documentation"
    echo "  http://localhost:8000/health - Health Check"
    echo ""
    echo -e "${YELLOW}For more info:${NC}"
    echo "  https://github.com/issdandavis/aws-lambda-simple-web-app"
    echo ""
}

# Main
main() {
    check_python
    check_pip
    create_venv
    install_package
    install_api_deps
    test_install
    print_usage
}

main "$@"
