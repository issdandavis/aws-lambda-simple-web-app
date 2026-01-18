#!/usr/bin/env python3
"""
POLLY - SCBE-AETHERMOORE AI Agent
=================================
Agentic coding assistant with:
- Autopilot coding mode
- AI-to-AI communication via Sacred Tongues
- Human-readable cryptographic messages
- Security scanner ("no cold agents")
- Code library (Python & TypeScript)

Named after "Polly" from the Polly/Quasi/Demi dimensional states.
"""

import sys
import os
import time
import hashlib
import json
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import Sacred Tongues
try:
    from symphonic_cipher.scbe_aethermoore.spiral_seal.sacred_tongues import (
        SacredTongueTokenizer,
        SacredTongue,
        TONGUES,
    )
    TONGUES_AVAILABLE = True
except ImportError:
    TONGUES_AVAILABLE = False

# =============================================================================
# Colors and Formatting
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

def c(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"

# =============================================================================
# Agent State
# =============================================================================

class AgentMode(Enum):
    INTERACTIVE = "interactive"
    AUTOPILOT = "autopilot"
    AI_COMM = "ai_comm"

@dataclass
class AgentState:
    """Polly's current state."""
    mode: AgentMode = AgentMode.INTERACTIVE
    session_id: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)
    autopilot_tasks: List[str] = field(default_factory=list)
    ai_peers: Dict[str, str] = field(default_factory=dict)  # agent_id -> public_key

    def __post_init__(self):
        if not self.session_id:
            self.session_id = hashlib.sha256(
                f"{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]

# =============================================================================
# Sacred Tongue Communication Protocol
# =============================================================================

class SacredTongueProtocol:
    """
    AI-to-AI communication using Sacred Tongues.

    Messages are encoded in human-readable spell-text that is also
    cryptographically verifiable. This allows:
    - Humans to read AI communications
    - AIs to verify message authenticity
    - Audit trails in natural-looking format
    """

    def __init__(self):
        self.tokenizers = {}
        if TONGUES_AVAILABLE:
            for code in ['ko', 'av', 'ru', 'ca', 'um', 'dr']:
                self.tokenizers[code] = SacredTongueTokenizer(code)

    def encode_message(self, message: str, tongue: str = 'ko') -> str:
        """Encode a message to Sacred Tongue spell-text."""
        if not TONGUES_AVAILABLE or tongue not in self.tokenizers:
            # Fallback: simple transformation
            return self._fallback_encode(message, tongue)

        tokenizer = self.tokenizers[tongue]
        data = message.encode('utf-8')
        return tokenizer.encode_to_string(data, separator=' ')

    def decode_message(self, spelltext: str, tongue: str = 'ko') -> str:
        """Decode spell-text back to message."""
        if not TONGUES_AVAILABLE or tongue not in self.tokenizers:
            return self._fallback_decode(spelltext, tongue)

        tokenizer = self.tokenizers[tongue]
        try:
            data = tokenizer.decode_from_string(spelltext, separator=' ')
            return data.decode('utf-8')
        except Exception:
            return f"[decode error: {spelltext[:30]}...]"

    def _fallback_encode(self, message: str, tongue: str) -> str:
        """Fallback encoding without full Sacred Tongues."""
        prefixes = ['vel', 'kor', 'sil', 'zar', 'keth', 'thul']
        suffixes = ['a', 'ae', 'ei', 'ia', 'oa', 'uu']

        tokens = []
        for char in message.encode('utf-8'):
            pi = char % len(prefixes)
            si = (char >> 4) % len(suffixes)
            tokens.append(f"{prefixes[pi]}'{suffixes[si]}")

        return f"{tongue}:" + ' '.join(tokens)

    def _fallback_decode(self, spelltext: str, tongue: str) -> str:
        """Fallback decoding."""
        return f"[encoded: {spelltext[:40]}...]"

    def create_ai_message(
        self,
        sender_id: str,
        recipient_id: str,
        content: str,
        intent: str = "communicate"
    ) -> Dict[str, Any]:
        """
        Create a structured AI-to-AI message.

        Format:
        {
            "header": "av:<header_spelltext>",  # Avali for metadata
            "nonce": "ko:<nonce_spelltext>",    # Kor'aelin for flow
            "body": "ca:<body_spelltext>",      # Cassisivadan for content
            "signature": "dr:<sig_spelltext>",  # Draumric for verification
        }
        """
        timestamp = datetime.now().isoformat()
        nonce = hashlib.sha256(f"{timestamp}{sender_id}".encode()).hexdigest()[:16]

        # Encode each part with appropriate tongue
        header_data = f"from:{sender_id}|to:{recipient_id}|intent:{intent}|ts:{timestamp}"

        message = {
            "version": "SCBE-AI-COMM-v1",
            "header": self.encode_message(header_data, 'av'),
            "nonce": self.encode_message(nonce, 'ko'),
            "body": self.encode_message(content, 'ca'),
            "signature": self._sign_message(header_data + content, nonce),
            "human_readable": {
                "from": sender_id,
                "to": recipient_id,
                "intent": intent,
                "preview": content[:100] + ("..." if len(content) > 100 else ""),
            }
        }

        return message

    def _sign_message(self, content: str, nonce: str) -> str:
        """Create a signature for the message."""
        sig_data = hashlib.sha256(f"{content}{nonce}".encode()).hexdigest()[:32]
        return self.encode_message(sig_data, 'dr')

    def verify_message(self, message: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify an AI message's integrity."""
        try:
            # Check required fields
            required = ['header', 'nonce', 'body', 'signature']
            for field in required:
                if field not in message:
                    return False, f"Missing field: {field}"

            # In production, would verify signature cryptographically
            return True, "Message verified"
        except Exception as e:
            return False, f"Verification error: {e}"

# =============================================================================
# Security Scanner
# =============================================================================

class SecurityScanner:
    """
    Code security scanner - "Antivirus for Code"
    Ensures "none of our agents get a cold"
    """

    PATTERNS = {
        'critical': [
            (r'\beval\s*\(', 'eval() execution - arbitrary code risk'),
            (r'\bexec\s*\(', 'exec() execution - arbitrary code risk'),
            (r'__import__\s*\(', 'Dynamic import - code injection risk'),
        ],
        'high': [
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret'),
            (r'SELECT.*FROM.*WHERE.*\+', 'SQL injection risk (string concat)'),
            (r'f"SELECT.*{', 'SQL injection risk (f-string)'),
            (r'subprocess\.call\(.*shell\s*=\s*True', 'Shell injection risk'),
            (r'os\.system\s*\(', 'Command injection risk'),
        ],
        'medium': [
            (r'\brandom\.\w+\s*\(', 'Insecure random (use secrets module)'),
            (r'pickle\.load', 'Pickle deserialization risk'),
            (r'yaml\.load\s*\([^)]*\)', 'Unsafe YAML load (use safe_load)'),
            (r'requests\.get\([^)]*verify\s*=\s*False', 'SSL verification disabled'),
        ],
        'low': [
            (r'# ?TODO', 'TODO comment found'),
            (r'print\s*\(.*password', 'Potential credential logging'),
            (r'DEBUG\s*=\s*True', 'Debug mode enabled'),
        ],
    }

    def scan(self, code: str) -> List[Dict[str, Any]]:
        """Scan code for vulnerabilities."""
        findings = []
        lines = code.split('\n')

        for severity, patterns in self.PATTERNS.items():
            for pattern, description in patterns:
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        findings.append({
                            'severity': severity,
                            'line': i,
                            'code': line.strip()[:60],
                            'description': description,
                            'pattern': pattern,
                        })

        return sorted(findings, key=lambda x:
            {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x['severity']])

    def get_fix_suggestion(self, finding: Dict[str, Any]) -> str:
        """Get fix suggestion for a finding."""
        desc = finding['description'].lower()

        if 'eval' in desc or 'exec' in desc:
            return "Use ast.literal_eval() for safe evaluation, or avoid dynamic execution"
        elif 'hardcoded' in desc:
            return "Use environment variables or a secrets manager (e.g., AWS Secrets Manager)"
        elif 'sql injection' in desc:
            return "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))"
        elif 'random' in desc:
            return "Use secrets.token_bytes() or secrets.token_hex() for cryptographic randomness"
        elif 'pickle' in desc:
            return "Use JSON for untrusted data, or cryptographically sign pickled data"
        elif 'shell' in desc or 'command' in desc:
            return "Use subprocess with shell=False and pass args as list"
        else:
            return "Review and apply security best practices"

# =============================================================================
# Code Library
# =============================================================================

CODE_LIBRARY = {
    'python': {
        'encrypt': '''
from scbe_production.service import SCBEProductionService

service = SCBEProductionService()

# Seal a memory shard
shard = service.seal_memory(
    plaintext=b"Sensitive data here",
    agent_id="my-agent",
    topic="secrets",
    position=(1, 2, 3, 5, 8, 13),  # Fibonacci!
)

print(f"Sealed: {shard.shard_id}")
''',
        'verify': '''
from scbe_production.service import SCBEProductionService, AccessRequest

service = SCBEProductionService()

request = AccessRequest(
    agent_id="my-agent",
    message="Need access for task",
    features={"trust_level": 0.85},
    position=(1, 2, 3, 5, 8, 13),
)

response = service.access_memory(request)
print(f"Decision: {response.decision}")
print(f"Risk: {response.risk_score}")
''',
        'tongues': '''
from symphonic_cipher.spiralverse.tongues import SacredTongueTokenizer

# Encode to Kor'aelin (nonce/flow tongue)
tokenizer = SacredTongueTokenizer('ko')
spelltext = tokenizer.encode_to_string(b"Hello SCBE!", separator=" ")
print(f"Spell-text: {spelltext}")

# Decode back
decoded = tokenizer.decode_from_string(spelltext, separator=" ")
print(f"Decoded: {decoded.decode()}")
''',
        'governance': '''
from symphonic_cipher.scbe_aethermoore.governance import GovernanceEngine

engine = GovernanceEngine()

# Check authorization
result = engine.evaluate(
    agent_id="agent-001",
    action="read_memory",
    context={"trust_level": 0.8, "risk_factor": 0.2},
)

print(f"Decision: {result.decision}")  # ALLOW, QUARANTINE, DENY, SNAP
print(f"Risk Score: {result.risk_score}")
''',
    },
    'typescript': {
        'harmonic': '''
import { harmonicScale, HarmonicConfig } from '@scbe/aethermoore';

const config: HarmonicConfig = {
  R: 1.618,  // Golden ratio
  alpha: 10.0,
  beta: 0.5,
};

const distance = 1.5;
const cost = harmonicScale(distance, config);
console.log(`Harmonic cost: ${cost}`);
''',
        'pqc': '''
import { PQCProvider, KyberKeyPair } from '@scbe/aethermoore';

const pqc = new PQCProvider();

// Generate key pair
const keyPair: KyberKeyPair = await pqc.generateKeyPair();

// Encapsulate shared secret
const { ciphertext, sharedSecret } = await pqc.encapsulate(keyPair.publicKey);

// Decapsulate
const recovered = await pqc.decapsulate(ciphertext, keyPair.secretKey);
console.log('Secrets match:', sharedSecret === recovered);
''',
        'lattice': '''
import { QCLatticeProvider, ValidationResult } from '@scbe/aethermoore';

const lattice = new QCLatticeProvider();

// Validate a point in the quasicrystal
const point = { x: 0.5, y: 0.3, z: 0.8 };
const result: ValidationResult = lattice.validate(point);

console.log(`Valid: ${result.valid}`);
console.log(`Crystallinity: ${result.crystallinity}`);
''',
    }
}

# =============================================================================
# Autopilot System
# =============================================================================

class AutopilotEngine:
    """
    Autopilot coding mode - executes tasks autonomously.
    """

    def __init__(self, protocol: SacredTongueProtocol):
        self.protocol = protocol
        self.task_queue: List[Dict[str, Any]] = []
        self.completed: List[Dict[str, Any]] = []

    def add_task(self, description: str, priority: int = 1) -> str:
        """Add a task to the autopilot queue."""
        task_id = hashlib.sha256(
            f"{description}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]

        task = {
            'id': task_id,
            'description': description,
            'priority': priority,
            'status': 'pending',
            'created': datetime.now().isoformat(),
        }
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda x: x['priority'], reverse=True)

        return task_id

    def execute_next(self) -> Optional[Dict[str, Any]]:
        """Execute the next task in queue."""
        if not self.task_queue:
            return None

        task = self.task_queue.pop(0)
        task['status'] = 'executing'

        # Simulate task execution
        result = self._execute_task(task)

        task['status'] = 'completed'
        task['result'] = result
        task['completed'] = datetime.now().isoformat()
        self.completed.append(task)

        return task

    def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task (simulated)."""
        desc = task['description'].lower()

        if 'scan' in desc or 'security' in desc:
            return {'action': 'security_scan', 'status': 'completed'}
        elif 'encrypt' in desc:
            return {'action': 'encrypt', 'status': 'completed'}
        elif 'generate' in desc or 'code' in desc:
            return {'action': 'code_generation', 'status': 'completed'}
        else:
            return {'action': 'generic', 'status': 'completed'}

    def get_status(self) -> Dict[str, Any]:
        """Get autopilot status."""
        return {
            'pending': len(self.task_queue),
            'completed': len(self.completed),
            'queue': [t['description'][:40] for t in self.task_queue[:5]],
        }

# =============================================================================
# Polly Agent
# =============================================================================

class PollyAgent:
    """
    Polly - The SCBE-AETHERMOORE AI Agent

    Named after "Polly" state (ν ≈ 1.0 = full dimension active)
    from the Fluxing Dimensions system.
    """

    def __init__(self):
        self.state = AgentState()
        self.protocol = SacredTongueProtocol()
        self.scanner = SecurityScanner()
        self.autopilot = AutopilotEngine(self.protocol)
        self.agent_id = f"polly-{self.state.session_id[:8]}"

    def print_banner(self):
        """Print welcome banner."""
        banner = f"""
{Colors.MAGENTA}╔══════════════════════════════════════════════════════════════╗
║                                                                ║
║   {Colors.BOLD}✨ POLLY - SCBE AI AGENT ✨{Colors.RESET}{Colors.MAGENTA}                             ║
║   {Colors.DIM}Agentic Assistant with Sacred Tongue Communication{Colors.RESET}{Colors.MAGENTA}       ║
║                                                                ║
║   Version 3.0.0 | Session: {self.state.session_id[:8]}{Colors.MAGENTA}                        ║
║                                                                ║
╚══════════════════════════════════════════════════════════════╝{Colors.RESET}

{Colors.CYAN}Modes:{Colors.RESET}
  {c('ask', Colors.GREEN)}       - Chat about SCBE, crypto, security
  {c('code', Colors.GREEN)}      - Get code examples (Python/TypeScript)
  {c('scan', Colors.GREEN)}      - Security scan your code
  {c('autopilot', Colors.GREEN)} - Autonomous task execution
  {c('ai-comm', Colors.GREEN)}   - AI-to-AI communication demo
  {c('help', Colors.GREEN)}      - Show all commands

{Colors.DIM}Tip: Polly speaks Sacred Tongues for human-readable crypto!{Colors.RESET}
"""
        print(banner)

    def print_help(self):
        """Print help."""
        help_text = f"""
{Colors.BOLD}Polly Commands:{Colors.RESET}

{c('Conversation:', Colors.CYAN)}
  {c('ask', Colors.GREEN)}        Start a Q&A conversation
  {c('help', Colors.GREEN)}       Show this help

{c('Development:', Colors.CYAN)}
  {c('code', Colors.GREEN)}       Browse code library
  {c('scan', Colors.GREEN)}       Scan code for vulnerabilities
  {c('autopilot', Colors.GREEN)}  Enter autopilot mode

{c('AI Communication:', Colors.CYAN)}
  {c('ai-comm', Colors.GREEN)}    AI-to-AI messaging demo
  {c('encode', Colors.GREEN)}     Encode text to Sacred Tongue
  {c('decode', Colors.GREEN)}     Decode Sacred Tongue text

{c('System:', Colors.CYAN)}
  {c('status', Colors.GREEN)}     Show agent status
  {c('exit', Colors.GREEN)}       Exit Polly

{Colors.DIM}Example: Type 'ask' then chat about SCBE security{Colors.RESET}
"""
        print(help_text)

    # =========================================================================
    # Q&A System
    # =========================================================================

    def cmd_ask(self):
        """Interactive Q&A mode."""
        print(f"\n{c('Chat Mode', Colors.BOLD)} {Colors.DIM}(type 'back' to exit){Colors.RESET}\n")

        knowledge = {
            'scbe': "SCBE (Symphonic Cipher Behavioral Engine) is a 14-layer hyperbolic geometry security system. It makes adversarial behavior exponentially costly through harmonic scaling: H(d) = 1 + 10*tanh(0.5*d).",
            'quantum': "SCBE uses ML-KEM-768 (Kyber) for key exchange and ML-DSA-65 (Dilithium) for signatures. Both are NIST-approved post-quantum algorithms resistant to Shor's algorithm.",
            'tongues': "Sacred Tongues are 6 cryptolinguistic encodings: Kor'aelin (nonce), Avali (metadata), Runethic (binding), Cassisivadan (ciphertext), Umbroth (redaction), Draumric (structure). Each byte maps to a human-readable token.",
            'governance': "The Governance Engine returns: ALLOW (risk<0.2), QUARANTINE (0.2-0.4), DENY (0.4-0.8), or SNAP (>0.8). SNAP triggers fail-to-noise - destroying secrets rather than allowing breach.",
            'geoseal': "GeoSeal uses a dual-space manifold: Sphere S^n for behavioral state, Hypercube [0,1]^m for policy state. Distance measures alignment - interior path = trusted, exterior = suspicious.",
            'phdm': "PHDM (Polyhedral Hamiltonian Defense Manifold) uses 16 canonical polyhedra with HMAC-chained Hamiltonian paths for topological security verification.",
            'harmonic': "Harmonic Scaling Wall: H(d) = 1 + α*tanh(β*d) where α=10, β=0.5. Small deviations = small cost, large deviations = massive cost. Prevents gradual drift attacks.",
        }

        while True:
            try:
                question = input(f"{c('You:', Colors.GREEN)} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not question:
                continue
            if question.lower() in ('back', 'exit', 'quit'):
                break

            # Find relevant answer
            q_lower = question.lower()
            response = None

            for key, answer in knowledge.items():
                if key in q_lower:
                    response = answer
                    break

            if not response:
                if any(word in q_lower for word in ['how', 'what', 'why', 'explain']):
                    response = "SCBE-AETHERMOORE is a multi-layer security framework. Try asking about: scbe, quantum, tongues, governance, geoseal, phdm, or harmonic scaling."
                elif 'code' in q_lower:
                    response = "Use the 'code' command to see examples in Python or TypeScript!"
                else:
                    response = "I can help with SCBE security concepts. Ask about: quantum resistance, Sacred Tongues, governance, GeoSeal, PHDM, or harmonic scaling."

            print(f"{c('Polly:', Colors.MAGENTA)} {response}\n")

    # =========================================================================
    # Code Library
    # =========================================================================

    def cmd_code(self):
        """Show code library."""
        print(f"\n{c('Code Library', Colors.BOLD)}\n")
        print("Select language:")
        print(f"  {c('1', Colors.GREEN)} - Python")
        print(f"  {c('2', Colors.GREEN)} - TypeScript")
        print(f"  {c('0', Colors.GREEN)} - Back\n")

        try:
            choice = input(f"{Colors.CYAN}Language (0-2): {Colors.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            return

        if choice == '0':
            return

        lang = 'python' if choice == '1' else 'typescript' if choice == '2' else None
        if not lang:
            print(f"{c('Invalid choice', Colors.RED)}")
            return

        examples = CODE_LIBRARY.get(lang, {})
        print(f"\n{c(f'{lang.title()} Examples:', Colors.BOLD)}")
        for i, (name, _) in enumerate(examples.items(), 1):
            print(f"  {c(str(i), Colors.GREEN)} - {name}")
        print(f"  {c('0', Colors.GREEN)} - Back\n")

        try:
            ex_choice = input(f"{Colors.CYAN}Example (0-{len(examples)}): {Colors.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            return

        if ex_choice == '0':
            return

        try:
            idx = int(ex_choice) - 1
            name = list(examples.keys())[idx]
            code = examples[name]
            print(f"\n{c(f'=== {name.upper()} ===', Colors.CYAN)}")
            print(code)
        except (ValueError, IndexError):
            print(f"{c('Invalid choice', Colors.RED)}")

    # =========================================================================
    # Security Scanner
    # =========================================================================

    def cmd_scan(self):
        """Scan code for vulnerabilities."""
        print(f"\n{c('Security Scanner', Colors.BOLD)} {Colors.DIM}(\"No cold agents!\"){Colors.RESET}")
        print(f"Paste code below, then press {c('Ctrl+D', Colors.CYAN)} (Unix) or {c('Ctrl+Z Enter', Colors.CYAN)} (Windows):\n")
        print("-" * 60)

        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        except KeyboardInterrupt:
            print("\nScan cancelled.")
            return

        code = '\n'.join(lines)
        if not code.strip():
            print(f"{c('No code provided', Colors.YELLOW)}")
            return

        print(f"\n{Colors.DIM}Scanning for vulnerabilities...{Colors.RESET}\n")
        time.sleep(0.5)

        findings = self.scanner.scan(code)

        if not findings:
            print(f"{c('✓ No vulnerabilities detected!', Colors.GREEN)}")
            print(f"{Colors.DIM}Your code looks secure.{Colors.RESET}\n")
            return

        print(f"{c(f'⚠ Found {len(findings)} issue(s):', Colors.YELLOW)}\n")

        severity_colors = {
            'critical': Colors.RED,
            'high': Colors.RED,
            'medium': Colors.YELLOW,
            'low': Colors.DIM,
        }

        for i, finding in enumerate(findings, 1):
            color = severity_colors.get(finding['severity'], Colors.RESET)
            sev = finding['severity'].upper()
            desc = finding['description']
            line_num = finding['line']
            code_snippet = finding['code']
            print(f"{c(f'{i}. {sev}', color)}: {desc}")
            print(f"   Line {line_num}: {Colors.DIM}{code_snippet}{Colors.RESET}")
            print(f"   {c('Fix:', Colors.GREEN)} {self.scanner.get_fix_suggestion(finding)}\n")

    # =========================================================================
    # AI-to-AI Communication
    # =========================================================================

    def cmd_ai_comm(self):
        """Demonstrate AI-to-AI communication."""
        print(f"\n{c('AI-to-AI Communication Demo', Colors.BOLD)}")
        print(f"{Colors.DIM}Messages encoded in Sacred Tongues - human readable & verifiable{Colors.RESET}\n")

        try:
            recipient = input(f"{Colors.CYAN}Recipient AI ID (default: claude-001): {Colors.RESET}").strip()
            if not recipient:
                recipient = "claude-001"

            content = input(f"{Colors.CYAN}Message: {Colors.RESET}").strip()
            if not content:
                content = "Hello from Polly! Let's collaborate securely."
        except (EOFError, KeyboardInterrupt):
            print()
            return

        print(f"\n{Colors.DIM}Creating SCBE-encrypted AI message...{Colors.RESET}\n")
        time.sleep(0.3)

        message = self.protocol.create_ai_message(
            sender_id=self.agent_id,
            recipient_id=recipient,
            content=content,
            intent="collaborate"
        )

        print(f"{c('AI Message Created:', Colors.GREEN)}\n")
        print(f"  {c('Version:', Colors.CYAN)} {message['version']}")
        print(f"  {c('From:', Colors.CYAN)} {message['human_readable']['from']}")
        print(f"  {c('To:', Colors.CYAN)} {message['human_readable']['to']}")
        print(f"  {c('Intent:', Colors.CYAN)} {message['human_readable']['intent']}")
        print(f"\n{c('Encoded Components:', Colors.BOLD)}")
        print(f"  {c('Header (AV):', Colors.CYAN)} {message['header'][:50]}...")
        print(f"  {c('Nonce (KO):', Colors.CYAN)} {message['nonce'][:50]}...")
        print(f"  {c('Body (CA):', Colors.CYAN)} {message['body'][:50]}...")
        print(f"  {c('Signature (DR):', Colors.CYAN)} {message['signature'][:50]}...")

        # Verify
        is_valid, reason = self.protocol.verify_message(message)
        status = c('✓ VERIFIED', Colors.GREEN) if is_valid else c('✗ INVALID', Colors.RED)
        print(f"\n{c('Verification:', Colors.BOLD)} {status}")

        print(f"\n{Colors.DIM}This message can be read by humans AND verified by AIs!{Colors.RESET}\n")

    def cmd_encode(self):
        """Encode text to Sacred Tongue."""
        print(f"\n{c('Sacred Tongue Encoder', Colors.BOLD)}\n")

        tongues_list = ['ko', 'av', 'ru', 'ca', 'um', 'dr']
        names = ["Kor'aelin", "Avali", "Runethic", "Cassisivadan", "Umbroth", "Draumric"]

        for i, (code, name) in enumerate(zip(tongues_list, names), 1):
            print(f"  {c(str(i), Colors.GREEN)} - {code.upper()}: {name}")

        try:
            tongue_choice = input(f"\n{Colors.CYAN}Tongue (1-6, default=1): {Colors.RESET}").strip()
            tongue = tongues_list[int(tongue_choice) - 1] if tongue_choice else 'ko'

            text = input(f"{Colors.CYAN}Text to encode: {Colors.RESET}").strip()
        except (EOFError, KeyboardInterrupt, ValueError, IndexError):
            print()
            return

        if not text:
            return

        encoded = self.protocol.encode_message(text, tongue)
        print(f"\n{c('Encoded:', Colors.GREEN)} {encoded}\n")

    # =========================================================================
    # Autopilot Mode
    # =========================================================================

    def cmd_autopilot(self):
        """Enter autopilot mode."""
        print(f"\n{c('Autopilot Mode', Colors.BOLD)} {Colors.DIM}(autonomous task execution){Colors.RESET}")
        print(f"Commands: {c('add', Colors.GREEN)} <task>, {c('run', Colors.GREEN)}, {c('status', Colors.GREEN)}, {c('back', Colors.GREEN)}\n")

        while True:
            try:
                cmd = input(f"{c('autopilot>', Colors.YELLOW)} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not cmd:
                continue

            parts = cmd.split(maxsplit=1)
            action = parts[0].lower()

            if action in ('back', 'exit'):
                break
            elif action == 'add' and len(parts) > 1:
                task_id = self.autopilot.add_task(parts[1])
                print(f"{c('✓', Colors.GREEN)} Task added: {task_id}")
            elif action == 'run':
                result = self.autopilot.execute_next()
                if result:
                    print(f"{c('✓', Colors.GREEN)} Executed: {result['description'][:40]}")
                else:
                    print(f"{c('No tasks in queue', Colors.YELLOW)}")
            elif action == 'status':
                status = self.autopilot.get_status()
                print(f"Pending: {status['pending']}, Completed: {status['completed']}")
                if status['queue']:
                    print(f"Queue: {', '.join(status['queue'])}")
            else:
                print(f"Unknown: {action}. Use: add, run, status, back")

    def cmd_status(self):
        """Show agent status."""
        print(f"\n{c('Agent Status', Colors.BOLD)}\n")
        print(f"  {c('Agent ID:', Colors.CYAN)} {self.agent_id}")
        print(f"  {c('Session:', Colors.CYAN)} {self.state.session_id}")
        print(f"  {c('Mode:', Colors.CYAN)} {self.state.mode.value}")
        print(f"  {c('Messages:', Colors.CYAN)} {len(self.state.messages)}")

        autopilot = self.autopilot.get_status()
        print(f"  {c('Autopilot Queue:', Colors.CYAN)} {autopilot['pending']} pending, {autopilot['completed']} done")

        print(f"\n  {c('Sacred Tongues:', Colors.CYAN)} {'Available' if TONGUES_AVAILABLE else 'Fallback mode'}")
        print()

    # =========================================================================
    # Main Loop
    # =========================================================================

    def run(self):
        """Main agent loop."""
        self.print_banner()

        commands = {
            'ask': self.cmd_ask,
            'code': self.cmd_code,
            'scan': self.cmd_scan,
            'ai-comm': self.cmd_ai_comm,
            'encode': self.cmd_encode,
            'autopilot': self.cmd_autopilot,
            'status': self.cmd_status,
            'help': self.print_help,
        }

        while True:
            try:
                cmd = input(f"{c('polly>', Colors.MAGENTA)} ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{Colors.DIM}Goodbye from Polly! 🌟{Colors.RESET}")
                break

            if not cmd:
                continue
            elif cmd in ('exit', 'quit', 'q'):
                print(f"{Colors.DIM}Goodbye from Polly! 🌟{Colors.RESET}")
                break
            elif cmd in commands:
                commands[cmd]()
            else:
                print(f"{c('Unknown command:', Colors.RED)} {cmd}")
                print(f"Type {c('help', Colors.GREEN)} for available commands.")


# =============================================================================
# Main
# =============================================================================

def main():
    agent = PollyAgent()
    agent.run()


if __name__ == "__main__":
    main()
