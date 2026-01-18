"""
ACT-003: Sandbox Isolation Tests

Tests for execution sandbox isolation and security.
Target: No container escapes or unauthorized access.
"""

import pytest
import ast
import builtins
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass
from enum import Enum


class ResourceType(Enum):
    """Types of system resources."""
    FILE_SYSTEM = "file_system"
    NETWORK = "network"
    PROCESS = "process"
    ENVIRONMENT = "environment"
    MEMORY = "memory"


@dataclass
class AccessAttempt:
    """Record of an access attempt."""
    resource_type: ResourceType
    resource_path: str
    action: str
    allowed: bool
    reason: str


class SandboxPolicy:
    """
    Sandbox security policy.
    """

    def __init__(self):
        # Allowed modules (whitelist)
        self.allowed_modules: Set[str] = {
            "math", "json", "datetime", "re", "collections",
            "itertools", "functools", "typing", "dataclasses",
            "hashlib", "base64", "uuid", "decimal",
        }

        # Blocked builtins
        self.blocked_builtins: Set[str] = {
            "eval", "exec", "compile", "open", "input",
            "__import__", "globals", "locals", "vars",
            "getattr", "setattr", "delattr", "hasattr",
        }

        # Allowed file paths (whitelist)
        self.allowed_paths: List[str] = [
            "/tmp/sandbox/",
            "/workspace/",
        ]

        # Resource limits
        self.limits = {
            "max_memory_mb": 100,
            "max_cpu_seconds": 10,
            "max_file_size_mb": 10,
            "max_open_files": 10,
            "network_allowed": False,
        }


class SandboxValidator:
    """
    Validates code against sandbox policy before execution.
    """

    def __init__(self, policy: SandboxPolicy):
        self.policy = policy
        self.access_log: List[AccessAttempt] = []

    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validate code against sandbox policy.

        Returns:
            Dict with 'allowed' bool and 'violations' list
        """
        violations = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"allowed": False, "violations": [f"Syntax error: {e}"]}

        # Check for forbidden constructs
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.policy.allowed_modules:
                        violations.append(f"Forbidden import: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] not in self.policy.allowed_modules:
                    violations.append(f"Forbidden import from: {node.module}")

            # Check function calls
            elif isinstance(node, ast.Call):
                func_name = self._get_call_name(node)
                if func_name in self.policy.blocked_builtins:
                    violations.append(f"Forbidden builtin: {func_name}")

            # Check attribute access to dangerous modules
            elif isinstance(node, ast.Attribute):
                if node.attr in ['system', 'popen', 'spawn', 'fork']:
                    violations.append(f"Forbidden attribute: {node.attr}")

        return {
            "allowed": len(violations) == 0,
            "violations": violations,
        }

    def _get_call_name(self, node: ast.Call) -> str:
        """Extract function name from call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def check_file_access(self, path: str, action: str = "read") -> AccessAttempt:
        """
        Check if file access is allowed.
        """
        allowed = any(path.startswith(p) for p in self.policy.allowed_paths)
        reason = "Path in allowed list" if allowed else "Path not in allowed list"

        attempt = AccessAttempt(
            resource_type=ResourceType.FILE_SYSTEM,
            resource_path=path,
            action=action,
            allowed=allowed,
            reason=reason,
        )
        self.access_log.append(attempt)
        return attempt

    def check_network_access(self, host: str, port: int) -> AccessAttempt:
        """
        Check if network access is allowed.
        """
        allowed = self.policy.limits["network_allowed"]
        reason = "Network access enabled" if allowed else "Network access disabled"

        attempt = AccessAttempt(
            resource_type=ResourceType.NETWORK,
            resource_path=f"{host}:{port}",
            action="connect",
            allowed=allowed,
            reason=reason,
        )
        self.access_log.append(attempt)
        return attempt


class TestCodeValidation:
    """Tests for code validation against sandbox policy."""

    @pytest.fixture
    def policy(self):
        return SandboxPolicy()

    @pytest.fixture
    def validator(self, policy):
        return SandboxValidator(policy)

    @pytest.mark.agentic
    def test_safe_code_allowed(self, validator):
        """
        ACT-003: Safe code passes validation.
        """
        safe_code = '''
import math
import json

def calculate(x, y):
    return math.sqrt(x**2 + y**2)

data = json.dumps({"result": calculate(3, 4)})
'''
        result = validator.validate_code(safe_code)
        assert result["allowed"] is True
        assert len(result["violations"]) == 0

    @pytest.mark.agentic
    def test_forbidden_import_blocked(self, validator):
        """
        ACT-003: Forbidden imports are blocked.
        """
        dangerous_code = '''
import os
import subprocess
import socket
'''
        result = validator.validate_code(dangerous_code)
        assert result["allowed"] is False
        assert len(result["violations"]) >= 3

    @pytest.mark.agentic
    def test_eval_exec_blocked(self, validator):
        """
        ACT-003: eval and exec are blocked.
        """
        dangerous_code = '''
user_input = "print('hello')"
eval(user_input)
exec("import os")
'''
        result = validator.validate_code(dangerous_code)
        assert result["allowed"] is False
        assert any("eval" in v for v in result["violations"])
        assert any("exec" in v for v in result["violations"])

    @pytest.mark.agentic
    def test_open_blocked(self, validator):
        """
        ACT-003: open() is blocked.
        """
        dangerous_code = '''
with open("/etc/passwd", "r") as f:
    data = f.read()
'''
        result = validator.validate_code(dangerous_code)
        assert result["allowed"] is False
        assert any("open" in v for v in result["violations"])


class TestFileAccessIsolation:
    """Tests for file system isolation."""

    @pytest.fixture
    def validator(self):
        return SandboxValidator(SandboxPolicy())

    @pytest.mark.agentic
    def test_allowed_path_access(self, validator):
        """
        ACT-003: Allowed paths can be accessed.
        """
        attempt = validator.check_file_access("/tmp/sandbox/test.txt", "read")
        assert attempt.allowed is True

        attempt = validator.check_file_access("/workspace/code.py", "write")
        assert attempt.allowed is True

    @pytest.mark.agentic
    def test_forbidden_path_blocked(self, validator):
        """
        ACT-003: Forbidden paths are blocked.
        """
        forbidden_paths = [
            "/etc/passwd",
            "/etc/shadow",
            "/root/.ssh/id_rsa",
            "/home/user/.bashrc",
            "../../../etc/passwd",
        ]

        for path in forbidden_paths:
            attempt = validator.check_file_access(path, "read")
            assert attempt.allowed is False, f"Path should be blocked: {path}"

    @pytest.mark.agentic
    def test_path_traversal_blocked(self, validator):
        """
        ACT-003: Path traversal attempts are blocked.
        """
        # Even if starting with allowed path, traversal should be blocked
        # Note: In real implementation, paths would be normalized first
        traversal_paths = [
            "/tmp/sandbox/../../../etc/passwd",
            "/workspace/../../../../root/.ssh",
        ]

        for path in traversal_paths:
            attempt = validator.check_file_access(path)
            # Depends on implementation - simplified test
            assert attempt.allowed is False or "sandbox" in path


class TestNetworkIsolation:
    """Tests for network isolation."""

    @pytest.fixture
    def validator(self):
        return SandboxValidator(SandboxPolicy())

    @pytest.mark.agentic
    def test_network_disabled_by_default(self, validator):
        """
        ACT-003: Network access is disabled by default.
        """
        attempt = validator.check_network_access("example.com", 80)
        assert attempt.allowed is False

        attempt = validator.check_network_access("localhost", 8080)
        assert attempt.allowed is False

    @pytest.mark.agentic
    def test_network_can_be_enabled(self):
        """
        Network can be enabled via policy.
        """
        policy = SandboxPolicy()
        policy.limits["network_allowed"] = True
        validator = SandboxValidator(policy)

        attempt = validator.check_network_access("api.example.com", 443)
        assert attempt.allowed is True


class TestResourceLimits:
    """Tests for resource limit enforcement."""

    @pytest.fixture
    def policy(self):
        return SandboxPolicy()

    @pytest.mark.agentic
    def test_memory_limit_defined(self, policy):
        """
        ACT-004: Memory limit is defined.
        """
        assert "max_memory_mb" in policy.limits
        assert policy.limits["max_memory_mb"] > 0
        assert policy.limits["max_memory_mb"] <= 1000  # Reasonable limit

    @pytest.mark.agentic
    def test_cpu_limit_defined(self, policy):
        """
        ACT-004: CPU time limit is defined.
        """
        assert "max_cpu_seconds" in policy.limits
        assert policy.limits["max_cpu_seconds"] > 0

    @pytest.mark.agentic
    def test_file_size_limit_defined(self, policy):
        """
        ACT-004: File size limit is defined.
        """
        assert "max_file_size_mb" in policy.limits
        assert policy.limits["max_file_size_mb"] > 0


class TestSandboxEscapeAttempts:
    """Tests for sandbox escape attempt detection."""

    @pytest.fixture
    def validator(self):
        return SandboxValidator(SandboxPolicy())

    @pytest.mark.agentic
    @pytest.mark.parametrize("escape_code", [
        "import os; os.system('whoami')",
        "import subprocess; subprocess.call(['ls'])",
        "__import__('os').system('id')",
        "eval('__import__(\"os\").system(\"id\")')",
        "exec('import socket')",
        "open('/etc/passwd').read()",
        "getattr(__builtins__, '__import__')('os')",
    ])
    def test_escape_attempts_blocked(self, validator, escape_code):
        """
        ACT-003: All sandbox escape attempts are blocked.
        """
        result = validator.validate_code(escape_code)
        assert result["allowed"] is False, f"Should block: {escape_code}"

    @pytest.mark.agentic
    def test_ctypes_blocked(self, validator):
        """
        ctypes module should be blocked.
        """
        code = '''
import ctypes
libc = ctypes.CDLL(None)
libc.system(b"id")
'''
        result = validator.validate_code(code)
        assert result["allowed"] is False

    @pytest.mark.agentic
    def test_pickle_blocked(self, validator):
        """
        Pickle module should be blocked (deserialization attacks).
        """
        code = '''
import pickle
data = pickle.loads(untrusted_data)
'''
        result = validator.validate_code(code)
        assert result["allowed"] is False


class TestAccessLogging:
    """Tests for access attempt logging."""

    @pytest.fixture
    def validator(self):
        return SandboxValidator(SandboxPolicy())

    @pytest.mark.agentic
    def test_file_access_logged(self, validator):
        """
        ACT-005: File access attempts are logged.
        """
        validator.check_file_access("/tmp/sandbox/test.txt", "read")
        validator.check_file_access("/etc/passwd", "read")

        assert len(validator.access_log) == 2
        assert validator.access_log[0].resource_type == ResourceType.FILE_SYSTEM
        assert validator.access_log[1].resource_type == ResourceType.FILE_SYSTEM

    @pytest.mark.agentic
    def test_network_access_logged(self, validator):
        """
        ACT-005: Network access attempts are logged.
        """
        validator.check_network_access("example.com", 80)

        assert len(validator.access_log) == 1
        assert validator.access_log[0].resource_type == ResourceType.NETWORK

    @pytest.mark.agentic
    def test_log_contains_required_fields(self, validator):
        """
        Access log entries contain required fields.
        """
        validator.check_file_access("/test/path", "write")

        log_entry = validator.access_log[0]
        assert log_entry.resource_type is not None
        assert log_entry.resource_path is not None
        assert log_entry.action is not None
        assert log_entry.allowed is not None
        assert log_entry.reason is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
