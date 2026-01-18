"""
SCBE Production Audit Logging
=============================

Structured logging with audit trail for compliance and debugging.
Supports JSON and text formats, with sensitive data redaction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .config import get_config, AuditConfig


class AuditEventType(Enum):
    """Types of audit events."""
    # Authentication
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    AUTH_EXPIRED = "auth.expired"

    # Cryptographic operations
    SEAL_REQUEST = "crypto.seal.request"
    SEAL_SUCCESS = "crypto.seal.success"
    SEAL_FAILURE = "crypto.seal.failure"
    UNSEAL_REQUEST = "crypto.unseal.request"
    UNSEAL_SUCCESS = "crypto.unseal.success"
    UNSEAL_FAILURE = "crypto.unseal.failure"

    # PQC operations
    PQC_KEYGEN = "pqc.keygen"
    PQC_ENCAP = "pqc.encapsulate"
    PQC_DECAP = "pqc.decapsulate"
    PQC_SIGN = "pqc.sign"
    PQC_VERIFY = "pqc.verify"

    # Governance decisions
    GOV_ALLOW = "governance.allow"
    GOV_QUARANTINE = "governance.quarantine"
    GOV_DENY = "governance.deny"
    GOV_SNAP = "governance.snap"

    # GeoSeal
    GEOSEAL_INTERIOR = "geoseal.interior"
    GEOSEAL_EXTERIOR = "geoseal.exterior"

    # Lattice consensus
    LATTICE_SUCCESS = "lattice.consensus.success"
    LATTICE_FAILURE = "lattice.consensus.failure"

    # System events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"


@dataclass
class AuditEvent:
    """Structured audit event."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: AuditEventType = AuditEventType.SYSTEM_ERROR
    service: str = "scbe-production"
    version: str = "1.0.0"

    # Request context
    request_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None

    # Event data
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    # Security context
    risk_score: Optional[float] = None
    decision: Optional[str] = None
    geometric_distance: Optional[float] = None

    # Performance
    duration_ms: Optional[float] = None

    # Error handling
    error: Optional[str] = None
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None

    def to_dict(self, redact: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "service": self.service,
            "version": self.version,
            "message": self.message,
        }

        if self.request_id:
            result["request_id"] = self.request_id
        if self.agent_id:
            result["agent_id"] = self._maybe_redact(self.agent_id, redact)
        if self.session_id:
            result["session_id"] = self._maybe_redact(self.session_id, redact)

        if self.details:
            result["details"] = self._redact_dict(self.details) if redact else self.details

        if self.risk_score is not None:
            result["risk_score"] = round(self.risk_score, 4)
        if self.decision:
            result["decision"] = self.decision
        if self.geometric_distance is not None:
            result["geometric_distance"] = round(self.geometric_distance, 4)

        if self.duration_ms is not None:
            result["duration_ms"] = round(self.duration_ms, 2)

        if self.error:
            result["error"] = self.error
        if self.error_code:
            result["error_code"] = self.error_code
        if self.stack_trace:
            result["stack_trace"] = self.stack_trace

        return result

    def _maybe_redact(self, value: str, redact: bool) -> str:
        """Optionally redact a value by hashing."""
        if not redact:
            return value
        return f"hash:{hashlib.sha256(value.encode()).hexdigest()[:16]}"

    def _redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive fields in a dictionary."""
        sensitive_keys = {"password", "secret", "key", "token", "credential", "auth"}
        result = {}
        for k, v in data.items():
            if any(s in k.lower() for s in sensitive_keys):
                result[k] = "[REDACTED]"
            elif isinstance(v, dict):
                result[k] = self._redact_dict(v)
            elif isinstance(v, bytes):
                result[k] = f"<bytes:{len(v)}>"
            else:
                result[k] = v
        return result

    def to_json(self, redact: bool = True) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(redact), default=str)


class AuditLogger:
    """
    Structured audit logger for SCBE operations.

    Provides:
    - Structured JSON logging
    - Automatic sensitive data redaction
    - Request context propagation
    - Performance timing
    - Error tracking with stack traces
    """

    def __init__(
        self,
        name: str = "scbe.audit",
        config: Optional[AuditConfig] = None,
    ):
        self.name = name
        self.config = config or get_config().audit
        self._logger = logging.getLogger(name)
        self._setup_logger()
        self._request_context: Dict[str, str] = {}

    def _setup_logger(self):
        """Configure the underlying logger."""
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        self._logger.setLevel(level)

        # Only add handler if none exist
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)

            if self.config.log_format == "json":
                formatter = logging.Formatter("%(message)s")
            else:
                formatter = logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                )

            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

    def set_request_context(
        self,
        request_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Set context for current request."""
        if request_id:
            self._request_context["request_id"] = request_id
        if agent_id:
            self._request_context["agent_id"] = agent_id
        if session_id:
            self._request_context["session_id"] = session_id

    def clear_request_context(self):
        """Clear request context."""
        self._request_context.clear()

    def _create_event(
        self,
        event_type: AuditEventType,
        message: str,
        **kwargs,
    ) -> AuditEvent:
        """Create an audit event with context."""
        event = AuditEvent(
            event_type=event_type,
            message=message,
            request_id=self._request_context.get("request_id"),
            agent_id=self._request_context.get("agent_id"),
            session_id=self._request_context.get("session_id"),
            **kwargs,
        )
        return event

    def _log_event(self, event: AuditEvent, level: int = logging.INFO):
        """Log an audit event."""
        if not self.config.enabled:
            return

        if self.config.log_format == "json":
            self._logger.log(level, event.to_json(self.config.redact_sensitive))
        else:
            self._logger.log(level, f"{event.event_type.value}: {event.message}")

    # Convenience methods for common events

    def auth_success(self, agent_id: str, method: str = "pqc"):
        """Log successful authentication."""
        event = self._create_event(
            AuditEventType.AUTH_SUCCESS,
            f"Authentication successful for agent via {method}",
            agent_id=agent_id,
            details={"auth_method": method},
        )
        self._log_event(event)

    def auth_failure(self, reason: str, agent_id: Optional[str] = None):
        """Log authentication failure."""
        event = self._create_event(
            AuditEventType.AUTH_FAILURE,
            f"Authentication failed: {reason}",
            agent_id=agent_id,
            error=reason,
        )
        self._log_event(event, logging.WARNING)

    def seal_success(self, position: tuple, duration_ms: float):
        """Log successful seal operation."""
        event = self._create_event(
            AuditEventType.SEAL_SUCCESS,
            f"Memory sealed at position {position}",
            details={"position": position},
            duration_ms=duration_ms,
        )
        self._log_event(event)

    def seal_failure(self, error: str, position: Optional[tuple] = None):
        """Log seal failure."""
        event = self._create_event(
            AuditEventType.SEAL_FAILURE,
            f"Seal operation failed: {error}",
            details={"position": position} if position else {},
            error=error,
        )
        self._log_event(event, logging.ERROR)

    def governance_decision(
        self,
        decision: str,
        risk_score: float,
        reason: str,
        agent_id: Optional[str] = None,
    ):
        """Log governance decision."""
        event_map = {
            "ALLOW": AuditEventType.GOV_ALLOW,
            "QUARANTINE": AuditEventType.GOV_QUARANTINE,
            "DENY": AuditEventType.GOV_DENY,
            "SNAP": AuditEventType.GOV_SNAP,
        }
        event_type = event_map.get(decision, AuditEventType.GOV_DENY)

        level = logging.INFO if decision == "ALLOW" else logging.WARNING

        event = self._create_event(
            event_type,
            f"Governance decision: {decision} (risk={risk_score:.3f})",
            decision=decision,
            risk_score=risk_score,
            details={"reason": reason, "target_agent": agent_id},
        )
        self._log_event(event, level)

    def geoseal_result(
        self,
        path: str,
        distance: float,
        time_dilation: float,
    ):
        """Log GeoSeal verification result."""
        event_type = (
            AuditEventType.GEOSEAL_INTERIOR
            if path == "interior"
            else AuditEventType.GEOSEAL_EXTERIOR
        )
        event = self._create_event(
            event_type,
            f"GeoSeal: {path} path (distance={distance:.4f})",
            geometric_distance=distance,
            details={"path": path, "time_dilation": time_dilation},
        )
        level = logging.INFO if path == "interior" else logging.WARNING
        self._log_event(event, level)

    def lattice_consensus(
        self,
        success: bool,
        primal_ok: bool,
        dual_ok: bool,
        duration_ms: float,
    ):
        """Log lattice consensus result."""
        event_type = (
            AuditEventType.LATTICE_SUCCESS
            if success
            else AuditEventType.LATTICE_FAILURE
        )
        event = self._create_event(
            event_type,
            f"Lattice consensus: {'achieved' if success else 'failed'}",
            details={
                "primal_ok": primal_ok,
                "dual_ok": dual_ok,
            },
            duration_ms=duration_ms,
        )
        level = logging.INFO if success else logging.WARNING
        self._log_event(event, level)

    def error(
        self,
        message: str,
        error_code: Optional[str] = None,
        exception: Optional[Exception] = None,
    ):
        """Log an error."""
        stack_trace = None
        if exception and self.config.include_stack_traces:
            stack_trace = traceback.format_exc()

        event = self._create_event(
            AuditEventType.SYSTEM_ERROR,
            message,
            error=str(exception) if exception else message,
            error_code=error_code,
            stack_trace=stack_trace,
        )
        self._log_event(event, logging.ERROR)

    def warning(self, message: str, details: Optional[Dict] = None):
        """Log a warning."""
        event = self._create_event(
            AuditEventType.SYSTEM_WARNING,
            message,
            details=details or {},
        )
        self._log_event(event, logging.WARNING)

    def info(self, message: str, details: Optional[Dict] = None):
        """Log an info message."""
        event = self._create_event(
            AuditEventType.SYSTEM_START,  # Generic info
            message,
            details=details or {},
        )
        self._log_event(event)


# Global logger instance
_logger: Optional[AuditLogger] = None


def get_logger(name: str = "scbe.audit") -> AuditLogger:
    """Get or create the audit logger."""
    global _logger
    if _logger is None:
        _logger = AuditLogger(name)
    return _logger


class TimedOperation:
    """Context manager for timing operations."""

    def __init__(self, name: str, logger: Optional[AuditLogger] = None):
        self.name = name
        self.logger = logger or get_logger()
        self.start_time: float = 0
        self.duration_ms: float = 0

    def __enter__(self) -> "TimedOperation":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration_ms = (time.perf_counter() - self.start_time) * 1000
        return False  # Don't suppress exceptions
