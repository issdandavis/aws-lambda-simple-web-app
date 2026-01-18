"""
SCBE Production Exceptions
==========================

Commercial-grade exception hierarchy for SCBE operations.
All exceptions include error codes for API responses.
"""

from typing import Any, Dict, Optional


class SCBEError(Exception):
    """Base exception for all SCBE errors."""

    code: str = "SCBE_ERROR"
    http_status: int = 500

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.code
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-friendly dictionary."""
        result = {
            "error": True,
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code!r}, message={self.message!r})"


class PQCError(SCBEError):
    """Post-quantum cryptography errors."""

    code = "PQC_ERROR"

    @classmethod
    def key_generation_failed(cls, algorithm: str, reason: str) -> "PQCError":
        return cls(
            f"Key generation failed for {algorithm}: {reason}",
            code="PQC_KEYGEN_FAILED",
            details={"algorithm": algorithm, "reason": reason},
        )

    @classmethod
    def encapsulation_failed(cls, reason: str) -> "PQCError":
        return cls(
            f"Key encapsulation failed: {reason}",
            code="PQC_ENCAP_FAILED",
            details={"reason": reason},
        )

    @classmethod
    def decapsulation_failed(cls, reason: str) -> "PQCError":
        return cls(
            f"Key decapsulation failed: {reason}",
            code="PQC_DECAP_FAILED",
            details={"reason": reason},
        )

    @classmethod
    def signature_failed(cls, reason: str) -> "PQCError":
        return cls(
            f"Digital signature failed: {reason}",
            code="PQC_SIGN_FAILED",
            details={"reason": reason},
        )

    @classmethod
    def verification_failed(cls, reason: str) -> "PQCError":
        return cls(
            f"Signature verification failed: {reason}",
            code="PQC_VERIFY_FAILED",
            details={"reason": reason},
        )


class GovernanceError(SCBEError):
    """Governance and access control errors."""

    code = "GOVERNANCE_ERROR"
    http_status = 403

    @classmethod
    def access_denied(cls, reason: str, risk_score: float) -> "GovernanceError":
        return cls(
            f"Access denied: {reason}",
            code="ACCESS_DENIED",
            details={"reason": reason, "risk_score": risk_score},
        )

    @classmethod
    def snap_triggered(cls, risk_score: float, threshold: float) -> "GovernanceError":
        return cls(
            f"SNAP protocol triggered: risk {risk_score:.3f} exceeds threshold {threshold:.3f}",
            code="SNAP_TRIGGERED",
            details={"risk_score": risk_score, "threshold": threshold},
        )

    @classmethod
    def quarantine_required(cls, reason: str, risk_score: float) -> "GovernanceError":
        return cls(
            f"Request quarantined: {reason}",
            code="QUARANTINE_REQUIRED",
            details={"reason": reason, "risk_score": risk_score},
        )

    @classmethod
    def consensus_failed(
        cls, required: int, achieved: int, tongues: list
    ) -> "GovernanceError":
        return cls(
            f"Roundtable consensus failed: {achieved}/{required} signatures",
            code="CONSENSUS_FAILED",
            details={
                "required_signatures": required,
                "achieved_signatures": achieved,
                "signing_tongues": tongues,
            },
        )


class ValidationError(SCBEError):
    """Input validation errors."""

    code = "VALIDATION_ERROR"
    http_status = 400

    @classmethod
    def invalid_input(cls, field: str, reason: str) -> "ValidationError":
        return cls(
            f"Invalid input for {field}: {reason}",
            code="INVALID_INPUT",
            details={"field": field, "reason": reason},
        )

    @classmethod
    def invalid_position(cls, position: tuple, reason: str) -> "ValidationError":
        return cls(
            f"Invalid voxel position {position}: {reason}",
            code="INVALID_POSITION",
            details={"position": position, "reason": reason},
        )

    @classmethod
    def quasicrystal_invalid(
        cls, status: str, crystallinity: float
    ) -> "ValidationError":
        return cls(
            f"Quasicrystal validation failed: {status}",
            code="QC_VALIDATION_FAILED",
            details={"status": status, "crystallinity": crystallinity},
        )


class AuthenticationError(SCBEError):
    """Authentication and identity errors."""

    code = "AUTH_ERROR"
    http_status = 401

    @classmethod
    def invalid_credentials(cls) -> "AuthenticationError":
        return cls("Invalid credentials", code="INVALID_CREDENTIALS")

    @classmethod
    def expired_token(cls) -> "AuthenticationError":
        return cls("Authentication token expired", code="TOKEN_EXPIRED")

    @classmethod
    def insufficient_permissions(cls, required: str) -> "AuthenticationError":
        return cls(
            f"Insufficient permissions: {required} required",
            code="INSUFFICIENT_PERMISSIONS",
            details={"required_permission": required},
        )

    @classmethod
    def geometric_mismatch(cls, distance: float, threshold: float) -> "AuthenticationError":
        return cls(
            f"Geometric trust verification failed: distance {distance:.4f} exceeds threshold {threshold:.4f}",
            code="GEOMETRIC_MISMATCH",
            details={"distance": distance, "threshold": threshold},
        )


class CryptoError(SCBEError):
    """Cryptographic operation errors."""

    code = "CRYPTO_ERROR"

    @classmethod
    def seal_failed(cls, reason: str) -> "CryptoError":
        return cls(f"Seal operation failed: {reason}", code="SEAL_FAILED")

    @classmethod
    def unseal_failed(cls, reason: str) -> "CryptoError":
        return cls(f"Unseal operation failed: {reason}", code="UNSEAL_FAILED")

    @classmethod
    def integrity_violation(cls) -> "CryptoError":
        return cls("Data integrity check failed", code="INTEGRITY_VIOLATION")


class LatticeError(SCBEError):
    """Dual lattice consensus errors."""

    code = "LATTICE_ERROR"

    @classmethod
    def primal_failed(cls) -> "LatticeError":
        return cls("Primal lattice (Kyber) verification failed", code="PRIMAL_FAILED")

    @classmethod
    def dual_failed(cls) -> "LatticeError":
        return cls("Dual lattice (Dilithium) verification failed", code="DUAL_FAILED")

    @classmethod
    def consensus_timeout(cls, elapsed: float) -> "LatticeError":
        return cls(
            f"Lattice consensus timeout after {elapsed:.2f}s",
            code="CONSENSUS_TIMEOUT",
            details={"elapsed_seconds": elapsed},
        )
