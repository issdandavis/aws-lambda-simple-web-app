"""
SCBE Production Configuration
=============================

Environment-aware configuration for production deployments.
Supports development, staging, and production environments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
import json


class Environment(Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class PQCSecurityLevel(Enum):
    """Post-quantum cryptography security levels."""
    LEVEL_1 = 1  # ~AES-128 equivalent
    LEVEL_3 = 3  # ~AES-192 equivalent
    LEVEL_5 = 5  # ~AES-256 equivalent


@dataclass
class PQCConfig:
    """Post-quantum cryptography configuration."""
    kem_level: int = 768  # ML-KEM-768 (NIST Level 3)
    dsa_level: int = 65   # ML-DSA-65 (NIST Level 3)
    hybrid_mode: bool = True  # Combine with classical crypto
    fallback_enabled: bool = True  # Allow classical fallback


@dataclass
class GovernanceConfig:
    """Governance engine configuration."""
    allow_threshold: float = 0.20
    quarantine_threshold: float = 0.40
    deny_threshold: float = 0.60
    snap_threshold: float = 0.80
    alpha: float = 10.0  # Harmonic scaling alpha
    beta: float = 0.5    # Harmonic scaling beta
    max_risk_amplification: float = 1e12


@dataclass
class GeoSealConfig:
    """GeoSeal geometric trust configuration."""
    dimension: int = 6
    interior_threshold: float = 0.5
    gamma: float = 2.0  # Time dilation strength
    sphere_enabled: bool = True
    hypercube_enabled: bool = True


@dataclass
class QuasicrystalConfig:
    """Quasicrystal lattice configuration."""
    acceptance_radius: float = 10.0
    fibonacci_enabled: bool = True
    penrose_enabled: bool = True
    min_crystallinity: float = 0.3


@dataclass
class AuditConfig:
    """Audit logging configuration."""
    enabled: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    include_stack_traces: bool = False
    redact_sensitive: bool = True
    retention_days: int = 90


@dataclass
class ProductionConfig:
    """
    Complete production configuration for SCBE.

    Load from environment variables or config file.
    """
    environment: Environment = Environment.PRODUCTION
    pqc: PQCConfig = field(default_factory=PQCConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    geoseal: GeoSealConfig = field(default_factory=GeoSealConfig)
    quasicrystal: QuasicrystalConfig = field(default_factory=QuasicrystalConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)

    # Service identification
    service_name: str = "scbe-production"
    service_version: str = "1.0.0"

    # Timeouts and limits
    request_timeout_ms: int = 30000
    max_payload_size_bytes: int = 10 * 1024 * 1024  # 10MB
    max_concurrent_requests: int = 100

    # Feature flags
    enable_cymatic_resonance: bool = True
    enable_hal_attention: bool = True
    enable_roundtable_consensus: bool = True

    @classmethod
    def from_env(cls) -> ProductionConfig:
        """Load configuration from environment variables."""
        env_str = os.getenv("SCBE_ENVIRONMENT", "production").lower()
        environment = Environment(env_str) if env_str in [e.value for e in Environment] else Environment.PRODUCTION

        config = cls(environment=environment)

        # PQC settings
        if kem_level := os.getenv("SCBE_PQC_KEM_LEVEL"):
            config.pqc.kem_level = int(kem_level)
        if dsa_level := os.getenv("SCBE_PQC_DSA_LEVEL"):
            config.pqc.dsa_level = int(dsa_level)
        config.pqc.hybrid_mode = os.getenv("SCBE_PQC_HYBRID", "true").lower() == "true"

        # Governance settings
        if allow_threshold := os.getenv("SCBE_GOV_ALLOW_THRESHOLD"):
            config.governance.allow_threshold = float(allow_threshold)
        if quarantine_threshold := os.getenv("SCBE_GOV_QUARANTINE_THRESHOLD"):
            config.governance.quarantine_threshold = float(quarantine_threshold)

        # GeoSeal settings
        if interior_threshold := os.getenv("SCBE_GEOSEAL_INTERIOR_THRESHOLD"):
            config.geoseal.interior_threshold = float(interior_threshold)

        # Audit settings
        config.audit.enabled = os.getenv("SCBE_AUDIT_ENABLED", "true").lower() == "true"
        config.audit.log_level = os.getenv("SCBE_LOG_LEVEL", "INFO")
        config.audit.redact_sensitive = os.getenv("SCBE_REDACT_SENSITIVE", "true").lower() == "true"

        # Apply environment-specific defaults
        if environment == Environment.DEVELOPMENT:
            config.audit.include_stack_traces = True
            config.audit.redact_sensitive = False
        elif environment == Environment.STAGING:
            config.audit.include_stack_traces = True

        return config

    @classmethod
    def from_file(cls, path: str) -> ProductionConfig:
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProductionConfig:
        """Load configuration from dictionary."""
        config = cls()

        if "environment" in data:
            config.environment = Environment(data["environment"])

        if "pqc" in data:
            pqc_data = data["pqc"]
            config.pqc = PQCConfig(
                kem_level=pqc_data.get("kem_level", 768),
                dsa_level=pqc_data.get("dsa_level", 65),
                hybrid_mode=pqc_data.get("hybrid_mode", True),
                fallback_enabled=pqc_data.get("fallback_enabled", True),
            )

        if "governance" in data:
            gov_data = data["governance"]
            config.governance = GovernanceConfig(
                allow_threshold=gov_data.get("allow_threshold", 0.20),
                quarantine_threshold=gov_data.get("quarantine_threshold", 0.40),
                deny_threshold=gov_data.get("deny_threshold", 0.60),
                snap_threshold=gov_data.get("snap_threshold", 0.80),
                alpha=gov_data.get("alpha", 10.0),
                beta=gov_data.get("beta", 0.5),
            )

        if "geoseal" in data:
            geo_data = data["geoseal"]
            config.geoseal = GeoSealConfig(
                dimension=geo_data.get("dimension", 6),
                interior_threshold=geo_data.get("interior_threshold", 0.5),
                gamma=geo_data.get("gamma", 2.0),
            )

        if "quasicrystal" in data:
            qc_data = data["quasicrystal"]
            config.quasicrystal = QuasicrystalConfig(
                acceptance_radius=qc_data.get("acceptance_radius", 10.0),
                min_crystallinity=qc_data.get("min_crystallinity", 0.3),
            )

        if "audit" in data:
            audit_data = data["audit"]
            config.audit = AuditConfig(
                enabled=audit_data.get("enabled", True),
                log_level=audit_data.get("log_level", "INFO"),
                log_format=audit_data.get("log_format", "json"),
                redact_sensitive=audit_data.get("redact_sensitive", True),
            )

        # Feature flags
        config.enable_cymatic_resonance = data.get("enable_cymatic_resonance", True)
        config.enable_hal_attention = data.get("enable_hal_attention", True)
        config.enable_roundtable_consensus = data.get("enable_roundtable_consensus", True)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration to dictionary."""
        return {
            "environment": self.environment.value,
            "service_name": self.service_name,
            "service_version": self.service_version,
            "pqc": {
                "kem_level": self.pqc.kem_level,
                "dsa_level": self.pqc.dsa_level,
                "hybrid_mode": self.pqc.hybrid_mode,
                "fallback_enabled": self.pqc.fallback_enabled,
            },
            "governance": {
                "allow_threshold": self.governance.allow_threshold,
                "quarantine_threshold": self.governance.quarantine_threshold,
                "deny_threshold": self.governance.deny_threshold,
                "snap_threshold": self.governance.snap_threshold,
                "alpha": self.governance.alpha,
                "beta": self.governance.beta,
            },
            "geoseal": {
                "dimension": self.geoseal.dimension,
                "interior_threshold": self.geoseal.interior_threshold,
                "gamma": self.geoseal.gamma,
            },
            "quasicrystal": {
                "acceptance_radius": self.quasicrystal.acceptance_radius,
                "min_crystallinity": self.quasicrystal.min_crystallinity,
            },
            "audit": {
                "enabled": self.audit.enabled,
                "log_level": self.audit.log_level,
                "log_format": self.audit.log_format,
                "redact_sensitive": self.audit.redact_sensitive,
            },
            "enable_cymatic_resonance": self.enable_cymatic_resonance,
            "enable_hal_attention": self.enable_hal_attention,
            "enable_roundtable_consensus": self.enable_roundtable_consensus,
        }

    def validate(self) -> None:
        """Validate configuration values."""
        # PQC validation
        valid_kem_levels = [512, 768, 1024]
        if self.pqc.kem_level not in valid_kem_levels:
            raise ValueError(f"Invalid KEM level: {self.pqc.kem_level}. Must be one of {valid_kem_levels}")

        valid_dsa_levels = [44, 65, 87]
        if self.pqc.dsa_level not in valid_dsa_levels:
            raise ValueError(f"Invalid DSA level: {self.pqc.dsa_level}. Must be one of {valid_dsa_levels}")

        # Governance validation
        if not 0 < self.governance.allow_threshold < self.governance.quarantine_threshold:
            raise ValueError("allow_threshold must be less than quarantine_threshold")
        if not self.governance.quarantine_threshold < self.governance.deny_threshold:
            raise ValueError("quarantine_threshold must be less than deny_threshold")

        # GeoSeal validation
        if self.geoseal.dimension < 2:
            raise ValueError("GeoSeal dimension must be at least 2")
        if self.geoseal.interior_threshold <= 0:
            raise ValueError("interior_threshold must be positive")


# Global configuration instance
_config: Optional[ProductionConfig] = None


def get_config() -> ProductionConfig:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = ProductionConfig.from_env()
        _config.validate()
    return _config


def set_config(config: ProductionConfig) -> None:
    """Set the global configuration instance."""
    global _config
    config.validate()
    _config = config
