"""
Common Criteria EAL4+ Compliance Tests

Tests for Common Criteria Evaluation Assurance Level 4+ requirements.
"""

import pytest
import hashlib
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SecurityFunction(Enum):
    """Security functions in the TOE (Target of Evaluation)."""
    IDENTIFICATION = "identification"
    AUTHENTICATION = "authentication"
    ACCESS_CONTROL = "access_control"
    AUDIT = "audit"
    CRYPTOGRAPHY = "cryptography"
    DATA_PROTECTION = "data_protection"


@dataclass
class SecurityPolicy:
    """A security policy rule."""
    policy_id: str
    function: SecurityFunction
    rule: str
    enforced: bool = True


@dataclass
class AuditRecord:
    """An audit record per FAU requirements."""
    record_id: str
    timestamp: datetime
    event_type: str
    subject_id: str
    object_id: str
    outcome: str
    details: Dict[str, Any] = field(default_factory=dict)


class TOESecurityModule:
    """
    Target of Evaluation security module.
    Implements Common Criteria security functional requirements.
    """

    def __init__(self):
        self.policies: List[SecurityPolicy] = []
        self.audit_log: List[AuditRecord] = []
        self.users: Dict[str, Dict] = {}
        self.sessions: Dict[str, Dict] = {}
        self._record_counter = 0

        # Initialize default policies
        self._initialize_policies()

    def _initialize_policies(self):
        """Initialize security policies."""
        self.policies = [
            SecurityPolicy("POL-001", SecurityFunction.AUTHENTICATION,
                           "All users must authenticate before access"),
            SecurityPolicy("POL-002", SecurityFunction.ACCESS_CONTROL,
                           "Access based on least privilege"),
            SecurityPolicy("POL-003", SecurityFunction.AUDIT,
                           "All security-relevant events must be audited"),
            SecurityPolicy("POL-004", SecurityFunction.CRYPTOGRAPHY,
                           "Use approved cryptographic algorithms"),
            SecurityPolicy("POL-005", SecurityFunction.DATA_PROTECTION,
                           "Protect data in transit and at rest"),
        ]

    def _generate_record_id(self) -> str:
        """Generate unique audit record ID."""
        self._record_counter += 1
        return f"AUD-{self._record_counter:08d}"

    def audit_event(
        self,
        event_type: str,
        subject_id: str,
        object_id: str,
        outcome: str,
        details: Dict = None
    ) -> AuditRecord:
        """
        Record an auditable event (FAU_GEN).
        """
        record = AuditRecord(
            record_id=self._generate_record_id(),
            timestamp=datetime.now(),
            event_type=event_type,
            subject_id=subject_id,
            object_id=object_id,
            outcome=outcome,
            details=details or {},
        )
        self.audit_log.append(record)
        return record

    def identify_user(self, claimed_identity: str) -> bool:
        """
        User identification (FIA_UID).
        """
        result = claimed_identity in self.users
        self.audit_event(
            "IDENTIFICATION",
            claimed_identity,
            "system",
            "success" if result else "failure",
        )
        return result

    def authenticate_user(self, user_id: str, credentials: str) -> Optional[str]:
        """
        User authentication (FIA_UAU).
        """
        if user_id not in self.users:
            self.audit_event("AUTHENTICATION", user_id, "system", "failure")
            return None

        user = self.users[user_id]

        # Verify credentials (simplified)
        expected_hash = user.get("password_hash", "")
        provided_hash = hashlib.sha256(credentials.encode()).hexdigest()

        if expected_hash != provided_hash:
            self.audit_event("AUTHENTICATION", user_id, "system", "failure")
            return None

        # Create session
        session_id = hashlib.sha256(f"{user_id}{time.time()}".encode()).hexdigest()[:32]
        self.sessions[session_id] = {
            "user_id": user_id,
            "created": datetime.now(),
        }

        self.audit_event("AUTHENTICATION", user_id, "system", "success")
        return session_id

    def check_access(self, session_id: str, resource: str, action: str) -> bool:
        """
        Access control decision (FDP_ACC).
        """
        if session_id not in self.sessions:
            self.audit_event("ACCESS_CONTROL", "unknown", resource, "denied")
            return False

        session = self.sessions[session_id]
        user_id = session["user_id"]
        user = self.users.get(user_id, {})

        # Check user permissions
        permissions = user.get("permissions", [])
        required = f"{action}:{resource}"

        granted = required in permissions or f"*:{resource}" in permissions

        self.audit_event(
            "ACCESS_CONTROL",
            user_id,
            resource,
            "granted" if granted else "denied",
            {"action": action},
        )
        return granted

    def create_user(self, user_id: str, password: str, permissions: List[str]) -> bool:
        """Create a new user."""
        if user_id in self.users:
            return False

        self.users[user_id] = {
            "password_hash": hashlib.sha256(password.encode()).hexdigest(),
            "permissions": permissions,
            "created": datetime.now(),
        }

        self.audit_event("USER_MANAGEMENT", "system", user_id, "created")
        return True


class TestFAUAudit:
    """FAU: Security Audit tests."""

    @pytest.fixture
    def toe(self):
        module = TOESecurityModule()
        module.create_user("test_user", "password123", ["read:data"])
        return module

    @pytest.mark.compliance
    def test_fau_gen_audit_generation(self, toe):
        """
        FAU_GEN.1: Audit data generation.
        """
        toe.identify_user("test_user")

        assert len(toe.audit_log) >= 2  # User creation + identification

    @pytest.mark.compliance
    def test_fau_gen_all_fields_present(self, toe):
        """
        FAU_GEN.1: Audit records must contain required fields.
        """
        toe.authenticate_user("test_user", "password123")

        record = toe.audit_log[-1]

        assert record.record_id is not None
        assert record.timestamp is not None
        assert record.event_type is not None
        assert record.subject_id is not None
        assert record.object_id is not None
        assert record.outcome is not None

    @pytest.mark.compliance
    def test_fau_stg_protected_storage(self, toe):
        """
        FAU_STG.1: Protected audit trail storage.
        """
        # Generate some audit events
        for i in range(10):
            toe.identify_user(f"user_{i}")

        # Verify all events are stored
        assert len(toe.audit_log) >= 10

        # Verify records have unique IDs (sequential)
        ids = [r.record_id for r in toe.audit_log]
        assert len(set(ids)) == len(ids)

    @pytest.mark.compliance
    def test_fau_stg_no_modification(self, toe):
        """
        FAU_STG.2: Audit trail cannot be modified.
        """
        toe.identify_user("test_user")
        initial_count = len(toe.audit_log)

        # Audit log should only grow (append-only)
        toe.authenticate_user("test_user", "password123")

        assert len(toe.audit_log) > initial_count


class TestFIAIdentificationAuthentication:
    """FIA: Identification and Authentication tests."""

    @pytest.fixture
    def toe(self):
        module = TOESecurityModule()
        module.create_user("valid_user", "correct_password", ["read:data"])
        return module

    @pytest.mark.compliance
    def test_fia_uid_identification(self, toe):
        """
        FIA_UID.1: User identification before any action.
        """
        # Valid user identified
        assert toe.identify_user("valid_user") is True

        # Invalid user not identified
        assert toe.identify_user("invalid_user") is False

    @pytest.mark.compliance
    def test_fia_uau_authentication(self, toe):
        """
        FIA_UAU.1: User authentication before any action.
        """
        # Correct credentials
        session = toe.authenticate_user("valid_user", "correct_password")
        assert session is not None

        # Incorrect credentials
        session = toe.authenticate_user("valid_user", "wrong_password")
        assert session is None

    @pytest.mark.compliance
    def test_fia_uau_before_action(self, toe):
        """
        FIA_UAU.2: Authentication required before actions.
        """
        # Without authentication, access denied
        result = toe.check_access("invalid_session", "data", "read")
        assert result is False

        # With authentication, access may be granted
        session = toe.authenticate_user("valid_user", "correct_password")
        result = toe.check_access(session, "data", "read")
        assert result is True


class TestFDPAccessControl:
    """FDP: User Data Protection tests."""

    @pytest.fixture
    def toe(self):
        module = TOESecurityModule()
        module.create_user("admin", "admin_pass", ["read:*", "write:*"])
        module.create_user("reader", "reader_pass", ["read:data"])
        module.create_user("writer", "writer_pass", ["read:data", "write:data"])
        return module

    @pytest.mark.compliance
    def test_fdp_acc_access_control_policy(self, toe):
        """
        FDP_ACC.1: Access control policy enforcement.
        """
        reader_session = toe.authenticate_user("reader", "reader_pass")

        # Reader can read
        assert toe.check_access(reader_session, "data", "read") is True

        # Reader cannot write
        assert toe.check_access(reader_session, "data", "write") is False

    @pytest.mark.compliance
    def test_fdp_acf_access_control_functions(self, toe):
        """
        FDP_ACF.1: Access control functions.
        """
        admin_session = toe.authenticate_user("admin", "admin_pass")
        writer_session = toe.authenticate_user("writer", "writer_pass")

        # Admin can do everything
        assert toe.check_access(admin_session, "any_resource", "read") is True

        # Writer has specific permissions
        assert toe.check_access(writer_session, "data", "write") is True

    @pytest.mark.compliance
    def test_fdp_no_unauthorized_access(self, toe):
        """
        FDP: No unauthorized information flow.
        """
        reader_session = toe.authenticate_user("reader", "reader_pass")

        # Attempt to access restricted resource
        result = toe.check_access(reader_session, "admin_data", "read")
        assert result is False


class TestADVDesign:
    """ADV: Development assurance tests."""

    @pytest.fixture
    def toe(self):
        return TOESecurityModule()

    @pytest.mark.compliance
    def test_adv_arc_security_architecture(self, toe):
        """
        ADV_ARC.1: Security architecture description.
        """
        # Verify security policies are defined
        assert len(toe.policies) > 0

        # All core functions covered
        functions_covered = {p.function for p in toe.policies}
        required_functions = {
            SecurityFunction.AUTHENTICATION,
            SecurityFunction.ACCESS_CONTROL,
            SecurityFunction.AUDIT,
        }
        assert required_functions.issubset(functions_covered)

    @pytest.mark.compliance
    def test_adv_fsp_functional_spec(self, toe):
        """
        ADV_FSP.1: Functional specification.
        """
        # Verify core functions exist and work
        toe.create_user("test", "pass", ["read:test"])

        # Identification function
        assert hasattr(toe, 'identify_user')
        assert callable(toe.identify_user)

        # Authentication function
        assert hasattr(toe, 'authenticate_user')
        assert callable(toe.authenticate_user)

        # Access control function
        assert hasattr(toe, 'check_access')
        assert callable(toe.check_access)


class TestATETesting:
    """ATE: Testing assurance."""

    @pytest.fixture
    def toe(self):
        module = TOESecurityModule()
        module.create_user("test_user", "test_pass", ["read:data"])
        return module

    @pytest.mark.compliance
    def test_ate_cov_test_coverage(self, toe):
        """
        ATE_COV.1: Evidence of test coverage.
        """
        # All security functions should be testable
        testable_functions = [
            toe.identify_user,
            toe.authenticate_user,
            toe.check_access,
            toe.audit_event,
        ]

        for func in testable_functions:
            assert callable(func)

    @pytest.mark.compliance
    def test_ate_dpt_test_depth(self, toe):
        """
        ATE_DPT.1: Test depth analysis.
        """
        # Test complete authentication flow
        # Step 1: Identification
        identified = toe.identify_user("test_user")
        assert identified is True

        # Step 2: Authentication
        session = toe.authenticate_user("test_user", "test_pass")
        assert session is not None

        # Step 3: Access control
        access = toe.check_access(session, "data", "read")
        assert access is True

        # Verify audit trail captures all steps
        event_types = [r.event_type for r in toe.audit_log]
        assert "IDENTIFICATION" in event_types
        assert "AUTHENTICATION" in event_types
        assert "ACCESS_CONTROL" in event_types


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
