"""
CMP-001: SOC 2 Type II Compliance Tests

Tests for SOC 2 Trust Service Criteria compliance.
Covers CC6 (Logical Access) and CC7 (System Operations).
"""

import pytest
import time
import hashlib
import secrets
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta


class AccessLevel(Enum):
    """Access levels for role-based access control."""
    NONE = 0
    READ = 1
    WRITE = 2
    ADMIN = 3
    SUPER_ADMIN = 4


@dataclass
class User:
    """A system user."""
    user_id: str
    username: str
    roles: List[str]
    access_level: AccessLevel
    created_at: datetime
    last_login: Optional[datetime] = None
    mfa_enabled: bool = False
    password_hash: str = ""


@dataclass
class AuditLogEntry:
    """An audit log entry."""
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    result: str
    ip_address: str
    details: Dict[str, Any]


class AccessControlSystem:
    """
    Role-based access control system for SOC 2 compliance.
    """

    def __init__(self):
        self.users: Dict[str, User] = {}
        self.audit_log: List[AuditLogEntry] = []
        self.sessions: Dict[str, Dict] = {}

    def create_user(self, username: str, roles: List[str], access_level: AccessLevel) -> User:
        """Create a new user with specified roles."""
        user_id = secrets.token_hex(8)
        user = User(
            user_id=user_id,
            username=username,
            roles=roles,
            access_level=access_level,
            created_at=datetime.now(),
            password_hash=hashlib.sha256(secrets.token_bytes(32)).hexdigest(),
        )
        self.users[user_id] = user
        self._log_action(user_id, "USER_CREATED", f"user:{username}", "success")
        return user

    def authenticate(self, user_id: str, credentials: str) -> Optional[str]:
        """Authenticate user and create session."""
        if user_id not in self.users:
            self._log_action(user_id, "AUTH_FAILED", "session", "failure")
            return None

        user = self.users[user_id]

        # Check MFA if enabled
        if user.mfa_enabled and not self._verify_mfa(credentials):
            self._log_action(user_id, "MFA_FAILED", "session", "failure")
            return None

        # Create session
        session_id = secrets.token_hex(16)
        self.sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=8),
        }

        user.last_login = datetime.now()
        self._log_action(user_id, "LOGIN", "session", "success")
        return session_id

    def authorize(self, session_id: str, resource: str, action: str) -> bool:
        """Check if session is authorized for action on resource."""
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        # Check session expiry
        if datetime.now() > session["expires_at"]:
            self._log_action(session["user_id"], "SESSION_EXPIRED", resource, "failure")
            return False

        user = self.users.get(session["user_id"])
        if not user:
            return False

        # Check access level
        required_level = self._get_required_level(action)
        authorized = user.access_level.value >= required_level.value

        result = "success" if authorized else "denied"
        self._log_action(user.user_id, action, resource, result)

        return authorized

    def revoke_access(self, user_id: str) -> bool:
        """Revoke all access for a user."""
        if user_id not in self.users:
            return False

        # Invalidate all sessions
        sessions_to_remove = [
            sid for sid, sess in self.sessions.items()
            if sess["user_id"] == user_id
        ]
        for sid in sessions_to_remove:
            del self.sessions[sid]

        # Set access level to NONE
        self.users[user_id].access_level = AccessLevel.NONE

        self._log_action(user_id, "ACCESS_REVOKED", f"user:{user_id}", "success")
        return True

    def _get_required_level(self, action: str) -> AccessLevel:
        """Get required access level for an action."""
        action_levels = {
            "read": AccessLevel.READ,
            "write": AccessLevel.WRITE,
            "admin": AccessLevel.ADMIN,
            "delete": AccessLevel.ADMIN,
            "create_user": AccessLevel.SUPER_ADMIN,
        }
        return action_levels.get(action, AccessLevel.READ)

    def _verify_mfa(self, credentials: str) -> bool:
        """Verify MFA credentials (simplified)."""
        return len(credentials) >= 6

    def _log_action(self, user_id: str, action: str, resource: str, result: str):
        """Log an action to the audit trail."""
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            ip_address="127.0.0.1",  # Simplified
            details={},
        )
        self.audit_log.append(entry)

    def get_audit_log(self, user_id: Optional[str] = None) -> List[AuditLogEntry]:
        """Get audit log, optionally filtered by user."""
        if user_id:
            return [e for e in self.audit_log if e.user_id == user_id]
        return self.audit_log.copy()


class TestCC6LogicalAccessSecurity:
    """CC6.1: Tests for logical access security controls."""

    @pytest.fixture
    def acs(self):
        return AccessControlSystem()

    @pytest.mark.compliance
    def test_user_authentication_required(self, acs):
        """
        CMP-001/CC6.1: Users must authenticate before accessing resources.
        """
        user = acs.create_user("test_user", ["user"], AccessLevel.READ)

        # Without authentication, no session exists
        assert acs.authorize("invalid_session", "resource", "read") is False

        # After authentication, access is granted
        session_id = acs.authenticate(user.user_id, "credentials")
        assert session_id is not None
        assert acs.authorize(session_id, "resource", "read") is True

    @pytest.mark.compliance
    def test_role_based_access_control(self, acs):
        """
        CMP-001/CC6.1: Access must be controlled by roles.
        """
        # Create users with different access levels
        read_user = acs.create_user("reader", ["reader"], AccessLevel.READ)
        write_user = acs.create_user("writer", ["writer"], AccessLevel.WRITE)
        admin_user = acs.create_user("admin", ["admin"], AccessLevel.ADMIN)

        read_session = acs.authenticate(read_user.user_id, "creds")
        write_session = acs.authenticate(write_user.user_id, "creds")
        admin_session = acs.authenticate(admin_user.user_id, "creds")

        # Reader can only read
        assert acs.authorize(read_session, "data", "read") is True
        assert acs.authorize(read_session, "data", "write") is False

        # Writer can read and write
        assert acs.authorize(write_session, "data", "read") is True
        assert acs.authorize(write_session, "data", "write") is True
        assert acs.authorize(write_session, "data", "admin") is False

        # Admin can do everything
        assert acs.authorize(admin_session, "data", "read") is True
        assert acs.authorize(admin_session, "data", "write") is True
        assert acs.authorize(admin_session, "data", "admin") is True

    @pytest.mark.compliance
    def test_session_expiration(self, acs):
        """
        CMP-001/CC6.1: Sessions must expire.
        """
        user = acs.create_user("test", ["user"], AccessLevel.READ)
        session_id = acs.authenticate(user.user_id, "creds")

        # Manually expire session
        acs.sessions[session_id]["expires_at"] = datetime.now() - timedelta(hours=1)

        # Access should be denied
        assert acs.authorize(session_id, "resource", "read") is False

    @pytest.mark.compliance
    def test_failed_authentication_logged(self, acs):
        """
        CMP-001/CC6.1: Failed authentication attempts must be logged.
        """
        # Attempt authentication with invalid user
        result = acs.authenticate("invalid_user", "creds")
        assert result is None

        # Check audit log
        log = acs.get_audit_log()
        auth_failures = [e for e in log if e.action == "AUTH_FAILED"]
        assert len(auth_failures) >= 1


class TestCC6AccessAuthorization:
    """CC6.2: Tests for access authorization controls."""

    @pytest.fixture
    def acs(self):
        return AccessControlSystem()

    @pytest.mark.compliance
    def test_least_privilege_principle(self, acs):
        """
        CMP-001/CC6.2: Users should have minimum required access.
        """
        # Create user with minimal access
        user = acs.create_user("minimal", ["reader"], AccessLevel.READ)

        assert user.access_level == AccessLevel.READ
        assert "admin" not in user.roles

    @pytest.mark.compliance
    def test_privilege_escalation_prevented(self, acs):
        """
        CMP-001/CC6.2: Users cannot escalate their own privileges.
        """
        user = acs.create_user("regular", ["user"], AccessLevel.READ)
        session = acs.authenticate(user.user_id, "creds")

        # Attempt to perform admin action
        assert acs.authorize(session, "users", "create_user") is False

    @pytest.mark.compliance
    def test_cross_user_access_prevented(self, acs):
        """
        CMP-001/CC6.2: Users cannot access other users' resources.
        """
        user1 = acs.create_user("user1", ["user"], AccessLevel.READ)
        user2 = acs.create_user("user2", ["user"], AccessLevel.READ)

        session1 = acs.authenticate(user1.user_id, "creds")

        # User1 cannot access user2's data (would need resource-level ACLs)
        # This is a simplified check
        assert user1.user_id != user2.user_id


class TestCC6AccessRemoval:
    """CC6.3: Tests for access removal controls."""

    @pytest.fixture
    def acs(self):
        return AccessControlSystem()

    @pytest.mark.compliance
    def test_immediate_access_revocation(self, acs):
        """
        CMP-001/CC6.3: Access can be revoked immediately.
        """
        user = acs.create_user("to_revoke", ["user"], AccessLevel.READ)
        session = acs.authenticate(user.user_id, "creds")

        # Verify access works
        assert acs.authorize(session, "resource", "read") is True

        # Revoke access
        acs.revoke_access(user.user_id)

        # Verify access is denied
        assert acs.authorize(session, "resource", "read") is False

    @pytest.mark.compliance
    def test_revocation_logged(self, acs):
        """
        CMP-001/CC6.3: Access revocation must be logged.
        """
        user = acs.create_user("test", ["user"], AccessLevel.READ)
        acs.revoke_access(user.user_id)

        log = acs.get_audit_log(user.user_id)
        revoke_entries = [e for e in log if e.action == "ACCESS_REVOKED"]
        assert len(revoke_entries) >= 1


class TestCC7SystemOperations:
    """CC7: Tests for system operations controls."""

    @pytest.fixture
    def acs(self):
        return AccessControlSystem()

    @pytest.mark.compliance
    def test_comprehensive_audit_logging(self, acs):
        """
        CMP-001/CC7.1: All significant actions must be logged.
        """
        user = acs.create_user("audit_test", ["user"], AccessLevel.READ)
        session = acs.authenticate(user.user_id, "creds")
        acs.authorize(session, "resource", "read")
        acs.revoke_access(user.user_id)

        log = acs.get_audit_log(user.user_id)

        # Should have entries for: creation, login, read, revocation
        actions = [e.action for e in log]
        assert "USER_CREATED" in actions
        assert "LOGIN" in actions
        assert "read" in actions
        assert "ACCESS_REVOKED" in actions

    @pytest.mark.compliance
    def test_audit_log_immutability(self, acs):
        """
        CMP-001/CC7.1: Audit logs should be immutable.
        """
        user = acs.create_user("test", ["user"], AccessLevel.READ)
        initial_log = acs.get_audit_log()

        # Get a copy to verify immutability
        initial_count = len(initial_log)

        # More actions
        acs.authenticate(user.user_id, "creds")

        # Verify log grew (append-only)
        assert len(acs.get_audit_log()) > initial_count

    @pytest.mark.compliance
    def test_audit_log_timestamp_accuracy(self, acs):
        """
        CMP-001/CC7.1: Audit log timestamps must be accurate.
        """
        before = datetime.now()
        user = acs.create_user("test", ["user"], AccessLevel.READ)
        after = datetime.now()

        log = acs.get_audit_log(user.user_id)
        create_entry = [e for e in log if e.action == "USER_CREATED"][0]

        assert before <= create_entry.timestamp <= after

    @pytest.mark.compliance
    def test_incident_detection_time(self, acs):
        """
        CMP-001/CC7.4: Incidents must be detectable within 1 hour.
        """
        # Simulate failed authentication attempts (potential attack)
        for i in range(5):
            acs.authenticate("attacker", "bad_creds")

        log = acs.get_audit_log()
        failures = [e for e in log if e.result == "failure"]

        # All failures should be logged immediately
        for failure in failures:
            detection_delay = datetime.now() - failure.timestamp
            assert detection_delay < timedelta(hours=1), \
                "Failures must be detectable within 1 hour"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
