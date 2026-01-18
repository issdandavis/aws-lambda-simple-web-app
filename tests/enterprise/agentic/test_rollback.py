"""
ACT-002: Rollback Capability Tests

Tests for code modification rollback functionality.
Target: Full state restoration capability.
"""

import pytest
import hashlib
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from copy import deepcopy


@dataclass
class FileState:
    """State of a file at a point in time."""
    path: str
    content: str
    checksum: str
    timestamp: datetime


@dataclass
class Checkpoint:
    """A system checkpoint for rollback."""
    checkpoint_id: str
    timestamp: datetime
    description: str
    files: Dict[str, FileState] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RollbackManager:
    """
    Manages checkpoints and rollback for code modifications.
    """

    def __init__(self, max_checkpoints: int = 100):
        self.checkpoints: List[Checkpoint] = []
        self.current_state: Dict[str, str] = {}  # path -> content
        self.max_checkpoints = max_checkpoints
        self._checkpoint_counter = 0

    def create_checkpoint(self, description: str = "") -> Checkpoint:
        """
        Create a checkpoint of current state.
        """
        self._checkpoint_counter += 1
        checkpoint_id = f"CP-{self._checkpoint_counter:06d}"

        files = {}
        for path, content in self.current_state.items():
            files[path] = FileState(
                path=path,
                content=content,
                checksum=hashlib.sha256(content.encode()).hexdigest(),
                timestamp=datetime.now(),
            )

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            description=description,
            files=files,
            metadata={"file_count": len(files)},
        )

        self.checkpoints.append(checkpoint)

        # Enforce max checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints.pop(0)

        return checkpoint

    def rollback_to(self, checkpoint_id: str) -> bool:
        """
        Rollback to a specific checkpoint.
        """
        # Find checkpoint
        checkpoint = None
        for cp in self.checkpoints:
            if cp.checkpoint_id == checkpoint_id:
                checkpoint = cp
                break

        if not checkpoint:
            return False

        # Restore state
        self.current_state = {}
        for path, file_state in checkpoint.files.items():
            self.current_state[path] = file_state.content

        return True

    def rollback_to_previous(self) -> bool:
        """
        Rollback to the previous checkpoint.
        """
        if len(self.checkpoints) < 2:
            return False

        # Get second-to-last checkpoint
        previous = self.checkpoints[-2]
        return self.rollback_to(previous.checkpoint_id)

    def modify_file(self, path: str, content: str):
        """
        Modify a file (simulated).
        """
        self.current_state[path] = content

    def get_file(self, path: str) -> Optional[str]:
        """
        Get current file content.
        """
        return self.current_state.get(path)

    def get_checkpoint_history(self) -> List[Dict[str, Any]]:
        """
        Get checkpoint history.
        """
        return [
            {
                "id": cp.checkpoint_id,
                "timestamp": cp.timestamp.isoformat(),
                "description": cp.description,
                "file_count": cp.metadata.get("file_count", 0),
            }
            for cp in self.checkpoints
        ]

    def verify_checkpoint(self, checkpoint_id: str) -> Dict[str, bool]:
        """
        Verify checkpoint integrity.
        """
        checkpoint = None
        for cp in self.checkpoints:
            if cp.checkpoint_id == checkpoint_id:
                checkpoint = cp
                break

        if not checkpoint:
            return {"valid": False, "error": "Checkpoint not found"}

        # Verify all file checksums
        results = {}
        for path, file_state in checkpoint.files.items():
            current_checksum = hashlib.sha256(file_state.content.encode()).hexdigest()
            results[path] = current_checksum == file_state.checksum

        return {"valid": all(results.values()), "files": results}


class TestCheckpointCreation:
    """Tests for checkpoint creation."""

    @pytest.fixture
    def manager(self):
        return RollbackManager()

    @pytest.mark.agentic
    def test_create_checkpoint(self, manager):
        """
        ACT-002: Create checkpoint captures current state.
        """
        manager.modify_file("test.py", "print('hello')")
        manager.modify_file("config.json", '{"key": "value"}')

        checkpoint = manager.create_checkpoint("Initial state")

        assert checkpoint is not None
        assert checkpoint.checkpoint_id.startswith("CP-")
        assert len(checkpoint.files) == 2
        assert "test.py" in checkpoint.files
        assert "config.json" in checkpoint.files

    @pytest.mark.agentic
    def test_checkpoint_captures_content(self, manager):
        """
        Checkpoint captures exact file content.
        """
        content = "def foo():\n    return 42"
        manager.modify_file("module.py", content)

        checkpoint = manager.create_checkpoint()

        assert checkpoint.files["module.py"].content == content

    @pytest.mark.agentic
    def test_checkpoint_has_checksum(self, manager):
        """
        Checkpoint includes content checksum.
        """
        manager.modify_file("test.py", "content")
        checkpoint = manager.create_checkpoint()

        expected_checksum = hashlib.sha256("content".encode()).hexdigest()
        assert checkpoint.files["test.py"].checksum == expected_checksum


class TestRollback:
    """Tests for rollback functionality."""

    @pytest.fixture
    def manager(self):
        return RollbackManager()

    @pytest.mark.agentic
    def test_rollback_restores_state(self, manager):
        """
        ACT-002: Rollback fully restores previous state.
        """
        # Initial state
        manager.modify_file("app.py", "version = 1")
        cp1 = manager.create_checkpoint("v1")

        # Modified state
        manager.modify_file("app.py", "version = 2")
        manager.create_checkpoint("v2")

        # Verify current state is v2
        assert manager.get_file("app.py") == "version = 2"

        # Rollback to v1
        success = manager.rollback_to(cp1.checkpoint_id)
        assert success is True

        # Verify state restored
        assert manager.get_file("app.py") == "version = 1"

    @pytest.mark.agentic
    def test_rollback_restores_all_files(self, manager):
        """
        Rollback restores all files, not just one.
        """
        # Initial state
        manager.modify_file("file1.py", "original1")
        manager.modify_file("file2.py", "original2")
        cp1 = manager.create_checkpoint()

        # Modify both files
        manager.modify_file("file1.py", "modified1")
        manager.modify_file("file2.py", "modified2")
        manager.create_checkpoint()

        # Rollback
        manager.rollback_to(cp1.checkpoint_id)

        # Both files should be restored
        assert manager.get_file("file1.py") == "original1"
        assert manager.get_file("file2.py") == "original2"

    @pytest.mark.agentic
    def test_rollback_to_previous(self, manager):
        """
        Rollback to previous checkpoint works.
        """
        manager.modify_file("test.py", "v1")
        manager.create_checkpoint()

        manager.modify_file("test.py", "v2")
        manager.create_checkpoint()

        manager.modify_file("test.py", "v3")
        manager.create_checkpoint()

        # Rollback to previous (v2)
        success = manager.rollback_to_previous()
        assert success is True
        assert manager.get_file("test.py") == "v2"

    @pytest.mark.agentic
    def test_rollback_invalid_checkpoint(self, manager):
        """
        Rollback to non-existent checkpoint fails gracefully.
        """
        manager.modify_file("test.py", "content")
        manager.create_checkpoint()

        success = manager.rollback_to("INVALID-ID")
        assert success is False

        # State should be unchanged
        assert manager.get_file("test.py") == "content"


class TestCheckpointIntegrity:
    """Tests for checkpoint integrity verification."""

    @pytest.fixture
    def manager(self):
        return RollbackManager()

    @pytest.mark.agentic
    def test_verify_checkpoint_valid(self, manager):
        """
        Valid checkpoint passes verification.
        """
        manager.modify_file("test.py", "content")
        checkpoint = manager.create_checkpoint()

        result = manager.verify_checkpoint(checkpoint.checkpoint_id)
        assert result["valid"] is True

    @pytest.mark.agentic
    def test_checkpoint_history(self, manager):
        """
        Checkpoint history is maintained.
        """
        manager.modify_file("test.py", "v1")
        manager.create_checkpoint("Version 1")

        manager.modify_file("test.py", "v2")
        manager.create_checkpoint("Version 2")

        history = manager.get_checkpoint_history()

        assert len(history) == 2
        assert history[0]["description"] == "Version 1"
        assert history[1]["description"] == "Version 2"


class TestCheckpointLimits:
    """Tests for checkpoint limits."""

    @pytest.mark.agentic
    def test_max_checkpoints_enforced(self):
        """
        Maximum checkpoint limit is enforced.
        """
        manager = RollbackManager(max_checkpoints=5)

        # Create more than max checkpoints
        for i in range(10):
            manager.modify_file("test.py", f"version {i}")
            manager.create_checkpoint(f"v{i}")

        # Should only keep last 5
        assert len(manager.checkpoints) == 5

        # Oldest should be v5
        assert manager.checkpoints[0].description == "v5"

    @pytest.mark.agentic
    def test_rollback_after_limit(self):
        """
        Rollback works correctly after checkpoint limit.
        """
        manager = RollbackManager(max_checkpoints=3)

        for i in range(5):
            manager.modify_file("test.py", f"v{i}")
            manager.create_checkpoint(f"v{i}")

        # Can still rollback to recent checkpoints
        success = manager.rollback_to_previous()
        assert success is True
        assert manager.get_file("test.py") == "v3"


class TestMultiFileRollback:
    """Tests for multi-file rollback scenarios."""

    @pytest.fixture
    def manager(self):
        return RollbackManager()

    @pytest.mark.agentic
    def test_rollback_with_new_files(self, manager):
        """
        Rollback handles files added after checkpoint.
        """
        manager.modify_file("original.py", "original content")
        cp1 = manager.create_checkpoint()

        # Add new file
        manager.modify_file("new_file.py", "new content")
        manager.create_checkpoint()

        # Rollback
        manager.rollback_to(cp1.checkpoint_id)

        # Original file should exist, new file should not
        assert manager.get_file("original.py") == "original content"
        assert manager.get_file("new_file.py") is None

    @pytest.mark.agentic
    def test_rollback_with_deleted_files(self, manager):
        """
        Rollback restores deleted files.
        """
        manager.modify_file("keep.py", "keep")
        manager.modify_file("delete.py", "delete me")
        cp1 = manager.create_checkpoint()

        # "Delete" file by removing from state
        del manager.current_state["delete.py"]
        manager.create_checkpoint()

        # Verify file is gone
        assert manager.get_file("delete.py") is None

        # Rollback
        manager.rollback_to(cp1.checkpoint_id)

        # File should be restored
        assert manager.get_file("delete.py") == "delete me"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
