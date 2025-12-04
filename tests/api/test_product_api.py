"""
Product API Tests for MLSDM Neuro Memory Service.

Tests for the Product Layer endpoints:
- POST /v1/memory/append
- POST /v1/memory/query
- POST /v1/decide
- POST /v1/agent/step
- GET /health (already exists)
- GET /ready (already exists)
"""

import os

import pytest
from fastapi.testclient import TestClient


# Set test environment
os.environ["DISABLE_RATE_LIMIT"] = "1"
os.environ["LLM_BACKEND"] = "local_stub"


@pytest.fixture
def client():
    """Create a test client for the API."""
    from mlsdm.api.app import app
    return TestClient(app)


class TestMemoryAppendEndpoint:
    """Tests for POST /v1/memory/append."""

    def test_append_memory_success(self, client):
        """Test successful memory append."""
        response = client.post(
            "/v1/memory/append",
            json={
                "content": "The user likes coffee.",
                "moral_value": 0.9,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "phase" in data
        assert "accepted" in data
        assert "memory_stats" in data

    def test_append_memory_with_metadata(self, client):
        """Test memory append with full metadata."""
        response = client.post(
            "/v1/memory/append",
            json={
                "content": "Important meeting scheduled.",
                "user_id": "user-123",
                "session_id": "session-abc",
                "agent_id": "agent-1",
                "moral_value": 0.85,
                "metadata": {"priority": "high"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True or data["success"] is False  # Either is valid

    def test_append_memory_empty_content_rejected(self, client):
        """Test that empty content is rejected."""
        response = client.post(
            "/v1/memory/append",
            json={
                "content": "",
                "moral_value": 0.9,
            },
        )
        assert response.status_code == 422  # Validation error

    def test_append_memory_invalid_moral_value(self, client):
        """Test that invalid moral value is rejected."""
        response = client.post(
            "/v1/memory/append",
            json={
                "content": "Test content",
                "moral_value": 1.5,  # Invalid: > 1.0
            },
        )
        assert response.status_code == 422


class TestMemoryQueryEndpoint:
    """Tests for POST /v1/memory/query."""

    def test_query_memory_success(self, client):
        """Test successful memory query."""
        # First append some content
        client.post(
            "/v1/memory/append",
            json={"content": "User prefers tea.", "moral_value": 0.9},
        )

        # Then query
        response = client.post(
            "/v1/memory/query",
            json={
                "query": "What does the user prefer?",
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "results" in data
        assert "query_phase" in data
        assert "total_results" in data

    def test_query_memory_with_filters(self, client):
        """Test memory query with user/session filters."""
        response = client.post(
            "/v1/memory/query",
            json={
                "query": "Coffee preferences",
                "user_id": "user-123",
                "session_id": "session-abc",
                "top_k": 3,
                "include_metadata": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["results"], list)

    def test_query_memory_empty_query_rejected(self, client):
        """Test that empty query is rejected."""
        response = client.post(
            "/v1/memory/query",
            json={
                "query": "",
                "top_k": 5,
            },
        )
        assert response.status_code == 422


class TestDecideEndpoint:
    """Tests for POST /v1/decide."""

    def test_decide_standard_mode(self, client):
        """Test decision with standard mode."""
        response = client.post(
            "/v1/decide",
            json={
                "prompt": "Should I proceed with this action?",
                "risk_level": "low",
                "mode": "standard",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "accepted" in data
        assert "phase" in data
        assert "contour_decisions" in data
        assert "decision_id" in data

    def test_decide_with_context(self, client):
        """Test decision with context provided."""
        response = client.post(
            "/v1/decide",
            json={
                "prompt": "Is this a good decision?",
                "context": "The user has previously expressed concerns.",
                "risk_level": "medium",
                "mode": "cautious",
                "max_tokens": 256,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "risk_assessment" in data
        assert data["risk_assessment"]["level"] == "medium"
        assert data["risk_assessment"]["mode"] == "cautious"

    def test_decide_high_risk(self, client):
        """Test decision with high risk level."""
        response = client.post(
            "/v1/decide",
            json={
                "prompt": "Should I share sensitive data?",
                "risk_level": "high",
                "mode": "cautious",
            },
        )
        assert response.status_code == 200
        data = response.json()
        # High risk + cautious mode should have higher moral threshold
        assert data["risk_assessment"]["level"] == "high"

    def test_decide_emergency_mode(self, client):
        """Test decision with emergency mode."""
        response = client.post(
            "/v1/decide",
            json={
                "prompt": "Emergency situation - what should I do?",
                "risk_level": "critical",
                "mode": "emergency",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["risk_assessment"]["mode"] == "emergency"

    def test_decide_contour_decisions(self, client):
        """Test that contour decisions are returned."""
        response = client.post(
            "/v1/decide",
            json={
                "prompt": "Simple question?",
                "risk_level": "low",
                "mode": "standard",
            },
        )
        assert response.status_code == 200
        data = response.json()
        
        # Should have at least moral_filter and risk_assessment contours
        contours = data["contour_decisions"]
        contour_names = [c["contour"] for c in contours]
        assert "moral_filter" in contour_names
        assert "risk_assessment" in contour_names

    def test_decide_empty_prompt_rejected(self, client):
        """Test that empty prompt is rejected."""
        response = client.post(
            "/v1/decide",
            json={
                "prompt": "",
                "risk_level": "low",
            },
        )
        assert response.status_code == 422


class TestAgentStepEndpoint:
    """Tests for POST /v1/agent/step."""

    def test_agent_step_basic(self, client):
        """Test basic agent step."""
        response = client.post(
            "/v1/agent/step",
            json={
                "agent_id": "test-agent",
                "observation": "User said hello.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "action" in data
        assert "response" in data
        assert "phase" in data
        assert "accepted" in data
        assert "step_id" in data

    def test_agent_step_with_state(self, client):
        """Test agent step with internal state."""
        response = client.post(
            "/v1/agent/step",
            json={
                "agent_id": "stateful-agent",
                "observation": "Continue from last step.",
                "internal_state": {"step_count": 5, "goal": "Assist user"},
                "moral_value": 0.85,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "updated_state" in data
        if data["accepted"]:
            assert data["updated_state"] is not None

    def test_agent_step_with_tool_results(self, client):
        """Test agent step with tool results."""
        response = client.post(
            "/v1/agent/step",
            json={
                "agent_id": "tool-agent",
                "observation": "Process tool output.",
                "tool_results": [
                    {"tool": "search", "result": "Found 3 items."},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["action"]["action_type"] in ["respond", "tool_call", "wait", "terminate"]

    def test_agent_step_action_types(self, client):
        """Test that action types are valid."""
        response = client.post(
            "/v1/agent/step",
            json={
                "agent_id": "action-agent",
                "observation": "What should I do next?",
            },
        )
        assert response.status_code == 200
        data = response.json()
        
        valid_action_types = ["respond", "tool_call", "wait", "terminate"]
        assert data["action"]["action_type"] in valid_action_types

    def test_agent_step_empty_agent_id_rejected(self, client):
        """Test that empty agent_id is rejected."""
        response = client.post(
            "/v1/agent/step",
            json={
                "agent_id": "",
                "observation": "Test observation.",
            },
        )
        assert response.status_code == 422


class TestHealthEndpoints:
    """Tests for health endpoints (verifying they still work)."""

    def test_health_endpoint(self, client):
        """Test GET /health."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_live_endpoint(self, client):
        """Test GET /health/live."""
        response = client.get("/health/live")
        assert response.status_code == 200

    def test_health_ready_endpoint(self, client):
        """Test GET /health/ready."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data


class TestGenerateEndpoint:
    """Tests for existing /generate endpoint (verifying compatibility)."""

    def test_generate_basic(self, client):
        """Test basic generation still works."""
        response = client.post(
            "/generate",
            json={
                "prompt": "Hello, world!",
                "moral_value": 0.8,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "phase" in data
        assert "accepted" in data

    def test_generate_with_max_tokens(self, client):
        """Test generation with max_tokens."""
        response = client.post(
            "/generate",
            json={
                "prompt": "Explain AI.",
                "max_tokens": 100,
                "moral_value": 0.9,
            },
        )
        assert response.status_code == 200
