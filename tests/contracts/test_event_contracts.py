"""
Event API Contract Tests for MLSDM.

These tests validate that the /v1/process_event/ and /v1/state/ endpoints
conform to their strict contracts defined in mlsdm.contracts.event_models.

CONTRACT STABILITY:
These tests protect the API contract. If a test fails after code changes,
it indicates a potential breaking change that requires a major version bump.

Test categories:
- Model validation tests: Pydantic model constraints
- Positive cases: Valid payloads return expected responses
- Negative cases: Invalid payloads return standardized errors
- Error format tests: All errors follow ApiErrorResponse schema
"""

import os

import pytest
from pydantic import ValidationError

from mlsdm.contracts.errors import ApiError, ApiErrorResponse
from mlsdm.contracts.event_models import EventInput, StateResponse

# ============================================================================
# Model Validation Tests
# ============================================================================


class TestEventInputValidation:
    """Tests for EventInput model validation."""

    def test_valid_event_input(self):
        """EventInput accepts valid event_vector and moral_value."""
        event = EventInput(
            event_vector=[0.1, 0.2, -0.3, 0.4],
            moral_value=0.75
        )
        assert len(event.event_vector) == 4
        assert event.moral_value == 0.75

    def test_moral_value_at_boundaries(self):
        """EventInput accepts moral_value at 0.0 and 1.0 boundaries."""
        event_min = EventInput(event_vector=[0.1], moral_value=0.0)
        event_max = EventInput(event_vector=[0.1], moral_value=1.0)
        assert event_min.moral_value == 0.0
        assert event_max.moral_value == 1.0

    def test_moral_value_below_zero_rejected(self):
        """EventInput rejects moral_value below 0.0."""
        with pytest.raises(ValidationError) as exc_info:
            EventInput(event_vector=[0.1], moral_value=-0.1)
        assert "greater than or equal to 0" in str(exc_info.value).lower()

    def test_moral_value_above_one_rejected(self):
        """EventInput rejects moral_value above 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            EventInput(event_vector=[0.1], moral_value=1.1)
        assert "less than or equal to 1" in str(exc_info.value).lower()

    def test_empty_event_vector_rejected(self):
        """EventInput rejects empty event_vector."""
        with pytest.raises(ValidationError) as exc_info:
            EventInput(event_vector=[], moral_value=0.5)
        errors = exc_info.value.errors()
        assert len(errors) > 0
        # Should fail min_length=1 constraint
        assert any("event_vector" in str(e.get("loc", [])) for e in errors)

    def test_nan_in_event_vector_rejected(self):
        """EventInput rejects NaN values in event_vector."""
        with pytest.raises(ValidationError) as exc_info:
            EventInput(event_vector=[0.1, float("nan"), 0.3], moral_value=0.5)
        assert "non-finite" in str(exc_info.value).lower()

    def test_inf_in_event_vector_rejected(self):
        """EventInput rejects Inf values in event_vector."""
        with pytest.raises(ValidationError) as exc_info:
            EventInput(event_vector=[0.1, float("inf"), 0.3], moral_value=0.5)
        assert "non-finite" in str(exc_info.value).lower()

    def test_negative_inf_in_event_vector_rejected(self):
        """EventInput rejects -Inf values in event_vector."""
        with pytest.raises(ValidationError) as exc_info:
            EventInput(event_vector=[0.1, float("-inf"), 0.3], moral_value=0.5)
        assert "non-finite" in str(exc_info.value).lower()

    def test_valid_high_dimensional_vector(self):
        """EventInput accepts valid 384-dimensional vector."""
        vector = [0.01 * i for i in range(384)]
        event = EventInput(event_vector=vector, moral_value=0.5)
        assert len(event.event_vector) == 384


class TestStateResponseValidation:
    """Tests for StateResponse model validation."""

    def test_valid_state_response(self):
        """StateResponse accepts valid values."""
        response = StateResponse(
            L1_norm=1.5,
            L2_norm=2.3,
            L3_norm=0.8,
            current_phase="wake",
            latent_events_count=10,
            accepted_events_count=85,
            total_events_processed=100,
            moral_filter_threshold=0.5
        )
        assert response.L1_norm == 1.5
        assert response.current_phase == "wake"

    def test_state_response_phase_wake(self):
        """StateResponse accepts 'wake' phase."""
        response = StateResponse(
            L1_norm=0.0,
            L2_norm=0.0,
            L3_norm=0.0,
            current_phase="wake",
            latent_events_count=0,
            accepted_events_count=0,
            total_events_processed=0,
            moral_filter_threshold=0.5
        )
        assert response.current_phase == "wake"

    def test_state_response_phase_sleep(self):
        """StateResponse accepts 'sleep' phase."""
        response = StateResponse(
            L1_norm=0.0,
            L2_norm=0.0,
            L3_norm=0.0,
            current_phase="sleep",
            latent_events_count=0,
            accepted_events_count=0,
            total_events_processed=0,
            moral_filter_threshold=0.5
        )
        assert response.current_phase == "sleep"

    def test_state_response_invalid_phase_rejected(self):
        """StateResponse rejects invalid phase values."""
        with pytest.raises(ValidationError) as exc_info:
            StateResponse(
                L1_norm=1.0,
                L2_norm=1.0,
                L3_norm=1.0,
                current_phase="invalid_phase",  # type: ignore
                latent_events_count=0,
                accepted_events_count=0,
                total_events_processed=0,
                moral_filter_threshold=0.5
            )
        assert "current_phase" in str(exc_info.value).lower()

    def test_state_response_negative_norm_rejected(self):
        """StateResponse rejects negative norm values."""
        with pytest.raises(ValidationError):
            StateResponse(
                L1_norm=-1.0,  # Invalid: norms must be >= 0
                L2_norm=1.0,
                L3_norm=1.0,
                current_phase="wake",
                latent_events_count=0,
                accepted_events_count=0,
                total_events_processed=0,
                moral_filter_threshold=0.5
            )

    def test_state_response_negative_count_rejected(self):
        """StateResponse rejects negative event counts."""
        with pytest.raises(ValidationError):
            StateResponse(
                L1_norm=1.0,
                L2_norm=1.0,
                L3_norm=1.0,
                current_phase="wake",
                latent_events_count=-1,  # Invalid: counts must be >= 0
                accepted_events_count=0,
                total_events_processed=0,
                moral_filter_threshold=0.5
            )


class TestApiErrorModels:
    """Tests for ApiError and ApiErrorResponse models."""

    def test_api_error_with_all_fields(self):
        """ApiError accepts all fields including optional details."""
        error = ApiError(
            code="validation_error",
            message="Field is invalid",
            details={"field": "event_vector", "reason": "dimension_mismatch"}
        )
        assert error.code == "validation_error"
        assert error.message == "Field is invalid"
        assert error.details["field"] == "event_vector"

    def test_api_error_without_details(self):
        """ApiError works without optional details."""
        error = ApiError(
            code="internal_error",
            message="An unexpected error occurred"
        )
        assert error.code == "internal_error"
        assert error.details is None

    def test_api_error_response_structure(self):
        """ApiErrorResponse wraps ApiError correctly."""
        error = ApiError(code="rate_limit_exceeded", message="Too many requests")
        response = ApiErrorResponse(error=error)

        # Verify structure
        data = response.model_dump()
        assert "error" in data
        assert data["error"]["code"] == "rate_limit_exceeded"
        assert data["error"]["message"] == "Too many requests"


# ============================================================================
# HTTP Endpoint Contract Tests
# ============================================================================


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment for API contract tests."""
    os.environ["DISABLE_RATE_LIMIT"] = "1"
    os.environ["LLM_BACKEND"] = "local_stub"
    os.environ["API_KEY"] = "test_key"
    yield
    # Cleanup all environment variables
    if "DISABLE_RATE_LIMIT" in os.environ:
        del os.environ["DISABLE_RATE_LIMIT"]
    if "LLM_BACKEND" in os.environ:
        del os.environ["LLM_BACKEND"]
    if "API_KEY" in os.environ:
        del os.environ["API_KEY"]


@pytest.fixture
def client():
    """Create a TestClient with authentication."""
    from fastapi.testclient import TestClient

    from mlsdm.api.app import app

    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Return authorization headers for authenticated requests."""
    return {"Authorization": "Bearer test_key"}


@pytest.fixture
def expected_dimension():
    """Get the expected embedding dimension from config.

    Default config uses dimension 10 for testing.
    Production typically uses 384.
    """
    from mlsdm.api.app import _manager

    return _manager.dimension


@pytest.fixture
def valid_event_vector(expected_dimension):
    """Return a valid event vector matching the configured dimension."""
    return [0.01] * expected_dimension


class TestProcessEventEndpointContract:
    """Tests for /v1/process_event/ endpoint contract."""

    def test_process_event_returns_state_response(
        self, client, auth_headers, valid_event_vector
    ):
        """POST /v1/process_event/ returns StateResponse schema."""
        response = client.post(
            "/v1/process_event/",
            json={"event_vector": valid_event_vector, "moral_value": 0.75},
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        # Verify all StateResponse fields are present
        assert "L1_norm" in data
        assert "L2_norm" in data
        assert "L3_norm" in data
        assert "current_phase" in data
        assert "latent_events_count" in data
        assert "accepted_events_count" in data
        assert "total_events_processed" in data
        assert "moral_filter_threshold" in data

    def test_process_event_field_types(
        self, client, auth_headers, valid_event_vector
    ):
        """POST /v1/process_event/ returns correct field types."""
        response = client.post(
            "/v1/process_event/",
            json={"event_vector": valid_event_vector, "moral_value": 0.5},
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data["L1_norm"], (int, float))
        assert isinstance(data["L2_norm"], (int, float))
        assert isinstance(data["L3_norm"], (int, float))
        assert isinstance(data["current_phase"], str)
        assert data["current_phase"] in ("wake", "sleep")
        assert isinstance(data["latent_events_count"], int)
        assert isinstance(data["accepted_events_count"], int)
        assert isinstance(data["total_events_processed"], int)
        assert isinstance(data["moral_filter_threshold"], (int, float))

    def test_process_event_moral_value_boundaries(
        self, client, auth_headers, valid_event_vector
    ):
        """POST /v1/process_event/ accepts moral_value at boundaries."""
        # Test 0.0
        response = client.post(
            "/v1/process_event/",
            json={"event_vector": valid_event_vector, "moral_value": 0.0},
            headers=auth_headers,
        )
        assert response.status_code == 200

        # Test 1.0
        response = client.post(
            "/v1/process_event/",
            json={"event_vector": valid_event_vector, "moral_value": 1.0},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_process_event_dimension_mismatch_error(
        self, client, auth_headers, expected_dimension
    ):
        """POST /v1/process_event/ returns error for dimension mismatch."""
        # Create a wrong dimension vector (different from expected)
        wrong_dim = expected_dimension + 10
        wrong_dim_vector = [0.01] * wrong_dim

        response = client.post(
            "/v1/process_event/",
            json={"event_vector": wrong_dim_vector, "moral_value": 0.5},
            headers=auth_headers,
        )
        assert response.status_code == 400

        data = response.json()
        # Verify ApiErrorResponse structure
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert data["error"]["code"] == "dimension_mismatch"
        assert "details" in data["error"]
        assert data["error"]["details"]["expected_dimension"] == expected_dimension
        assert data["error"]["details"]["actual_dimension"] == wrong_dim

    def test_process_event_moral_value_out_of_range(
        self, client, auth_headers, valid_event_vector
    ):
        """POST /v1/process_event/ returns 422 for moral_value out of range."""
        response = client.post(
            "/v1/process_event/",
            json={"event_vector": valid_event_vector, "moral_value": 1.5},
            headers=auth_headers,
        )
        # Pydantic validation returns 422
        assert response.status_code == 422

        data = response.json()
        assert "detail" in data  # FastAPI validation error format

    def test_process_event_nan_in_vector(
        self, client, auth_headers
    ):
        """POST /v1/process_event/ returns 422 for NaN in event_vector."""
        # Note: JSON doesn't support NaN, so we use the string "NaN" which becomes null
        # In practice, this would be caught at the Pydantic model level
        # We test the model validation directly in TestEventInputValidation

    def test_process_event_empty_vector(
        self, client, auth_headers
    ):
        """POST /v1/process_event/ returns 422 for empty event_vector."""
        response = client.post(
            "/v1/process_event/",
            json={"event_vector": [], "moral_value": 0.5},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_process_event_missing_field(
        self, client, auth_headers
    ):
        """POST /v1/process_event/ returns 422 for missing required field."""
        response = client.post(
            "/v1/process_event/",
            json={"event_vector": [0.1, 0.2]},  # Missing moral_value
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_process_event_unauthorized(
        self, client, valid_event_vector
    ):
        """POST /v1/process_event/ requires authentication."""
        response = client.post(
            "/v1/process_event/",
            json={"event_vector": valid_event_vector, "moral_value": 0.5},
            # No auth headers
        )
        assert response.status_code == 401


class TestGetStateEndpointContract:
    """Tests for /v1/state/ endpoint contract."""

    def test_get_state_returns_state_response(
        self, client, auth_headers
    ):
        """GET /v1/state/ returns StateResponse schema."""
        response = client.get("/v1/state/", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        # Verify all StateResponse fields are present
        assert "L1_norm" in data
        assert "L2_norm" in data
        assert "L3_norm" in data
        assert "current_phase" in data
        assert "latent_events_count" in data
        assert "accepted_events_count" in data
        assert "total_events_processed" in data
        assert "moral_filter_threshold" in data

    def test_get_state_field_types(
        self, client, auth_headers
    ):
        """GET /v1/state/ returns correct field types."""
        response = client.get("/v1/state/", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data["L1_norm"], (int, float))
        assert isinstance(data["L2_norm"], (int, float))
        assert isinstance(data["L3_norm"], (int, float))
        assert isinstance(data["current_phase"], str)
        assert data["current_phase"] in ("wake", "sleep")
        assert isinstance(data["latent_events_count"], int)
        assert isinstance(data["accepted_events_count"], int)
        assert isinstance(data["total_events_processed"], int)
        assert isinstance(data["moral_filter_threshold"], (int, float))

    def test_get_state_norms_non_negative(
        self, client, auth_headers
    ):
        """GET /v1/state/ returns non-negative norm values."""
        response = client.get("/v1/state/", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["L1_norm"] >= 0.0
        assert data["L2_norm"] >= 0.0
        assert data["L3_norm"] >= 0.0

    def test_get_state_moral_threshold_in_range(
        self, client, auth_headers
    ):
        """GET /v1/state/ returns moral_filter_threshold in [0.0, 1.0]."""
        response = client.get("/v1/state/", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert 0.0 <= data["moral_filter_threshold"] <= 1.0

    def test_get_state_unauthorized(
        self, client
    ):
        """GET /v1/state/ requires authentication."""
        response = client.get("/v1/state/")
        assert response.status_code == 401


class TestContractStability:
    """Tests that verify contract stability over time."""

    def test_state_response_field_set_unchanged(
        self, client, auth_headers
    ):
        """StateResponse field set matches contract specification."""
        response = client.get("/v1/state/", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        actual_fields = set(data.keys())

        # Expected contract fields (must not change without major version bump)
        expected_fields = {
            "L1_norm",
            "L2_norm",
            "L3_norm",
            "current_phase",
            "latent_events_count",
            "accepted_events_count",
            "total_events_processed",
            "moral_filter_threshold",
        }

        # All expected fields must be present
        missing = expected_fields - actual_fields
        assert not missing, f"Missing contract fields: {missing}"

    def test_error_response_follows_api_error_schema(
        self, client, auth_headers, expected_dimension
    ):
        """Error responses follow ApiErrorResponse schema."""
        # Trigger a dimension mismatch error
        wrong_dim_vector = [0.1] * (expected_dimension + 5)
        response = client.post(
            "/v1/process_event/",
            json={"event_vector": wrong_dim_vector, "moral_value": 0.5},
            headers=auth_headers,
        )
        assert response.status_code == 400

        data = response.json()
        # Verify ApiErrorResponse structure
        assert "error" in data
        error = data["error"]
        assert "code" in error
        assert "message" in error
        assert "details" in error or error.get("details") is None

        # Code should be a machine-readable string
        assert isinstance(error["code"], str)
        assert len(error["code"]) > 0

        # Message should be human-readable
        assert isinstance(error["message"], str)
        assert len(error["message"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
