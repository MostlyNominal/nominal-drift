"""Tests for tool_router module."""
import pytest
from nominal_drift.core.tool_router import route_intent, list_routes


class TestRouteIntent:
    """Test route_intent function."""

    def test_route_diffusion_run(self):
        """route_intent('diffusion_run', '') returns string containing 'diffusion_engine'."""
        result = route_intent("diffusion_run", "")
        assert result is not None
        assert isinstance(result, str)
        assert "diffusion_engine" in result

    def test_route_sensitization_run(self):
        """route_intent('sensitization_run', '') contains 'sensitization_model'."""
        result = route_intent("sensitization_run", "")
        assert result is not None
        assert "sensitization_model" in result

    def test_route_mechanism_animation(self):
        """route_intent('mechanism_animation', '') contains 'animator'."""
        result = route_intent("mechanism_animation", "")
        assert result is not None
        assert "animator" in result

    def test_route_unknown_intent_returns_none(self):
        """route_intent('unknown_intent', '') returns None."""
        result = route_intent("unknown_intent_xyz", "")
        assert result is None

    def test_route_report_request(self):
        """route_intent('report_request', '') contains 'report_builder'."""
        result = route_intent("report_request", "")
        assert result is not None
        assert "report_builder" in result

    def test_route_dataset_import(self):
        """route_intent('dataset_import', '') contains 'adapters'."""
        result = route_intent("dataset_import", "")
        assert result is not None
        assert "adapters" in result

    def test_route_literature_entry(self):
        """route_intent('literature_entry', '') contains 'literature'."""
        result = route_intent("literature_entry", "")
        assert result is not None
        assert "literature" in result

    def test_route_comparison(self):
        """route_intent('comparison', '') is not None."""
        result = route_intent("comparison", "")
        assert result is not None

    def test_route_sensitization_experiment(self):
        """route_intent('sensitization_experiment', '') is not None."""
        result = route_intent("sensitization_experiment", "")
        assert result is not None

    def test_route_result_format(self):
        """route_intent result contains routing indicator '🔀'."""
        result = route_intent("diffusion_run", "")
        assert result is not None
        assert "🔀" in result

    def test_route_result_contains_intent_name(self):
        """route_intent result contains the intent name or related module."""
        result = route_intent("diffusion_run", "")
        assert "diffusion" in result.lower()


class TestListRoutes:
    """Test list_routes function."""

    def test_list_routes_returns_dict(self):
        """list_routes() returns dict with 'diffusion_run' key."""
        routes = list_routes()
        assert isinstance(routes, dict)
        assert "diffusion_run" in routes

    def test_list_routes_has_entries(self):
        """list_routes() has at least 5 entries."""
        routes = list_routes()
        assert len(routes) >= 5

    def test_list_routes_all_known_intents(self):
        """route_intent returns string (not None) for all known intents."""
        routes = list_routes()
        for intent in routes.keys():
            result = route_intent(intent, "")
            assert result is not None
            assert isinstance(result, str)

    def test_list_routes_values_non_empty(self):
        """All route values in list_routes are non-empty strings."""
        routes = list_routes()
        for module in routes.values():
            assert isinstance(module, str)
            assert len(module) > 0

    def test_list_routes_contains_expected_intents(self):
        """list_routes contains expected template type intents."""
        routes = list_routes()
        expected = [
            "diffusion_run",
            "sensitization_run",
            "mechanism_animation",
            "dataset_import",
            "report_request",
        ]
        for intent in expected:
            assert intent in routes
