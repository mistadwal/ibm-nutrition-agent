"""
tests/test_agents.py
Unit tests for the Nutrition Agent system.
Run with: pytest tests/ -v
"""
from __future__ import annotations

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock

from core.models import (
    ActivityLevel,
    DietPlanRequest,
    HealthAdviceRequest,
    HealthCondition,
    MealLogRequest,
    MealType,
    NutrientInfo,
    NutritionQueryRequest,
    UserProfile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_profile():
    return UserProfile(
        user_id="test_user_001",
        name="Test User",
        age=35,
        gender="female",
        weight_kg=68.0,
        height_cm=165.0,
        activity_level=ActivityLevel.moderately_active,
        health_conditions=[HealthCondition.diabetes],
        allergies=["peanuts"],
        dietary_preferences=["vegetarian"],
        cultural_background="South Asian",
        fitness_goal="lose_weight",
    )


# ---------------------------------------------------------------------------
# Core models tests
# ---------------------------------------------------------------------------

class TestUserProfile:
    def test_valid_profile_creation(self, sample_profile):
        assert sample_profile.user_id == "test_user_001"
        assert sample_profile.age == 35
        assert HealthCondition.diabetes in sample_profile.health_conditions

    def test_profile_defaults(self):
        profile = UserProfile(user_id="u1", age=30, weight_kg=70, height_cm=170)
        assert profile.activity_level == ActivityLevel.moderately_active
        assert profile.health_conditions == []
        assert profile.allergies == []

    def test_age_validation(self):
        with pytest.raises(Exception):
            UserProfile(user_id="u1", age=0, weight_kg=70, height_cm=170)

    def test_weight_validation(self):
        with pytest.raises(Exception):
            UserProfile(user_id="u1", age=25, weight_kg=-5, height_cm=170)


class TestNutrientInfo:
    def test_default_zeros(self):
        n = NutrientInfo()
        assert n.calories == 0.0
        assert n.protein_g == 0.0

    def test_with_values(self):
        n = NutrientInfo(calories=350.5, protein_g=25.0, carbs_g=45.0, fat_g=12.0)
        assert n.calories == 350.5
        assert n.source == "estimated"


# ---------------------------------------------------------------------------
# Diet Recommendation Agent tests
# ---------------------------------------------------------------------------

class TestDietRecommendationAgent:
    def test_tdee_male(self):
        from agents.diet_recommendation_agent import calculate_tdee
        profile = UserProfile(
            user_id="u1", name="M", age=30, gender="male",
            weight_kg=80, height_cm=180,
            activity_level=ActivityLevel.moderately_active,
            fitness_goal="maintain_weight",
        )
        tdee = calculate_tdee(profile)
        # Expected ~2600 kcal for a 30yo male, 80kg, 180cm, moderately active
        assert 2400 < tdee < 2900

    def test_tdee_female(self):
        from agents.diet_recommendation_agent import calculate_tdee
        profile = UserProfile(
            user_id="u1", name="F", age=28, gender="female",
            weight_kg=60, height_cm=162,
            activity_level=ActivityLevel.moderately_active,
            fitness_goal="maintain_weight",
        )
        tdee = calculate_tdee(profile)
        assert 1800 < tdee < 2200

    def test_tdee_lose_weight(self):
        from agents.diet_recommendation_agent import calculate_tdee
        profile_maintain = UserProfile(user_id="u1", age=30, gender="male", weight_kg=80, height_cm=180, fitness_goal="maintain_weight")
        profile_lose = UserProfile(user_id="u1", age=30, gender="male", weight_kg=80, height_cm=180, fitness_goal="lose_weight")
        assert calculate_tdee(profile_lose) == calculate_tdee(profile_maintain) - 500

    def test_tdee_minimum_1200(self):
        from agents.diet_recommendation_agent import calculate_tdee
        # Very small person + sedentary + lose_weight might go below 1200
        profile = UserProfile(user_id="u1", age=80, gender="female", weight_kg=40, height_cm=150,
                              activity_level=ActivityLevel.sedentary, fitness_goal="lose_weight")
        assert calculate_tdee(profile) >= 1200

    def test_macro_diabetes_reduced_carbs(self):
        from agents.diet_recommendation_agent import calculate_macros
        macros_healthy = calculate_macros(2000, [])
        macros_diabetes = calculate_macros(2000, [HealthCondition.diabetes])
        assert macros_diabetes["carbs_g"] < macros_healthy["carbs_g"]
        assert macros_diabetes["protein_g"] > macros_healthy["protein_g"]

    def test_macro_kidney_disease_low_protein(self):
        from agents.diet_recommendation_agent import calculate_macros
        macros_kidney = calculate_macros(2000, [HealthCondition.kidney_disease])
        macros_normal = calculate_macros(2000, [])
        assert macros_kidney["protein_g"] < macros_normal["protein_g"]


# ---------------------------------------------------------------------------
# Food Log Agent tests
# ---------------------------------------------------------------------------

class TestFoodLogAgent:
    def test_aggregate_nutrients(self):
        from agents.food_log_feedback_agent import _aggregate_nutrients
        from core.models import FoodItem, NutrientInfo
        foods = [
            FoodItem(name="apple", quantity_g=150, nutrients=NutrientInfo(calories=78, protein_g=0.4, carbs_g=21, fat_g=0.2)),
            FoodItem(name="chicken breast", quantity_g=100, nutrients=NutrientInfo(calories=165, protein_g=31, carbs_g=0, fat_g=3.6)),
        ]
        total = _aggregate_nutrients(foods)
        assert round(total.calories, 1) == 243.0
        assert round(total.protein_g, 1) == 31.4

    def test_daily_summary_empty(self):
        from agents.food_log_feedback_agent import get_daily_summary
        summary = get_daily_summary("nonexistent_user_xyz", "2099-01-01")
        assert summary.total_nutrients.calories == 0.0
        assert summary.meals == []

    def test_get_daily_summary_returns_model(self):
        from agents.food_log_feedback_agent import get_daily_summary
        from core.models import DailyNutritionSummary
        result = get_daily_summary("test_user", "2025-01-01")
        assert isinstance(result, DailyNutritionSummary)


# ---------------------------------------------------------------------------
# Orchestrator tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestOrchestrator:
    @patch("agents.nutrition_knowledge_agent.seed_knowledge_base")
    @patch("agents.nutrition_knowledge_agent._build_rag_chain")
    def test_query_nutrition_knowledge(self, mock_chain_builder, mock_seed):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"result": "Brown rice has 216 kcal per 100g.", "source_documents": []}
        mock_chain_builder.return_value = mock_chain

        import orchestrator
        request = NutritionQueryRequest(query="nutritional value of brown rice")
        response = orchestrator.query_nutrition_knowledge(request)
        assert response.agent == "nutrition_knowledge_agent"
        assert response.status == "success"
        assert "result" in response.model_dump()

    @patch("agents.diet_recommendation_agent.get_llm")
    def test_diet_plan_returns_agent_response(self, mock_llm_fn, sample_profile):
        from core.models import AgentResponse
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"text": "Day 1: Breakfast — Oatmeal with berries..."}
        mock_llm_fn.return_value = mock_llm

        with patch("langchain.chains.LLMChain") as mock_lc:
            mock_lc.return_value = mock_chain
            import orchestrator
            request = DietPlanRequest(profile=sample_profile, num_days=3)
            response = orchestrator.get_diet_plan(request)
            assert response.agent == "diet_recommendation_agent"
            assert isinstance(response.result, dict)
            assert "daily_targets" in response.result


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_settings_loads(self):
        from core.config import get_settings, Settings
        settings = get_settings()
        assert isinstance(settings, Settings)
        assert settings.app_port > 0

    def test_default_port(self):
        from core.config import get_settings
        assert get_settings().app_port == 8000

    def test_granite_model_defaults(self):
        from core.config import get_settings
        s = get_settings()
        assert "granite" in s.granite_llm_model.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
