"""
orchestrator.py
Multi-Agent Orchestrator
────────────────────────
Routes incoming requests to the appropriate agent(s) and can invoke
compound workflows that span multiple agents in sequence.
"""
from __future__ import annotations

import logging
from typing import Any

from core.models import (
    AgentResponse,
    DietPlanRequest,
    HealthAdviceRequest,
    MealLogRequest,
    NutritionQueryRequest,
    UserProfile,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-agent dispatch
# ---------------------------------------------------------------------------

def query_nutrition_knowledge(request: NutritionQueryRequest) -> AgentResponse:
    """Route to Nutrition Knowledge Agent (RAG + USDA)."""
    from agents.nutrition_knowledge_agent import run as _run, seed_knowledge_base
    seed_knowledge_base()
    return _run(query=request.query, user_context=request.context)


def get_diet_plan(request: DietPlanRequest) -> AgentResponse:
    """Route to Diet Recommendation Agent."""
    from agents.diet_recommendation_agent import run as _run
    return _run(profile=request.profile, num_days=request.num_days, preferences=request.preferences)


def get_health_advice(request: HealthAdviceRequest) -> AgentResponse:
    """Route to Health Advisory Agent."""
    from agents.health_advisory_agent import run as _run
    return _run(profile=request.profile, specific_concern=request.specific_concern)


def log_meal(request: MealLogRequest, profile: UserProfile | None = None) -> AgentResponse:
    """Route to Food Log & Feedback Agent."""
    from agents.food_log_feedback_agent import run as _run
    return _run(
        user_id=request.user_id,
        date_str=request.date,
        meal_type=request.meal_type,
        description=request.description,
        input_mode=request.input_mode,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Compound workflow: Full Onboarding Assessment
# ---------------------------------------------------------------------------

def full_onboarding(profile: UserProfile) -> dict[str, Any]:
    """
    Compound workflow triggered on new user signup.
    1. Generate 7-day personalised meal plan.
    2. Generate preventive health advice.
    3. Seed knowledge base in background.

    Returns combined results from both agents.
    """
    from agents.nutrition_knowledge_agent import seed_knowledge_base
    seed_knowledge_base()

    diet_response = get_diet_plan(
        DietPlanRequest(profile=profile, num_days=7)
    )
    health_response = get_health_advice(
        HealthAdviceRequest(profile=profile)
    )

    return {
        "diet_plan": diet_response.model_dump(),
        "health_advice": health_response.model_dump(),
        "summary": (
            f"Onboarding complete for {profile.name or 'user'}. "
            f"Daily calorie target: {diet_response.result.get('daily_targets', {}).get('calories', '—')} kcal."
        ),
    }
