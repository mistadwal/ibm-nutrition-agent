"""
agents/diet_recommendation_agent.py

Diet Recommendation Agent
─────────────────────────
Responsibilities:
  • Generate personalised multi-day meal plans based on user profile.
  • Respect allergies, cultural preferences, health conditions, and fitness goals.
  • Calculate estimated daily calorie and macro targets using standard formulas.
  • Return structured meal plans (breakfast / lunch / dinner / snacks).

Architecture:
  UserProfile  →  Mifflin-St Jeor BMR  →  TDEE + goal adjustment
               →  IBM Granite LLM (structured meal plan generation)
"""
from __future__ import annotations

import logging
from typing import Any

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from core.llm import get_llm
from core.models import AgentResponse, UserProfile, ActivityLevel, HealthCondition

logger = logging.getLogger(__name__)
AGENT_NAME = "diet_recommendation_agent"

# ---------------------------------------------------------------------------
# Calorie & macro calculation
# ---------------------------------------------------------------------------

_ACTIVITY_MULTIPLIERS = {
    ActivityLevel.sedentary: 1.2,
    ActivityLevel.lightly_active: 1.375,
    ActivityLevel.moderately_active: 1.55,
    ActivityLevel.very_active: 1.725,
    ActivityLevel.extra_active: 1.9,
}


def calculate_tdee(profile: UserProfile) -> float:
    """Mifflin-St Jeor BMR × activity factor."""
    w, h, a = profile.weight_kg, profile.height_cm, profile.age
    if profile.gender.lower() in ("male", "m"):
        bmr = 10 * w + 6.25 * h - 5 * a + 5
    else:
        bmr = 10 * w + 6.25 * h - 5 * a - 161

    tdee = bmr * _ACTIVITY_MULTIPLIERS.get(profile.activity_level, 1.55)

    # Adjust for goal
    goal = profile.fitness_goal.lower()
    if "lose" in goal:
        tdee -= 500
    elif "gain" in goal or "muscle" in goal:
        tdee += 300

    return round(max(tdee, 1200), 1)   # never below 1200


def calculate_macros(calories: float, health_conditions: list[HealthCondition]) -> dict[str, float]:
    """Return recommended macro grams based on calorie target and conditions."""
    # Default split: 50% carbs, 25% protein, 25% fat
    carb_pct, prot_pct, fat_pct = 0.50, 0.25, 0.25

    if HealthCondition.diabetes in health_conditions:
        carb_pct, prot_pct, fat_pct = 0.40, 0.30, 0.30
    elif HealthCondition.heart_disease in health_conditions:
        carb_pct, prot_pct, fat_pct = 0.55, 0.20, 0.25
    elif HealthCondition.kidney_disease in health_conditions:
        carb_pct, prot_pct, fat_pct = 0.55, 0.10, 0.35

    return {
        "carbs_g": round(calories * carb_pct / 4, 1),
        "protein_g": round(calories * prot_pct / 4, 1),
        "fat_g": round(calories * fat_pct / 9, 1),
    }


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_DIET_PLAN_PROMPT = PromptTemplate(
    input_variables=[
        "name", "age", "gender", "weight_kg", "height_cm",
        "activity_level", "health_conditions", "allergies",
        "dietary_preferences", "cultural_background", "fitness_goal",
        "target_calories", "target_carbs_g", "target_protein_g", "target_fat_g",
        "num_days",
    ],
    template="""You are an expert registered dietitian and nutritionist powered by IBM Granite.
Create a detailed, personalised {num_days}-day meal plan for the following individual.

=== PATIENT PROFILE ===
Name: {name}
Age: {age} | Gender: {gender}
Weight: {weight_kg} kg | Height: {height_cm} cm
Activity Level: {activity_level}
Fitness Goal: {fitness_goal}
Health Conditions: {health_conditions}
Allergies / Intolerances: {allergies}
Dietary Preferences: {dietary_preferences}
Cultural Background: {cultural_background}

=== DAILY NUTRITION TARGETS ===
Calories: {target_calories} kcal
Carbohydrates: {target_carbs_g} g
Protein: {target_protein_g} g
Fat: {target_fat_g} g

=== MEAL PLAN INSTRUCTIONS ===
For each day (Day 1 through Day {num_days}), provide:
  • Breakfast: [meal name + key ingredients + estimated kcal]
  • Morning Snack: [optional, if calories allow]
  • Lunch: [meal name + key ingredients + estimated kcal]
  • Afternoon Snack: [optional]
  • Dinner: [meal name + key ingredients + estimated kcal]
  • Hydration tip for the day

Rules:
1. Strictly respect allergies and dietary preferences.
2. For diabetes: keep GI low, distribute carbs evenly across meals.
3. For hypertension / heart disease: minimise sodium (<1500 mg/day), maximise potassium.
4. For kidney disease: limit phosphorus, potassium, and protein.
5. Incorporate cultural food preferences where possible.
6. Include at least 5 servings of fruit/vegetables per day.
7. Vary the meals — do not repeat the same meal on consecutive days.
8. After the meal plan, add a brief "Nutrition Highlights" section summarising key benefits.

Meal Plan:""",
)


# ---------------------------------------------------------------------------
# Public agent interface
# ---------------------------------------------------------------------------

def run(profile: UserProfile, num_days: int = 7, preferences: dict[str, Any] | None = None) -> AgentResponse:
    """Generate a personalised multi-day meal plan."""
    target_calories = calculate_tdee(profile)
    macros = calculate_macros(target_calories, profile.health_conditions)

    llm = get_llm(temperature=0.5, max_new_tokens=2048)
    chain = LLMChain(llm=llm, prompt=_DIET_PLAN_PROMPT)

    result = chain.invoke({
        "name": profile.name or "Patient",
        "age": profile.age,
        "gender": profile.gender,
        "weight_kg": profile.weight_kg,
        "height_cm": profile.height_cm,
        "activity_level": profile.activity_level.value,
        "health_conditions": ", ".join(c.value for c in profile.health_conditions) or "None",
        "allergies": ", ".join(profile.allergies) or "None",
        "dietary_preferences": ", ".join(profile.dietary_preferences) or "None",
        "cultural_background": profile.cultural_background or "Not specified",
        "fitness_goal": profile.fitness_goal,
        "target_calories": target_calories,
        "target_carbs_g": macros["carbs_g"],
        "target_protein_g": macros["protein_g"],
        "target_fat_g": macros["fat_g"],
        "num_days": num_days,
    })

    meal_plan_text: str = result.get("text", "")

    return AgentResponse(
        agent=AGENT_NAME,
        result={
            "meal_plan": meal_plan_text,
            "daily_targets": {
                "calories": target_calories,
                **macros,
            },
            "num_days": num_days,
        },
        sources=["Mifflin-St Jeor formula", "IBM Granite LLM"],
    )
