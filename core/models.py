"""
core/models.py
Shared Pydantic data models used across all agents and the API.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# User Profile
# ---------------------------------------------------------------------------

class ActivityLevel(str, Enum):
    sedentary = "sedentary"
    lightly_active = "lightly_active"
    moderately_active = "moderately_active"
    very_active = "very_active"
    extra_active = "extra_active"


class HealthCondition(str, Enum):
    diabetes = "diabetes"
    hypertension = "hypertension"
    heart_disease = "heart_disease"
    celiac = "celiac"
    kidney_disease = "kidney_disease"
    obesity = "obesity"
    none = "none"


class UserProfile(BaseModel):
    user_id: str
    name: str = ""
    age: int = Field(ge=1, le=120)
    gender: str = "unspecified"
    weight_kg: float = Field(gt=0)
    height_cm: float = Field(gt=0)
    activity_level: ActivityLevel = ActivityLevel.moderately_active
    health_conditions: list[HealthCondition] = []
    allergies: list[str] = []
    dietary_preferences: list[str] = []          # e.g. ["vegetarian", "halal"]
    cultural_background: str = ""
    fitness_goal: str = "maintain_weight"        # lose_weight | gain_muscle | maintain_weight


# ---------------------------------------------------------------------------
# Food & Nutrition
# ---------------------------------------------------------------------------

class NutrientInfo(BaseModel):
    calories: float = 0.0
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    fiber_g: float = 0.0
    sugar_g: float = 0.0
    sodium_mg: float = 0.0
    potassium_mg: float = 0.0
    calcium_mg: float = 0.0
    iron_mg: float = 0.0
    vitamin_c_mg: float = 0.0
    vitamin_d_iu: float = 0.0
    vitamin_b12_ug: float = 0.0
    source: str = "estimated"


class FoodItem(BaseModel):
    name: str
    quantity_g: float = 100.0
    nutrients: NutrientInfo = Field(default_factory=NutrientInfo)


# ---------------------------------------------------------------------------
# Meal Logging
# ---------------------------------------------------------------------------

class MealType(str, Enum):
    breakfast = "breakfast"
    lunch = "lunch"
    dinner = "dinner"
    snack = "snack"


class MealLogEntry(BaseModel):
    user_id: str
    date: str                               # ISO format YYYY-MM-DD
    meal_type: MealType
    description: str                        # free-text, image OCR text, or voice transcript
    food_items: list[FoodItem] = []
    total_nutrients: NutrientInfo = Field(default_factory=NutrientInfo)
    ai_feedback: str = ""


# ---------------------------------------------------------------------------
# Agent Requests / Responses
# ---------------------------------------------------------------------------

class NutritionQueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    context: dict[str, Any] = {}


class DietPlanRequest(BaseModel):
    profile: UserProfile
    num_days: int = Field(default=7, ge=1, le=30)
    preferences: dict[str, Any] = {}


class MealLogRequest(BaseModel):
    user_id: str
    date: str
    meal_type: MealType
    description: str
    input_mode: str = "text"        # text | image_base64 | voice_base64


class HealthAdviceRequest(BaseModel):
    profile: UserProfile
    specific_concern: str = ""


class AgentResponse(BaseModel):
    agent: str
    status: str = "success"
    result: Any
    tokens_used: int = 0
    sources: list[str] = []


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class DailyNutritionSummary(BaseModel):
    user_id: str
    date: str
    meals: list[MealLogEntry] = []
    total_nutrients: NutrientInfo = Field(default_factory=NutrientInfo)
    goal_calories: float = 2000.0
    deficiencies: list[str] = []
    surpluses: list[str] = []
