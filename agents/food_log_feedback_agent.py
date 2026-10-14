"""
agents/food_log_feedback_agent.py

Food Log & Feedback Agent
─────────────────────────
Responsibilities:
  • Accept meal descriptions via text, OCR'd image text, or voice transcript.
  • Parse food items and estimate nutritional values (USDA FDC API + LLM).
  • Compute daily totals and compare to user's calorie/macro targets.
  • Provide instant personalised feedback and suggestions.
  • Maintain an in-memory daily log (production: replace with DB).

Architecture:
  Raw input (text/image/voice)
    → text normalisation (OCR / speech-to-text)
    → LLM food extraction
    → USDA FDC nutrient lookup
    → aggregation
    → LLM feedback generation
"""
from __future__ import annotations

import base64
import io
import logging
from collections import defaultdict
from datetime import date
from typing import Any

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from agents.nutrition_knowledge_agent import lookup_food
from core.llm import get_llm
from core.models import (
    AgentResponse,
    DailyNutritionSummary,
    FoodItem,
    MealLogEntry,
    MealType,
    NutrientInfo,
    UserProfile,
)

logger = logging.getLogger(__name__)
AGENT_NAME = "food_log_feedback_agent"

# ---------------------------------------------------------------------------
# In-memory meal store  {user_id: {date_str: [MealLogEntry]}}
# ---------------------------------------------------------------------------
_meal_store: dict[str, dict[str, list[MealLogEntry]]] = defaultdict(lambda: defaultdict(list))

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_FOOD_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["meal_description"],
    template="""You are a nutrition analyst. Extract all food items and their approximate quantities 
from the meal description below. Return a JSON list only — no explanation, no markdown fences.

Format: [{"name": "food name", "quantity_g": 150}, ...]

If quantity is not specified, use a reasonable default serving size in grams.

Meal description: {meal_description}

JSON list:""",
)

_FEEDBACK_PROMPT = PromptTemplate(
    input_variables=[
        "user_name", "meal_type", "date",
        "foods_logged", "total_calories", "total_protein_g",
        "total_carbs_g", "total_fat_g", "total_fiber_g",
        "goal_calories", "goal_protein_g", "goal_carbs_g", "goal_fat_g",
        "health_conditions", "remaining_calories",
    ],
    template="""You are a friendly and knowledgeable nutrition coach powered by IBM Granite.

{user_name} logged their {meal_type} on {date}:
Foods: {foods_logged}

Nutritional breakdown:
  Calories: {total_calories} kcal  (Daily goal: {goal_calories} kcal | Remaining: {remaining_calories} kcal)
  Protein: {total_protein_g}g (Goal: {goal_protein_g}g)
  Carbohydrates: {total_carbs_g}g (Goal: {goal_carbs_g}g)
  Fat: {total_fat_g}g (Goal: {goal_fat_g}g)
  Fiber: {total_fiber_g}g

Health conditions: {health_conditions}

Provide brief, encouraging feedback (3–5 sentences):
1. Comment on this meal's nutritional quality.
2. Note any nutritional concern relative to their conditions/goals.
3. Suggest one specific improvement or addition for their next meal.

Keep the tone positive and practical. Feedback:""",
)

# ---------------------------------------------------------------------------
# Input pre-processing
# ---------------------------------------------------------------------------

def _preprocess_image(base64_image: str) -> str:
    """OCR an image (base64-encoded) to extract text. Returns extracted text."""
    try:
        import pytesseract
        from PIL import Image

        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        logger.info("OCR extracted %d characters from image.", len(text))
        return text.strip() or "Unable to extract text from image."
    except Exception as exc:
        logger.warning("OCR failed: %s", exc)
        return "Image received but OCR unavailable. Please describe the meal as text."


def _preprocess_voice(base64_audio: str) -> str:
    """Transcribe base64-encoded audio. Returns transcript text."""
    try:
        import speech_recognition as sr

        audio_bytes = base64.b64decode(base64_audio)
        audio_file = io.BytesIO(audio_bytes)
        recogniser = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recogniser.record(source)
        transcript = recogniser.recognize_google(audio_data)
        logger.info("Speech transcription: %s", transcript)
        return transcript
    except Exception as exc:
        logger.warning("Speech recognition failed: %s", exc)
        return "Audio received but transcription unavailable. Please describe the meal as text."


# ---------------------------------------------------------------------------
# Food parsing
# ---------------------------------------------------------------------------

def _extract_food_items_from_text(description: str) -> list[FoodItem]:
    """Use LLM to extract structured food items from free text, then enrich with USDA."""
    import json as _json

    llm = get_llm(temperature=0.1, max_new_tokens=512)
    chain = LLMChain(llm=llm, prompt=_FOOD_EXTRACTION_PROMPT)
    raw_output: str = chain.invoke({"meal_description": description}).get("text", "")

    # Parse JSON
    try:
        # Strip any accidental markdown fences
        clean = raw_output.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        items_raw: list[dict[str, Any]] = _json.loads(clean)
    except Exception:
        logger.warning("LLM returned non-JSON for food extraction: %s", raw_output[:200])
        # Fall back: treat whole description as a single item
        items_raw = [{"name": description[:80], "quantity_g": 200}]

    food_items: list[FoodItem] = []
    for item in items_raw:
        name = str(item.get("name", "unknown"))
        qty = float(item.get("quantity_g", 100))

        # Try USDA lookup
        fi = lookup_food(name)
        if fi:
            # Scale nutrients to actual quantity
            scale = qty / 100.0
            n = fi.nutrients
            fi = FoodItem(
                name=name,
                quantity_g=qty,
                nutrients=NutrientInfo(
                    calories=round(n.calories * scale, 1),
                    protein_g=round(n.protein_g * scale, 1),
                    carbs_g=round(n.carbs_g * scale, 1),
                    fat_g=round(n.fat_g * scale, 1),
                    fiber_g=round(n.fiber_g * scale, 1),
                    sugar_g=round(n.sugar_g * scale, 1),
                    sodium_mg=round(n.sodium_mg * scale, 1),
                    potassium_mg=round(n.potassium_mg * scale, 1),
                    calcium_mg=round(n.calcium_mg * scale, 1),
                    iron_mg=round(n.iron_mg * scale, 1),
                    vitamin_c_mg=round(n.vitamin_c_mg * scale, 1),
                    source="USDA FDC (scaled)",
                ),
            )
        else:
            fi = FoodItem(name=name, quantity_g=qty)

        food_items.append(fi)

    return food_items


def _aggregate_nutrients(food_items: list[FoodItem]) -> NutrientInfo:
    """Sum nutrient values across all food items in a meal."""
    total = NutrientInfo()
    for fi in food_items:
        n = fi.nutrients
        total.calories += n.calories
        total.protein_g += n.protein_g
        total.carbs_g += n.carbs_g
        total.fat_g += n.fat_g
        total.fiber_g += n.fiber_g
        total.sugar_g += n.sugar_g
        total.sodium_mg += n.sodium_mg
        total.potassium_mg += n.potassium_mg
        total.calcium_mg += n.calcium_mg
        total.iron_mg += n.iron_mg
        total.vitamin_c_mg += n.vitamin_c_mg
    # Round all fields
    for field in total.model_fields:
        val = getattr(total, field)
        if isinstance(val, float):
            setattr(total, field, round(val, 1))
    return total


# ---------------------------------------------------------------------------
# Daily summary
# ---------------------------------------------------------------------------

def get_daily_summary(user_id: str, target_date: str | None = None) -> DailyNutritionSummary:
    """Aggregate all meals for a user on a given date."""
    if not target_date:
        target_date = date.today().isoformat()

    meals = _meal_store[user_id].get(target_date, [])
    all_foods: list[FoodItem] = []
    for meal in meals:
        all_foods.extend(meal.food_items)

    total = _aggregate_nutrients(all_foods)

    # Simple deficiency/surplus detection (vs standard RDAs)
    deficiencies, surpluses = [], []
    rda = {"protein_g": 50, "fiber_g": 25, "calcium_mg": 1000, "iron_mg": 14, "vitamin_c_mg": 65}
    upper = {"sodium_mg": 2300, "sugar_g": 50, "fat_g": 78}

    for nutrient, threshold in rda.items():
        if getattr(total, nutrient, 0) < threshold * 0.7:
            deficiencies.append(nutrient.replace("_", " ").replace(" g", " (g)").replace(" mg", " (mg)"))

    for nutrient, limit in upper.items():
        if getattr(total, nutrient, 0) > limit:
            surpluses.append(nutrient.replace("_", " ").replace(" g", " (g)").replace(" mg", " (mg)"))

    return DailyNutritionSummary(
        user_id=user_id,
        date=target_date,
        meals=meals,
        total_nutrients=total,
        deficiencies=deficiencies,
        surpluses=surpluses,
    )


# ---------------------------------------------------------------------------
# Public agent interface
# ---------------------------------------------------------------------------

def run(
    user_id: str,
    date_str: str,
    meal_type: MealType,
    description: str,
    input_mode: str = "text",
    profile: UserProfile | None = None,
) -> AgentResponse:
    """Log a meal and return instant nutritional feedback."""

    # Pre-process input
    if input_mode == "image_base64":
        description = _preprocess_image(description)
    elif input_mode == "voice_base64":
        description = _preprocess_voice(description)

    # Extract food items
    food_items = _extract_food_items_from_text(description)
    total_nutrients = _aggregate_nutrients(food_items)

    # Store meal
    entry = MealLogEntry(
        user_id=user_id,
        date=date_str,
        meal_type=meal_type,
        description=description,
        food_items=food_items,
        total_nutrients=total_nutrients,
    )
    _meal_store[user_id][date_str].append(entry)

    # Determine calorie targets
    if profile:
        from agents.diet_recommendation_agent import calculate_tdee, calculate_macros
        goal_calories = calculate_tdee(profile)
        macros = calculate_macros(goal_calories, profile.health_conditions)
        conditions_str = ", ".join(c.value for c in profile.health_conditions) or "none"
        user_name = profile.name or "User"
    else:
        goal_calories = 2000.0
        macros = {"protein_g": 125.0, "carbs_g": 250.0, "fat_g": 55.0}
        conditions_str = "none"
        user_name = "User"

    # Calculate daily total so far for remaining calories
    daily_summary = get_daily_summary(user_id, date_str)
    day_total = daily_summary.total_nutrients
    remaining = round(goal_calories - day_total.calories, 1)

    # Generate LLM feedback
    llm = get_llm(temperature=0.4, max_new_tokens=400)
    chain = LLMChain(llm=llm, prompt=_FEEDBACK_PROMPT)
    foods_logged_str = ", ".join(f"{fi.name} ({fi.quantity_g}g)" for fi in food_items) or description

    feedback: str = chain.invoke({
        "user_name": user_name,
        "meal_type": meal_type.value,
        "date": date_str,
        "foods_logged": foods_logged_str,
        "total_calories": round(total_nutrients.calories, 1),
        "total_protein_g": round(total_nutrients.protein_g, 1),
        "total_carbs_g": round(total_nutrients.carbs_g, 1),
        "total_fat_g": round(total_nutrients.fat_g, 1),
        "total_fiber_g": round(total_nutrients.fiber_g, 1),
        "goal_calories": goal_calories,
        "goal_protein_g": macros["protein_g"],
        "goal_carbs_g": macros["carbs_g"],
        "goal_fat_g": macros["fat_g"],
        "health_conditions": conditions_str,
        "remaining_calories": remaining,
    }).get("text", "")

    entry.ai_feedback = feedback

    return AgentResponse(
        agent=AGENT_NAME,
        result={
            "meal_logged": entry.model_dump(),
            "feedback": feedback,
            "daily_summary": daily_summary.model_dump(),
            "remaining_calories": remaining,
        },
        sources=["USDA FoodData Central", "IBM Granite LLM"],
    )
