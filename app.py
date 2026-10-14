"""
app.py
FastAPI application — REST API for the IBM Nutrition Agent system.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.models import (
    AgentResponse,
    DietPlanRequest,
    HealthAdviceRequest,
    MealLogRequest,
    MealType,
    NutritionQueryRequest,
    UserProfile,
)
import orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: seed vector store on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting IBM Nutrition Agent API...")
    try:
        from agents.nutrition_knowledge_agent import seed_knowledge_base
        seed_knowledge_base()
        logger.info("✅ Knowledge base ready.")
    except Exception as exc:
        logger.error("Failed to seed knowledge base: %s", exc)
    yield
    logger.info("🛑 Shutting down Nutrition Agent API.")


# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="IBM Nutrition Agent API",
    description=(
        "AI-powered Nutrition Agent using IBM Granite Models via watsonx.ai. "
        "Provides personalised diet plans, nutrition knowledge retrieval (RAG), "
        "meal tracking with feedback, and preventive health advice."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static dashboard files
import os
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to the dashboard."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "IBM Nutrition Agent"}


# ---------------------------------------------------------------------------
# Agent 1: Nutrition Knowledge (RAG)
# ---------------------------------------------------------------------------

@app.post("/api/nutrition/query", response_model=AgentResponse, tags=["Knowledge Agent"])
async def nutrition_query(request: NutritionQueryRequest):
    """
    Query the nutrition knowledge base using RAG + IBM Granite.
    Also fetches live nutritional data from USDA FoodData Central.
    """
    try:
        return orchestrator.query_nutrition_knowledge(request)
    except Exception as exc:
        logger.exception("Error in nutrition_query")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Agent 2: Diet Recommendation
# ---------------------------------------------------------------------------

@app.post("/api/diet/plan", response_model=AgentResponse, tags=["Diet Recommendation Agent"])
async def diet_plan(request: DietPlanRequest):
    """
    Generate a personalised multi-day meal plan for the given user profile.
    Respects allergies, health conditions, cultural preferences, and fitness goals.
    """
    try:
        return orchestrator.get_diet_plan(request)
    except Exception as exc:
        logger.exception("Error in diet_plan")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Agent 3: Health Advisory
# ---------------------------------------------------------------------------

@app.post("/api/health/advice", response_model=AgentResponse, tags=["Health Advisory Agent"])
async def health_advice(request: HealthAdviceRequest):
    """
    Get preventive health and chronic disease management dietary advice.
    Tailored to the user's specific health conditions and concerns.
    """
    try:
        return orchestrator.get_health_advice(request)
    except Exception as exc:
        logger.exception("Error in health_advice")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Agent 4: Meal Logging & Feedback
# ---------------------------------------------------------------------------

@app.post("/api/meals/log", response_model=AgentResponse, tags=["Food Log & Feedback Agent"])
async def log_meal(request: MealLogRequest):
    """
    Log a meal (text description) and receive instant nutritional analysis and feedback.
    """
    try:
        return orchestrator.log_meal(request)
    except Exception as exc:
        logger.exception("Error in log_meal")
        raise HTTPException(status_code=500, detail=str(exc))


class ImageMealLog(BaseModel):
    user_id: str
    date: str
    meal_type: MealType
    image_base64: str


@app.post("/api/meals/log/image", response_model=AgentResponse, tags=["Food Log & Feedback Agent"])
async def log_meal_image(request: ImageMealLog):
    """
    Log a meal by uploading a base64-encoded food image.
    OCR extracts text, LLM parses food items.
    """
    try:
        log_request = MealLogRequest(
            user_id=request.user_id,
            date=request.date,
            meal_type=request.meal_type,
            description=request.image_base64,
            input_mode="image_base64",
        )
        return orchestrator.log_meal(log_request)
    except Exception as exc:
        logger.exception("Error in log_meal_image")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Daily Summary & Dashboard Data
# ---------------------------------------------------------------------------

@app.get("/api/meals/summary/{user_id}", tags=["Food Log & Feedback Agent"])
async def daily_summary(user_id: str, date_str: Optional[str] = None):
    """Get aggregated daily nutrition summary for a user."""
    try:
        from agents.food_log_feedback_agent import get_daily_summary
        summary = get_daily_summary(user_id, date_str or date.today().isoformat())
        return summary
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Compound: Full Onboarding
# ---------------------------------------------------------------------------

@app.post("/api/onboard", tags=["Orchestrator"])
async def onboard_user(profile: UserProfile):
    """
    Full onboarding: generates 7-day meal plan + health advice in one call.
    """
    try:
        return orchestrator.full_onboarding(profile)
    except Exception as exc:
        logger.exception("Error in onboard_user")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Dashboard (serves HTML)
# ---------------------------------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    """Serve the interactive nutrition dashboard."""
    dashboard_path = os.path.join("static", "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard not found. Place dashboard.html in /static/</h1>", status_code=404)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    from core.config import get_settings
    s = get_settings()
    uvicorn.run("app:app", host=s.app_host, port=s.app_port, reload=True, log_level=s.log_level)
