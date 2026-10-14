"""
agents/health_advisory_agent.py

Health Advisory Agent
─────────────────────
Responsibilities:
  • Generate preventive health and disease-management dietary advice.
  • Use RAG to pull condition-specific nutrition evidence.
  • Provide actionable, evidence-based recommendations with clear warnings.
  • Flag when the user should consult a healthcare professional.

Architecture:
  UserProfile + specific_concern
    → RAG retrieval (condition-specific docs)
    → IBM Granite LLM
    → structured advice with risk flags
"""
from __future__ import annotations

import logging

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from core.llm import get_llm
from core.models import AgentResponse, HealthCondition, UserProfile
from core.vector_store import get_vector_store

logger = logging.getLogger(__name__)
AGENT_NAME = "health_advisory_agent"

# ---------------------------------------------------------------------------
# Condition-to-keyword mapping for focused retrieval
# ---------------------------------------------------------------------------

_CONDITION_KEYWORDS: dict[HealthCondition, str] = {
    HealthCondition.diabetes: "diabetes glycemic glucose carbohydrate insulin",
    HealthCondition.hypertension: "hypertension blood pressure sodium DASH potassium",
    HealthCondition.heart_disease: "heart cardiovascular cholesterol omega-3 saturated fat",
    HealthCondition.celiac: "celiac gluten wheat barley rye",
    HealthCondition.kidney_disease: "kidney CKD phosphorus potassium protein dialysis",
    HealthCondition.obesity: "weight loss calorie deficit metabolism plant-based",
}

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_HEALTH_ADVISORY_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a preventive health and clinical nutrition advisor powered by IBM Granite.
Your role is to provide evidence-based dietary guidance for chronic disease prevention and management.

IMPORTANT: Always include a disclaimer that your advice does not replace professional medical consultation.

Context from nutrition knowledge base:
{context}

Patient query:
{question}

Provide your response in the following structured format:

## Dietary Recommendations
[Specific food choices and eating patterns to follow]

## Foods to Limit or Avoid
[Items that may worsen the condition]

## Key Nutrients to Focus On
[Specific vitamins, minerals, or compounds and their food sources]

## Lifestyle Tips
[Non-dietary but nutrition-adjacent tips: meal timing, hydration, exercise interaction]

## Warning Signs
[Symptoms that should prompt immediate medical attention]

## Medical Disclaimer
[Standard disclaimer]

Response:""",
)


# ---------------------------------------------------------------------------
# Public agent interface
# ---------------------------------------------------------------------------

def run(profile: UserProfile, specific_concern: str = "") -> AgentResponse:
    """Generate preventive health and dietary advice for the user's conditions."""

    # Build a rich query combining profile conditions and the specific concern
    condition_keywords = []
    for condition in profile.health_conditions:
        condition_keywords.append(_CONDITION_KEYWORDS.get(condition, condition.value))

    if specific_concern:
        condition_keywords.append(specific_concern)

    # Age-based additions
    if profile.age >= 65:
        condition_keywords.append("elderly nutrition vitamin D B12 sarcopenia")
    elif profile.age <= 18:
        condition_keywords.append("child adolescent nutrition growth calcium iron")

    query = " ".join(condition_keywords) if condition_keywords else (
        f"healthy balanced diet for {profile.age} year old {profile.gender}"
    )

    full_question = (
        f"Patient: {profile.name or 'Anonymous'}, {profile.age}y {profile.gender}, "
        f"weight {profile.weight_kg}kg, goal: {profile.fitness_goal}.\n"
        f"Conditions: {', '.join(c.value for c in profile.health_conditions) or 'none'}.\n"
        f"Allergies: {', '.join(profile.allergies) or 'none'}.\n"
        f"Specific concern: {specific_concern or 'general preventive health advice'}.\n\n"
        f"Please provide comprehensive dietary guidance."
    )

    llm = get_llm(temperature=0.2, max_new_tokens=1500)
    store = get_vector_store()

    # Use metadata filter if a single known condition
    retriever_kwargs: dict = {"k": 6}
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=store.as_retriever(search_type="similarity", search_kwargs=retriever_kwargs),
        return_source_documents=True,
        chain_type_kwargs={"prompt": _HEALTH_ADVISORY_PROMPT},
    )

    rag_result = chain.invoke({"query": full_question})
    advice: str = rag_result.get("result", "")
    source_docs = rag_result.get("source_documents", [])
    sources = list({doc.metadata.get("title", "") for doc in source_docs if doc.metadata.get("title")})

    # Risk-level tagging
    risk_flags: list[str] = []
    if HealthCondition.kidney_disease in profile.health_conditions:
        risk_flags.append("⚠️ CKD detected — dietary changes must be supervised by a nephrologist.")
    if HealthCondition.diabetes in profile.health_conditions:
        risk_flags.append("⚠️ Diabetes detected — monitor blood glucose when changing diet.")
    if HealthCondition.heart_disease in profile.health_conditions:
        risk_flags.append("⚠️ Cardiovascular condition — consult cardiologist before major dietary changes.")

    return AgentResponse(
        agent=AGENT_NAME,
        result={
            "advice": advice,
            "risk_flags": risk_flags,
            "conditions_addressed": [c.value for c in profile.health_conditions],
            "specific_concern": specific_concern,
        },
        sources=sources,
    )
