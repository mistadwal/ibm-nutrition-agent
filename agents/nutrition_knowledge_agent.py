"""
agents/nutrition_knowledge_agent.py

Nutrition Knowledge Agent
─────────────────────────
Responsibilities:
  • Maintain a RAG pipeline over a curated nutrition knowledge base.
  • Fetch real-time nutritional values from the USDA FoodData Central API.
  • Answer factual questions about food composition, calories, macros, and micronutrients.

Architecture:
  LangChain RetrievalQA  →  ChromaDB (IBM Granite Embeddings)  →  IBM Granite LLM
  +
  USDA FDC REST API  →  live food lookup
"""
from __future__ import annotations

import json
import logging
from typing import Any

import requests
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from core.config import get_settings
from core.llm import get_llm
from core.models import AgentResponse, FoodItem, NutrientInfo
from core.vector_store import get_vector_store, ingest_documents

logger = logging.getLogger(__name__)

AGENT_NAME = "nutrition_knowledge_agent"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_NUTRITION_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a certified nutrition expert assistant powered by IBM Granite.
Use the retrieved context below to answer the question accurately and concisely.
If the context does not contain sufficient information, say so honestly — do not fabricate data.

Context:
{context}

Question: {question}

Answer (include specific nutritional values when available, cite source):""",
)

# ---------------------------------------------------------------------------
# USDA FoodData Central helper
# ---------------------------------------------------------------------------

_USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
_USDA_FOOD_URL = "https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"


def _usda_search(food_name: str, max_results: int = 3) -> list[dict[str, Any]]:
    """Search USDA FDC for a food item. Returns a list of result dicts."""
    settings = get_settings()
    try:
        resp = requests.get(
            _USDA_SEARCH_URL,
            params={"query": food_name, "pageSize": max_results, "api_key": settings.usda_api_key},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("foods", [])
    except Exception as exc:
        logger.warning("USDA API error for '%s': %s", food_name, exc)
        return []


def _parse_usda_nutrients(food_dict: dict) -> NutrientInfo:
    """Extract macros/micros from a USDA food dict into NutrientInfo."""
    nutrient_map = {
        "Energy": "calories",
        "Protein": "protein_g",
        "Carbohydrate, by difference": "carbs_g",
        "Total lipid (fat)": "fat_g",
        "Fiber, total dietary": "fiber_g",
        "Sugars, total including NLEA": "sugar_g",
        "Sodium, Na": "sodium_mg",
        "Potassium, K": "potassium_mg",
        "Calcium, Ca": "calcium_mg",
        "Iron, Fe": "iron_mg",
        "Vitamin C, total ascorbic acid": "vitamin_c_mg",
        "Vitamin D (D2 + D3), International Units": "vitamin_d_iu",
        "Vitamin B-12": "vitamin_b12_ug",
    }
    info: dict[str, float] = {}
    for nutrient in food_dict.get("foodNutrients", []):
        n_name = nutrient.get("nutrientName") or nutrient.get("name", "")
        if n_name in nutrient_map:
            info[nutrient_map[n_name]] = round(float(nutrient.get("value", 0) or 0), 2)

    return NutrientInfo(**info, source="USDA FoodData Central")


def lookup_food(food_name: str) -> FoodItem | None:
    """Look up a food item in USDA FDC and return a FoodItem with nutrient data."""
    results = _usda_search(food_name, max_results=1)
    if not results:
        return None
    top = results[0]
    nutrients = _parse_usda_nutrients(top)
    return FoodItem(
        name=top.get("description", food_name),
        quantity_g=100.0,
        nutrients=nutrients,
    )


# ---------------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------------

def _build_rag_chain() -> RetrievalQA:
    llm = get_llm()
    store = get_vector_store()
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": _NUTRITION_RAG_PROMPT},
    )
    return chain


# ---------------------------------------------------------------------------
# Seed knowledge base (called once at startup)
# ---------------------------------------------------------------------------

def seed_knowledge_base() -> None:
    """Load curated nutrition facts into the vector store if it is empty."""
    store = get_vector_store()
    if store._collection.count() > 0:
        logger.info("Knowledge base already seeded (%d docs). Skipping.", store._collection.count())
        return

    seed_docs = _build_seed_documents()
    ingest_documents(seed_docs)
    logger.info("Knowledge base seeded with %d documents.", len(seed_docs))


def _build_seed_documents() -> list[Document]:
    """Return a curated list of nutrition reference documents."""
    entries = [
        # --- Macronutrients ---
        ("macros_carbs", "Carbohydrates", "nutrition",
         "Carbohydrates are the body's primary energy source. 1 gram provides 4 kcal. "
         "Complex carbs (whole grains, legumes) are preferred over simple sugars. "
         "Recommended intake: 45–65% of total daily calories. High-fiber carbs improve gut health "
         "and reduce risk of type-2 diabetes. Diabetics should monitor glycemic index (GI)."),

        ("macros_protein", "Protein", "nutrition",
         "Protein provides 4 kcal per gram and is essential for muscle repair, immune function, "
         "and enzyme production. RDA is 0.8 g/kg body weight for sedentary adults; athletes may "
         "need 1.2–2.2 g/kg. Complete proteins contain all 9 essential amino acids (meat, eggs, soy, quinoa). "
         "High protein diets can stress kidneys — caution for kidney disease patients."),

        ("macros_fat", "Dietary Fat", "nutrition",
         "Fat provides 9 kcal/g. Unsaturated fats (olive oil, avocado, nuts) are heart-healthy. "
         "Saturated fat should be <10% of calories. Trans fats should be avoided entirely. "
         "Omega-3 fatty acids (fish, flaxseed) reduce inflammation and support cardiovascular health. "
         "Recommended fat intake: 20–35% of total calories."),

        # --- Micronutrients ---
        ("micro_vitamins", "Vitamins Overview", "micronutrients",
         "Fat-soluble vitamins: A, D, E, K — stored in body fat; excess can be toxic. "
         "Water-soluble vitamins: B-complex (B1, B2, B3, B5, B6, B7, B9, B12) and Vitamin C — excreted in urine. "
         "Vitamin D deficiency is widespread; sources include sunlight, fatty fish, fortified dairy. "
         "Vitamin B12 is found only in animal products; vegans must supplement. "
         "Folate (B9) is critical during pregnancy to prevent neural tube defects."),

        ("micro_minerals", "Minerals Overview", "micronutrients",
         "Calcium (RDA 1000 mg): bone health; sources: dairy, leafy greens, fortified foods. "
         "Iron (RDA 8–18 mg): oxygen transport; sources: red meat, lentils, spinach. Pair with Vitamin C to boost absorption. "
         "Sodium: limit to <2300 mg/day; excess raises blood pressure. "
         "Potassium (RDA 4700 mg): counteracts sodium, supports heart rhythm; sources: bananas, potatoes, beans. "
         "Magnesium (RDA 310–420 mg): muscle and nerve function; sources: nuts, seeds, whole grains."),

        # --- Disease-specific diets ---
        ("disease_diabetes", "Diabetes Nutrition", "disease_management",
         "For type-2 diabetes management: reduce refined carbohydrates and added sugars. "
         "Focus on low-GI foods: legumes, non-starchy vegetables, whole grains. "
         "Distribute carbohydrate intake evenly across meals (carb counting). "
         "Increase dietary fiber (>25 g/day) to slow glucose absorption. "
         "Limit saturated fat to protect cardiovascular health. "
         "Mediterranean and DASH diets show strong evidence for glycemic control. "
         "Avoid sugary beverages; choose water, unsweetened tea, or black coffee."),

        ("disease_hypertension", "Hypertension (High Blood Pressure) Diet", "disease_management",
         "DASH Diet is the gold standard for hypertension: rich in fruits, vegetables, whole grains, low-fat dairy. "
         "Reduce sodium to <1500 mg/day for stage-2 hypertension. "
         "Increase potassium (bananas, sweet potatoes), magnesium, and calcium intake. "
         "Limit alcohol: no more than 1 drink/day for women, 2 for men. "
         "Avoid processed meats, canned soups, and fast food (high sodium sources). "
         "Regular physical activity and weight management are essential adjuncts."),

        ("disease_heart", "Heart-Healthy Diet", "disease_management",
         "AHA recommends: ≥5 servings of fruits and vegetables daily, whole grains, lean proteins, and fish 2x/week. "
         "Replace saturated fats with unsaturated fats (olive oil, canola oil). "
         "Eliminate trans fats entirely. Reduce added sugars and sodium. "
         "Oily fish (salmon, mackerel) provide EPA/DHA omega-3s that lower triglycerides. "
         "Plant sterols/stanols (2 g/day) can reduce LDL cholesterol by 5–15%. "
         "Soluble fiber (oats, barley, psyllium) also lowers LDL."),

        ("disease_celiac", "Celiac Disease & Gluten-Free Diet", "disease_management",
         "Celiac disease requires strict lifelong gluten avoidance (wheat, barley, rye, and contaminated oats). "
         "Safe grains: rice, corn, quinoa, buckwheat, millet, sorghum, teff, certified gluten-free oats. "
         "Watch for hidden gluten in sauces, dressings, processed meats, and medications. "
         "Nutrient deficiencies common: iron, calcium, B12, folate, vitamin D — monitor and supplement. "
         "Ensure adequate fiber intake from vegetables, legumes, and GF whole grains."),

        ("disease_kidney", "Chronic Kidney Disease (CKD) Diet", "disease_management",
         "CKD diet goals: reduce workload on kidneys by limiting protein (0.6–0.8 g/kg/day for non-dialysis stages). "
         "Restrict phosphorus (dairy, nuts, seeds, cola drinks) to prevent bone disease. "
         "Limit potassium if hyperkalemia: avoid bananas, oranges, potatoes, tomatoes; choose apples, berries, cabbage. "
         "Sodium restriction (<2000 mg/day) helps manage fluid retention and blood pressure. "
         "Dialysis patients may need increased protein. Always follow nephrologist guidance."),

        # --- Dietary patterns ---
        ("pattern_mediterranean", "Mediterranean Diet", "dietary_patterns",
         "One of the most evidence-backed dietary patterns. Key components: "
         "abundant plant foods (vegetables, fruits, legumes, whole grains, nuts), "
         "olive oil as primary fat, moderate fish and poultry, minimal red meat, "
         "moderate wine (optional), and low added sugar. "
         "Benefits: reduced cardiovascular disease, type-2 diabetes, cognitive decline, and all-cause mortality. "
         "WHO recommends it as a sustainable, culturally adaptable healthy eating pattern."),

        ("pattern_dash", "DASH Diet", "dietary_patterns",
         "Dietary Approaches to Stop Hypertension. Emphasizes: fruits, vegetables, whole grains, "
         "low-fat dairy, lean meats, fish, nuts, beans. Limits: sodium, sweets, sugary drinks, red meat. "
         "Sodium target: 1500–2300 mg/day. Proven to lower systolic BP by 8–14 mmHg. "
         "Also reduces LDL cholesterol and improves insulin sensitivity."),

        ("pattern_plant_based", "Plant-Based Diets", "dietary_patterns",
         "Vegetarian: excludes meat/fish. Vegan: excludes all animal products. "
         "Health benefits: lower BMI, reduced cardiovascular risk, lower cancer incidence. "
         "Nutritional watch-points: B12 (must supplement), vitamin D, omega-3 (use algae oil), "
         "iron (use cast-iron cookware, pair with Vitamin C), zinc, calcium, iodine. "
         "Well-planned plant-based diets are nutritionally adequate at all life stages."),

        # --- Life-stage nutrition ---
        ("life_pregnancy", "Pregnancy Nutrition", "life_stages",
         "Increased needs during pregnancy: folate (600 mcg/day — critical in first trimester), "
         "iron (27 mg/day), calcium (1000 mg/day), iodine (220 mcg/day), DHA (200 mg/day). "
         "Avoid: alcohol, high-mercury fish (shark, swordfish, king mackerel), raw/undercooked meat, "
         "unpasteurised cheeses, excess vitamin A (>10,000 IU/day). "
         "Weight gain recommendation: 11.5–16 kg for normal BMI."),

        ("life_children", "Child & Adolescent Nutrition", "life_stages",
         "Children need adequate protein, calcium, iron, and zinc for growth. "
         "Calcium and vitamin D critical for bone density (peak bone mass set by age 25). "
         "Iron-deficiency anaemia is the most common nutritional deficiency in children worldwide. "
         "Limit added sugars, sugary drinks, and ultra-processed foods. "
         "Adolescents: increased iron needs for girls (menstruation); protein for muscle development in boys."),

        ("life_elderly", "Elderly Nutrition", "life_stages",
         "Older adults (>65): calorie needs decrease but micronutrient needs remain or increase. "
         "Protein needs increase (1.0–1.2 g/kg/day) to prevent sarcopenia. "
         "Vitamin D (800–1000 IU/day) and calcium for fracture prevention. "
         "B12 absorption decreases — crystalline B12 supplements or fortified foods recommended. "
         "Hydration is critical — thirst sensation diminishes with age. "
         "Omega-3s support cognitive health."),
    ]

    docs = []
    for doc_id, title, category, content in entries:
        docs.append(
            Document(
                page_content=f"{title}\n\n{content}",
                metadata={"doc_id": doc_id, "title": title, "category": category, "source": "Curated Nutrition KB"},
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Public agent interface
# ---------------------------------------------------------------------------

def run(query: str, user_context: dict | None = None) -> AgentResponse:
    """
    Answer a nutrition knowledge query using RAG.
    Also attempts a live USDA FDC lookup if the query looks like a food name.
    """
    sources: list[str] = []

    # Try live food lookup first for short, food-name-like queries
    live_food: FoodItem | None = None
    if len(query.split()) <= 5 and not query.endswith("?"):
        live_food = lookup_food(query)

    chain = _build_rag_chain()
    rag_result = chain.invoke({"query": query})
    answer: str = rag_result.get("result", "")
    source_docs: list[Document] = rag_result.get("source_documents", [])

    for doc in source_docs:
        title = doc.metadata.get("title", "")
        if title and title not in sources:
            sources.append(title)

    # Enrich answer with live USDA data if available
    if live_food:
        n = live_food.nutrients
        usda_snippet = (
            f"\n\n📊 Live USDA Data for '{live_food.name}' (per 100 g):\n"
            f"  Calories: {n.calories} kcal | Protein: {n.protein_g} g | "
            f"Carbs: {n.carbs_g} g | Fat: {n.fat_g} g | "
            f"Fiber: {n.fiber_g} g | Sodium: {n.sodium_mg} mg"
        )
        answer += usda_snippet
        sources.append("USDA FoodData Central (live)")

    return AgentResponse(
        agent=AGENT_NAME,
        result={"answer": answer, "live_food_data": live_food.model_dump() if live_food else None},
        sources=sources,
    )
