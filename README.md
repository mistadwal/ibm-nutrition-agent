# IBM Nutrition Agent 🥗🤖

> **AI-powered multi-agent system** for personalised nutrition guidance, built with **IBM Granite Models via watsonx.ai**, LangChain, RAG (ChromaDB), and the USDA FoodData Central API.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  FastAPI REST Backend                    │
│                  (app.py — port 8000)                   │
└──────────────────────┬──────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │   Orchestrator   │
              │ (orchestrator.py)│
              └──┬──┬──┬──┬────┘
     ┌───────────┘  │  │  └──────────────┐
     ▼              ▼  ▼                 ▼
┌─────────┐  ┌──────┐  ┌────────┐  ┌────────────┐
│Knowledge │  │ Diet │  │Health  │  │  Food Log  │
│  Agent   │  │Agent │  │Advisor │  │  & Feedback│
│ (RAG)   │  │      │  │        │  │   Agent    │
└────┬────┘  └──┬───┘  └───┬────┘  └─────┬──────┘
     │          │           │              │
     ▼          ▼           ▼              ▼
 ChromaDB   Granite     Granite       USDA FDC
 + Granite  LLM         LLM +         API +
 Embeddings             ChromaDB      Granite LLM
     │
     ▼
  USDA FDC
  API (live)
```

### The 4 Agents

| Agent | Responsibility | Key Tech |
|-------|---------------|----------|
| **Nutrition Knowledge Agent** | RAG over curated nutrition KB + live USDA food lookup | ChromaDB · Granite Embeddings · USDA FDC API |
| **Diet Recommendation Agent** | Personalised multi-day meal plans (allergies, conditions, culture) | Granite LLM · Mifflin-St Jeor formula |
| **Health Advisory Agent** | Disease-specific dietary guidance (diabetes, hypertension, CKD…) | Granite LLM · RAG · Risk flagging |
| **Food Log & Feedback Agent** | Meal logging (text/image/voice) + nutritional analysis + feedback | OCR · Speech-to-text · USDA FDC · Granite LLM |

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- IBM Cloud account with **watsonx.ai** access ([Get free trial](https://cloud.ibm.com/registration))
- IBM watsonx.ai Project ID
- (Optional) USDA FoodData Central API key ([Free registration](https://fdc.nal.usda.gov/api-guide.html))

### 2. Install dependencies

```bash
cd "ibm project/nutrition_agent"
pip install -r requirements.txt
```

### 3. Configure credentials

```bash
cp .env.example .env
# Edit .env with your IBM watsonx.ai credentials:
#   WATSONX_API_KEY=...
#   WATSONX_PROJECT_ID=...
```

### 4. Run the server

```bash
python app.py
```

Server starts at **http://localhost:8000**

### 5. Open the dashboard

Navigate to **http://localhost:8000/dashboard** for the full visual interface.

Or explore the API docs at **http://localhost:8000/docs** (Swagger UI).

---

## 📡 API Reference

### Nutrition Knowledge (RAG)
```
POST /api/nutrition/query
{
  "query": "What are the nutritional values of 100g quinoa?",
  "user_id": "optional"
}
```

### Diet Recommendation
```
POST /api/diet/plan
{
  "profile": {
    "user_id": "u001", "name": "Alice", "age": 32, "gender": "female",
    "weight_kg": 65, "height_cm": 165,
    "activity_level": "moderately_active",
    "health_conditions": ["diabetes"],
    "allergies": ["shellfish"],
    "dietary_preferences": ["vegetarian"],
    "cultural_background": "South Asian",
    "fitness_goal": "lose_weight"
  },
  "num_days": 7
}
```

### Health Advisory
```
POST /api/health/advice
{
  "profile": { ... },
  "specific_concern": "How can I lower my HbA1c through diet?"
}
```

### Meal Logging
```
POST /api/meals/log
{
  "user_id": "u001",
  "date": "2025-10-14",
  "meal_type": "lunch",
  "description": "Grilled salmon 150g, brown rice 180g, steamed broccoli",
  "input_mode": "text"
}
```

### Daily Summary
```
GET /api/meals/summary/{user_id}?date_str=2025-10-14
```

### Full Onboarding
```
POST /api/onboard
{ "profile": { ... } }
```
Returns 7-day meal plan + health advice in one call.

---

## 🧪 Running Tests

```bash
cd "ibm project/nutrition_agent"
pytest tests/ -v
```

Tests use mocked LLM calls — no IBM credentials required for testing.

---

## 📂 Project Structure

```
nutrition_agent/
├── app.py                          # FastAPI application + API routes
├── orchestrator.py                 # Multi-agent orchestrator
├── requirements.txt
├── .env.example                    # Credential template
│
├── core/
│   ├── config.py                   # Settings (loaded from .env)
│   ├── llm.py                      # Granite LLM + embeddings factory
│   ├── vector_store.py             # ChromaDB vector store
│   └── models.py                   # Shared Pydantic data models
│
├── agents/
│   ├── nutrition_knowledge_agent.py  # Agent 1: RAG + USDA lookup
│   ├── diet_recommendation_agent.py  # Agent 2: Personalised meal plans
│   ├── health_advisory_agent.py      # Agent 3: Preventive health advice
│   └── food_log_feedback_agent.py    # Agent 4: Meal logging + analysis
│
├── data/
│   ├── chroma_db/                  # Persistent vector store (auto-created)
│   └── ingest_custom_docs.py       # CLI tool for adding custom documents
│
├── static/
│   └── dashboard.html              # Interactive visualization dashboard
│
└── tests/
    └── test_agents.py              # pytest unit tests
```

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | IBM Granite 13B Instruct v2 via watsonx.ai |
| **Embeddings** | IBM Slate 125M English via watsonx.ai (fallback: sentence-transformers) |
| **RAG** | LangChain RetrievalQA + ChromaDB |
| **Framework** | LangChain 0.2 |
| **API** | FastAPI + Uvicorn |
| **Food Data** | USDA FoodData Central REST API |
| **Image OCR** | Pytesseract + Pillow |
| **Voice** | SpeechRecognition (Google STT) |
| **Dashboard** | Vanilla JS + Chart.js |
| **Validation** | Pydantic v2 |

---

## 🩺 Supported Health Conditions

- **Diabetes** — low-GI, carb-controlled, Mediterranean/DASH patterns
- **Hypertension** — DASH diet, sodium restriction, potassium-rich foods
- **Heart Disease** — AHA guidelines, omega-3 focus, LDL-lowering strategies
- **Celiac Disease** — strict gluten-free, hidden gluten awareness
- **Chronic Kidney Disease (CKD)** — phosphorus/potassium/protein restriction
- **Obesity** — caloric deficit, plant-forward, fiber-rich

---

## 🌍 Fallback / Demo Mode

If no IBM watsonx.ai credentials are configured, the system automatically:
- Falls back to a **local sentence-transformers model** for embeddings
- Returns a **demo stub** from the LLM (clearly labelled)
- Still performs **USDA FDC lookups** (uses `DEMO_KEY` for basic access)

This allows local testing and UI development without cloud credentials.

---

## 🤝 Extending the System

### Add custom nutrition documents
```bash
python data/ingest_custom_docs.py --path ./my_docs/
python data/ingest_custom_docs.py --url https://www.who.int/news-room/fact-sheets/detail/healthy-diet
```

### Switch to a different Granite model
Edit `.env`:
```
GRANITE_LLM_MODEL=ibm/granite-34b-code-instruct
```

### Use IBM Cloud Agentic Lab
Replace the `WatsonxLLM` initialisation in `core/llm.py` with the
IBM Cloud Agentic Lab SDK client — all agent prompts and orchestrator
logic remain the same.

---

## ⚠️ Medical Disclaimer

This system provides **general dietary information and guidance only**.
It does **not** replace the advice of a qualified healthcare professional,
registered dietitian, or physician. Always consult a medical professional
before making significant dietary changes, especially for managing
chronic diseases.
