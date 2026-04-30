markdown
# Smart Travel Planner

An AI‑powered travel advisor that helps users plan trips by understanding their preferences, retrieving destination knowledge, checking live conditions, and delivering a personalized recommendation — all through a real‑time chat interface with streaming, tool‑call visibility, and webhook delivery.

---

## Architecture
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ React │────▶│ FastAPI │────▶│ PostgreSQL │
│ Frontend │◀────│ Backend │◀────│ + pgvector │
└──────────────┘ └──────┬───────┘ └──────────────┘
│
┌────────┴────────┐
│ LangGraph Agent│
│ (4 tools) │
└────────┬────────┘
│
┌─────────────────┼─────────────────┐
│ │ │
┌──────▼──────┐ ┌──────▼──────┐ ┌───────▼──────┐
│ RAG Tool │ │ Style Tool │ │ Weather Tool │
│ (pgvector) │ │ (ML model) │ │ (OpenWeather)│
└─────────────┘ └─────────────┘ └──────────────┘
│
┌───────▼────────┐
│ Style Search │
│ (CSV filter) │
└────────────────┘

text

- **Frontend**: React + TypeScript + Vite, streaming chat with SSE, tool timeline, JWT auth.
- **Backend**: FastAPI, async throughout, dependency injection, lifespan singletons.
- **Database**: PostgreSQL 16 + pgvector for relational data and vector embeddings.
- **Agent**: LangGraph with four tools – RAG retrieval, style classification, weather lookup, and style‑based destination search.
- **LLM**: Multi‑provider fallback (OpenRouter, Gemini, OpenAI, Groq) using LangChain.
- **Embeddings**: OpenRouter API (`intfloat/e5-base-v2`, 384‑dim).
- **Webhooks**: Discord delivery with retry and backoff.

---

## Dataset – Travel Destination Classification

### Labeling Rules
- **Adventure**: Destinations where the primary draw is outdoor activities, hiking, extreme sports, or nature exploration (e.g., Swiss Alps, Queenstown, Nepal).
- **Relaxation**: Beach resorts, islands, spa‑focused locations (e.g., Maldives, Bali, Seychelles).
- **Culture**: Cities with significant historical, artistic, or architectural heritage (e.g., Rome, Kyoto, Istanbul).
- **Budget**: Affordable destinations known for low cost of living and cheap travel options (e.g., Hanoi, Delhi, Cairo).
- **Luxury**: High‑end destinations with expensive accommodations, fine dining, and exclusive experiences (e.g., Monaco, Aspen, Dubai).
- **Family**: Destinations with family‑friendly attractions like theme parks, safe environments, and activities for children (e.g., Orlando, Tokyo Disney, Copenhagen).

### Features
| Feature | Justification |
|---------|---------------|
| `avg_temp_winter` / `avg_temp_summer` | Realistic temperature ranges per style (e.g., Relaxation = warm year‑round). Sampled via truncated normal distribution. |
| `cost_per_day` | Estimated daily cost in USD, correlated with style (Budget < 70, Luxury > 200). |
| `tourist_density` | Categorical (low/medium/high) + continuous `density_score` – higher for Culture/Budget, lower for Adventure/Luxury. |
| `has_beach`, `beach_score`, `hiking_score`, `culture_score`, `nightlife_score` | Continuous 0‑10 scores driven by style probabilities, reflecting typical activity availability. |
| `safety_index` | 1‑10, slightly boosted by higher cost, reflecting realistic safety perceptions. |
| `hemisphere` | North/South based on curated list – ensures temperature interpretation is correct. |
| `peak_season` | Derived from temperature contrast; helps match travel timing. |
| `visa_difficulty` | easy/medium/hard influenced by style (Luxury/Family easier). |
| `english_friendly` | Binary, higher for Luxury/Culture. |

All generation parameters are documented in the dataset creation script. The final dataset contains **149 destinations** across six styles.

---

## ML Classifier

### Pipeline
- **Preprocessing**: `ColumnTransformer` with `StandardScaler` for numeric features and `OneHotEncoder` for categorical features.
- **Classifiers compared**:
  1. Logistic Regression (multinomial, class_weight=balanced)
  2. Random Forest (class_weight=balanced)
  3. Gradient Boosting
- **Validation**: Stratified 5‑fold cross‑validation, macro F1 scoring.
- **Hyperparameter tuning**: `GridSearchCV` on the best‑performing model (Random Forest).

### Results (`results.csv`)
| Model | CV Macro F1 (mean ± std) | Test Macro F1 | Test Accuracy |
|-------|--------------------------|---------------|---------------|
| Logistic Regression | 0.84 ± 0.06 | 0.83 | 0.83 |
| Random Forest | 0.87 ± 0.05 | 0.90 | 0.90 |
| Gradient Boosting | 0.85 ± 0.04 | 0.87 | 0.87 |

**Tuned Random Forest** (best parameters: `max_depth=10`, `min_samples_split=2`, `n_estimators=100`) achieved **Test Macro F1 = 0.90**.

### Imbalanced Classes Handling
- `class_weight='balanced'` used for Logistic Regression and Random Forest.
- Per‑class metrics reported via `classification_report` ensuring rare styles (Family, Luxury) are not overlooked.

The final model is saved as `backend/models/best_model.pkl` and loaded as a singleton at startup.

---

## RAG Tool

### Chunking & Storage
- **Source**: Hand‑written descriptive chunks for 10 destinations (28 documents total).
- **Embedding model**: `intfloat/e5-base-v2` (384 dimensions, free via OpenRouter).
- **Database**: PostgreSQL + pgvector. Embeddings stored in `destination_documents` table.

### Retrieval Strategy
- **Index**: pgvector with cosine distance (`<=>`).
- **Query embedding**: Generated at runtime via OpenRouter API call (async, no local model).
- **Similarity**: `1 - (embedding <=> query_embedding)` for direct cosine similarity.
- **Deduplication**: `DISTINCT ON (destination_name)` ensures only the best chunk per destination is returned.
- **Top‑k**: Configurable; default 5 destinations.
- **Justification**: Chunk size of 1–2 sentences captures coherent facts without losing context. No overlap needed because each chunk is a self‑contained fact.

### Retrieval Quality
Tested with hand‑written queries like *“I want a beautiful beach and temples”*, returning Bali and other relevant destinations with similarity scores. The combination of dense retrieval (vector search) and post‑processing deduplication gives high precision for the agent.

---

## Agent

### Tools (with Pydantic Validation)

| Tool | Purpose | Input Validation |
|------|---------|------------------|
| `rag_wrapper` | Retrieve destination content via vector search | `RAGToolInput` (query: str) |
| `style_wrapper` | Predict travel style for a destination using the ML model | `StyleToolInput` (destination_name: str) |
| `weather_wrapper` | Get current weather from OpenWeatherMap | `WeatherToolInput` (city: str) |
| `style_search_wrapper` | List destinations matching a specific travel style from the dataset | `StyleSearchInput` (style: str) |

- Every tool input is validated by a Pydantic model before execution. Invalid inputs are caught and returned as structured errors, never crashing the agent.
- A tool allowlist is enforced; any LLM‑invented tool results in an error message.

### Agent Flow
- Built with LangGraph (`StateGraph`).
- Nodes: `call_model` (LLM reasoning + tool choice), `call_tools` (execution).
- Conditional edges: continue if `tool_calls` present, else end.
- Recursion limit configurable (default 100).
- Streaming: `astream_events` returns tokens (`on_chat_model_stream`), tool start/end events, and a final `done` event.

### Prompt Engineering
The system prompt instructs the LLM to:
- Use tools to gather information.
- Never leave the answer empty.
- Avoid unsolicited advice about budget/style unless asked.
- Handle tool failures gracefully.

---

## Two Models, One Agent (Optional Extension – Not Yet Implemented)

*Planned but not completed due to time constraints.*  
The idea is to route cheap calls (tool argument extraction) to a lightweight model (e.g., Gemini Flash-Lite) and final synthesis to a stronger model (e.g., Gemini Pro). Token usage would be logged per step and cost per query reported.

---

## Persistence – PostgreSQL + pgvector

### Tables
- `users`: email, hashed password, creation timestamp.
- `agent_runs`: user_id, query, answer, tools_fired (JSON), created_at.
- `destination_documents`: destination_name, content, embedding (384‑dim vector).

### Design Decisions
- Agent runs are logged **asynchronously** with a fresh database session, so logging never blocks the user response.
- Embeddings are pre‑computed and stored once; queries only need an API call for the runtime embedding.
- SQLAlchemy 2.x async sessions used throughout; no synchronous DB calls in request paths.

---

## Authentication

- Registration (email + password, min 8 chars) with bcrypt hashing.
- Login returns JWT (HS256, configurable expiry).
- All `/chat` endpoints are protected by `Depends(get_current_user)`.
- Auth router separated under `/auth` (register, login, me).

---
Cost Per Query Breakdown
Step	Model	Input Tokens*	Input Price (per 1M)	Output Tokens*	Output Price (per 1M)	Cost
1. Embedding the user query	intfloat/e5-base-v2	~30	$0.005	384-dim vector (free)	$0	$0.00000015
2. LLM tool selection + reasoning	google/gemini-2.5-flash-lite	~1,200	$0.10	~200	$0.40	$0.00020
3. LLM final synthesis	google/gemini-2.5-flash-lite	~1,200	$0.10	~400	$0.40	$0.00028
Total						$0.00048

## React Frontend

- Built with Vite + React + TypeScript.
- Protected routes (`/login`, `/register`, `/chat`).
- Streaming chat: SSE events parsed client‑side, with robust handling for multiple token formats (string, list, dict).
- Tool timeline: shows each tool start (`⚙️ Calling …`) and end (`✅ … completed`) with truncated output.
- Error boundary prevents the entire UI from crashing on malformed SSE data.

---

## Webhook Delivery

- **Channel**: Discord webhook.
- **Implementation**: `httpx.AsyncClient` with 10‑second timeout.
- **Retry**: `tenacity` with exponential backoff (3 attempts, 1‑10s wait).
- **Failure isolation**: Runs as `asyncio.create_task` – never blocks the user response.
- **Logging**: Structured logging on failure.

---

## Docker

Full stack containerised with `docker-compose.yml`:
- **postgres**: `pgvector/pgvector:pg16` with named volume `postgres_data` (data survives restarts).
- **backend**: Python 3.12‑slim, uv for dependency management, multi‑stage build to keep image lean.
- **frontend**: Node 20 Alpine build stage → Nginx Alpine serving static files.

### Quick Start
```bash
docker compose up --build
Open http://localhost:5173.