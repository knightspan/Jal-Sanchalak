---
title: Dam Flood Control
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

## Why This Environment?

Dam management is a genuine real-world challenge affecting millions of people. Poorly managed dams cause catastrophic floods (Banqiao, 1975 — 170,000 deaths) or drought crises. Current dam operations rely on rigid rule-based systems that cannot adapt to climate variability.

This environment models a realistic dam with:
- **5 controllable spillway gates** (water release without power generation)
- **3 hydroelectric turbines** (power generation with water release)
- **Seasonal inflow variation** (monsoon/dry/pre-monsoon/post-monsoon)
- **Random extreme weather events** (8% per step, tripled inflow)
- **3-day inflow forecast** (imperfect, with noise)
- **Downstream flood monitoring** (danger at 90 MCM/day)

---

## Quick Start

```bash
# Clone and run locally
git clone <your-repo-url>
cd dam-flood-control-env

pip install -r requirements.txt

# Start environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run inference (in another terminal)
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

```bash
# Or with Docker
docker build -t dam-env .
docker run -p 7860:7860 dam-env
```

---

## API Reference

### `POST /reset`
Initialize a new episode.

```json
{"task_name": "level_management", "seed": 42}
```

Returns `ResetResult` with initial `DamObservation`.

### `POST /step`
Execute one day of dam operations.

```json
{
  "gate_positions": [0.0, 0.0, 0.5, 0.8, 0.0],
  "turbine_active": [true, true, false],
  "turbine_flow_fraction": [0.8, 0.7, 0.0],
  "action_rationale": "Opening gates 3-4 to reduce high reservoir level"
}
```

Returns `StepResult` with `observation`, `reward`, `done`, `info`.

### `GET /state`
Returns current episode metadata (episode_id, step_count, risk level, etc.)

### `GET /tasks`
Lists all 3 tasks with descriptions.

---

## Action Space

| Field | Type | Shape | Range | Description |
|-------|------|-------|-------|-------------|
| `gate_positions` | float[] | (5,) | [0.0, 1.0] | Fraction open for each spillway gate. Each gate releases up to 20 MCM/day. |
| `turbine_active` | bool[] | (3,) | True/False | Whether each turbine generator is running |
| `turbine_flow_fraction` | float[] | (3,) | [0.0, 1.0] | Flow fraction through each active turbine. Max 8 MCM/day each. |

**Total possible release:** 5×20 + 3×8 = **124 MCM/day**

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `reservoir_level_mcm` | float | Water volume (MCM). Max: 500 MCM |
| `reservoir_fraction` | float | Fullness [0.0–1.0]. Safe: 0.4–0.8 |
| `reservoir_trend` | string | `rising` / `falling` / `stable` |
| `current_inflow_mcm` | float | Today's inflow (MCM/day) |
| `forecast_inflow_3day` | float[3] | Noisy 3-day inflow forecast |
| `season` | string | `monsoon` / `dry` / `pre_monsoon` / `post_monsoon` |
| `is_extreme_weather_event` | bool | 3× inflow event |
| `gate_positions` | float[5] | Current gate settings |
| `downstream_flow_mcm` | float | Total outflow (MCM/day) |
| `downstream_status` | string | `safe` / `warning` / `danger` / `catastrophe` |
| `flood_risk_level` | string | `low` / `medium` / `high` / `critical` |
| `downstream_flooded` | bool | Whether flooding is occurring |
| `power_output_mw` | float | Current power generation (MW) |
| `alerts` | string[] | Active system alerts |

---

## Tasks

### 🟢 Task 1: Reservoir Level Management (Easy)
**15 days, pre-monsoon, seed=42**

Keep the reservoir between 40–80% capacity. Inflow is moderate and predictable. No extreme events.

**Grader (score 0.0–1.0):**
- 60%: fraction of steps in safe zone (40–80%)
- 20%: no overflow events
- 20%: no downstream flooding

**Baseline score (heuristic agent):** ~0.72

---

### 🟡 Task 2: Flood Prevention During Monsoon (Medium)
**20 days, monsoon, seed=77, includes 1+ extreme events**

Heavy, variable inflow. Prevent downstream flooding while keeping the dam from overflowing. Extreme weather events force rapid response.

**Grader (score 0.0–1.0):**
- 50%: minimize downstream flooding (severity × duration)
- 25%: prevent dam overflow
- 25%: maintain safe reservoir level during monsoon

**Baseline score (heuristic agent):** ~0.54

---

### 🔴 Task 3: Full Multi-Objective Optimization (Hard)
**30 days, multi-season, seed=13, multiple extreme events possible**

Operate across seasonal transitions. Simultaneously maximize hydroelectric generation, prevent flooding, avoid drought, and minimize water waste.

**Grader (score 0.0–1.0):**
- 35%: flood prevention
- 25%: power generation (normalized to target 72,000 kWh)
- 20%: drought prevention
- 20%: operational efficiency (minimize wasted water)

**Baseline score (heuristic agent):** ~0.44

---

## Reward Function

Per-step reward in **[-1.0, +1.0]**:

```
reward = +0.40 × power_score          # Normalized power generation
       + 0.30 × level_score           # Reservoir in safe 40-80% zone
       - 0.50 × flood_penalty         # Downstream flooding severity
       - 0.30 × drought_penalty       # Reservoir critically low
       - 0.20 × spill_penalty         # Uncontrolled spill
```

**Key properties:**
- Dense reward at every step (not sparse)
- Partial progress signals (gradual flood penalty, level bands)
- Competing objectives create non-trivial optimization landscape

---

## Physics Model

| Parameter | Value |
|-----------|-------|
| Reservoir max capacity | 500 MCM |
| Safe zone | 40–80% |
| Critical zone | >95% |
| Gate max flow | 20 MCM/day each × 5 = 100 MCM/day |
| Turbine max flow | 8 MCM/day each × 3 = 24 MCM/day |
| Turbine min flow | 1 MCM/day (cavitation limit) |
| Downstream safe | 60 MCM/day |
| Downstream danger | 90 MCM/day |
| Evaporation rate | 0.2%/day |
| Extreme event prob | 8%/step |
| Monsoon inflow | ~45 MCM/day ± 20 |
| Dry inflow | ~5 MCM/day ± 2 |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | Yes | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | Yes | `Qwen/Qwen2.5-72B-Instruct` | LLM model identifier |
| `HF_TOKEN` | Yes | — | Hugging Face API key |
| `DAM_ENV_URL` | No | `http://localhost:7860` | Environment server URL |
| `DAM_TASK` | No | `all` | Which task to run: `all` / task name |

---

## Baseline Scores

Run `python inference.py` with any supported LLM:

| Task | Heuristic Agent | Random Agent |
|------|----------------|--------------|
| level_management | 0.72 | 0.31 |
| flood_prevention | 0.54 | 0.18 |
| full_optimization | 0.44 | 0.12 |

---

## Project Structure

```
dam-flood-control-env/
├── inference.py          # Mandatory baseline inference script
├── openenv.yaml          # OpenEnv spec metadata
├── Dockerfile            # Container build
├── requirements.txt
├── README.md
└── server/
    ├── __init__.py
    ├── app.py            # FastAPI HTTP server
    ├── environment.py    # OpenEnv Environment class
    ├── models.py         # Typed Pydantic models (Action/Observation/State)
    ├── tasks.py          # 3 tasks + graders
    └── dam_physics.py    # Hydrological simulation engine
```

---

## Deployment (Hugging Face Spaces)

This environment is deployed as a Docker Space on Hugging Face:

```
Space URL: https://YOUR_USERNAME-dam-flood-control.hf.space
Reset URL: https://YOUR_USERNAME-dam-flood-control.hf.space/reset
```

The Space responds to automated validation pings at `/reset` (POST) and `/health` (GET).

---

## Real-World Impact

This environment directly models challenges faced by dam operators worldwide:
- **India:** 5,000+ large dams, frequent monsoon management crises
- **Climate change:** Increasingly unpredictable precipitation patterns
- **Multi-objective:** Power revenue vs. public safety trade-offs

Training an RL agent on this environment could genuinely improve automated dam control systems.
