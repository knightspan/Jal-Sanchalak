"""
inference.py — Dam Flood Control OpenEnv
=========================================
Mandatory variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    DAM_ENV_URL    Your HF Space URL (default: localhost:7860)

Stdout format:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import json
import requests
from typing import List, Optional
from openai import OpenAI

# ── Environment variables ──────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN")
DAM_ENV_URL  = os.getenv("DAM_ENV_URL",  "http://localhost:7860")
BENCHMARK    = "dam-flood-control"

# ── Tasks to run ───────────────────────────────────────────────────────────────
TASKS = [
    {"name": "level_management",  "max_steps": 15, "seed": 42},
    {"name": "flood_prevention",  "max_steps": 20, "seed": 77},
    {"name": "full_optimization", "max_steps": 30, "seed": 13},
]

SUCCESS_THRESHOLD = 0.5  # score >= this = success

# ── Logging helpers ────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── Environment API helpers ────────────────────────────────────────────────────
def env_reset(task_name: str, seed: int) -> dict:
    resp = requests.post(
        f"{DAM_ENV_URL}/reset",
        json={"task_name": task_name, "seed": seed},
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()

def env_step(action: dict) -> dict:
    resp = requests.post(
        f"{DAM_ENV_URL}/step",
        json=action,
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()

def env_state() -> dict:
    resp = requests.get(f"{DAM_ENV_URL}/state", timeout=30)
    resp.raise_for_status()
    return resp.json()

# ── LLM prompt ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI agent controlling a hydroelectric dam.
Your goal is to prevent floods, generate electricity, and maintain safe reservoir levels.

You control:
- gate_positions: list of 5 floats [0.0-1.0] — spillway gate openings (each releases up to 20 MCM/day)
- turbine_active: list of 3 bools — whether each turbine is running
- turbine_flow_fraction: list of 3 floats [0.0-1.0] — flow through each turbine (max 8 MCM/day each)

Rules:
- If reservoir_fraction > 0.80: open gates more, activate turbines
- If reservoir_fraction < 0.40: close gates, reduce turbine flow
- If downstream_status is 'danger' or 'catastrophe': reduce ALL outflow immediately
- If is_extreme_weather_event is true: pre-emptively open gates

Respond with ONLY a valid JSON object, no explanation:
{
  "gate_positions": [0.0, 0.0, 0.0, 0.0, 0.0],
  "turbine_active": [true, true, false],
  "turbine_flow_fraction": [0.8, 0.7, 0.0],
  "action_rationale": "brief reason"
}"""

def build_user_prompt(obs: dict, step: int) -> str:
    return f"""Step {step} — Current dam state:
- Reservoir: {obs.get('reservoir_fraction', 0)*100:.1f}% full ({obs.get('reservoir_level_mcm', 0):.1f} MCM)
- Trend: {obs.get('reservoir_trend', 'unknown')}
- Inflow today: {obs.get('current_inflow_mcm', 0):.1f} MCM/day
- 3-day forecast: {obs.get('forecast_inflow_3day', [])}
- Season: {obs.get('season', 'unknown')}
- Extreme weather: {obs.get('is_extreme_weather_event', False)}
- Downstream status: {obs.get('downstream_status', 'unknown')}
- Flood risk: {obs.get('flood_risk_level', 'unknown')}
- Power output: {obs.get('power_output_mw', 0):.1f} MW
- Alerts: {obs.get('alerts', [])}

Respond with JSON action only."""

def get_action(client: OpenAI, obs: dict, step: int) -> dict:
    """Ask LLM for next action, with fallback if parsing fails."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(obs, step)},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if present
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        action = json.loads(text)

        # Validate and clamp values
        gate_positions        = [max(0.0, min(1.0, float(x))) for x in action.get("gate_positions", [0.0]*5)]
        turbine_active        = [bool(x) for x in action.get("turbine_active", [True, True, False])]
        turbine_flow_fraction = [max(0.0, min(1.0, float(x))) for x in action.get("turbine_flow_fraction", [0.8, 0.7, 0.0])]
        rationale             = action.get("action_rationale", "LLM action")

        return {
            "gate_positions":        gate_positions,
            "turbine_active":        turbine_active,
            "turbine_flow_fraction": turbine_flow_fraction,
            "action_rationale":      rationale,
        }

    except Exception as e:
        # Fallback: safe default action based on reservoir level
        fraction = obs.get("reservoir_fraction", 0.5)
        if fraction > 0.75:
            gates   = [0.3, 0.3, 0.3, 0.3, 0.3]
            turbines = [True, True, True]
            flows    = [0.8, 0.8, 0.8]
        elif fraction < 0.45:
            gates   = [0.0, 0.0, 0.0, 0.0, 0.0]
            turbines = [True, False, False]
            flows    = [0.5, 0.0, 0.0]
        else:
            gates   = [0.1, 0.1, 0.0, 0.0, 0.0]
            turbines = [True, True, False]
            flows    = [0.7, 0.6, 0.0]

        return {
            "gate_positions":        gates,
            "turbine_active":        turbines,
            "turbine_flow_fraction": flows,
            "action_rationale":      f"fallback ({e})",
        }

# ── Action → compact string for logging ───────────────────────────────────────
def action_to_str(action: dict) -> str:
    gates   = [f"{g:.2f}" for g in action["gate_positions"]]
    turbines = [str(t)[0] for t in action["turbine_active"]]   # T/F
    flows    = [f"{f:.2f}" for f in action["turbine_flow_fraction"]]
    return f"gates=[{','.join(gates)}] turbines=[{''.join(turbines)}] flows=[{','.join(flows)}]"

# ── Run one task episode ───────────────────────────────────────────────────────
def run_task(client: OpenAI, task: dict) -> None:
    task_name = task["name"]
    max_steps = task["max_steps"]
    seed      = task["seed"]

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        reset_result = env_reset(task_name, seed)
        obs          = reset_result.get("observation", reset_result)
        done         = reset_result.get("done", False)

        for step in range(1, max_steps + 1):
            if done:
                break

            action      = get_action(client, obs, step)
            action_str  = action_to_str(action)
            error_msg   = None

            try:
                step_result = env_step(action)
                obs         = step_result.get("observation", {})
                reward      = float(step_result.get("reward", 0.0))
                done        = bool(step_result.get("done", False))
                info        = step_result.get("info", {})
                error_msg   = info.get("error") if info else None

                # Get final score from info if available
                if done and "score" in (info or {}):
                    score = float(info["score"])

            except Exception as e:
                reward    = 0.0
                done      = True
                error_msg = str(e)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        # Compute final score
        if score == 0.0 and rewards:
            # Normalize: reward range is [-1, 1], map to [0, 1]
            avg_reward = sum(rewards) / len(rewards)
            score      = max(0.0, min(1.0, (avg_reward + 1.0) / 2.0))

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_name} failed: {e}", flush=True)
        if not rewards:
            rewards = [0.0]
        steps_taken = steps_taken or 1

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Check which tasks to run
    run_task_name = os.getenv("DAM_TASK", "all")

    for task in TASKS:
        if run_task_name == "all" or run_task_name == task["name"]:
            run_task(client, task)

if __name__ == "__main__":
    main()