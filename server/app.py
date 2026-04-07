"""
FastAPI server for Dam Flood Control OpenEnv environment.
Exposes: POST /reset, POST /step, GET /state, GET /tasks, GET /health
"""

from server.tasks import TASK_CONFIGS
from server.models import DamAction, StepResult, ResetResult, DamState
from server.environment import DamFloodControlEnvironment
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI, HTTPException, Request
from typing import Optional
import os
import sys

sys.path.insert(0, "/app")


# ─── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Dam Flood Control — OpenEnv",
    description=(
        "An RL environment where an AI agent learns to operate a hydroelectric "
        "dam to prevent flooding, generate power, and avoid drought."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment instance (stateless per session via episode_id)
env = DamFloodControlEnvironment()


# ─── Request/Response Schemas ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: Optional[str] = "level_management"
    seed: Optional[int] = None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — judges ping this."""
    return {"status": "ok", "environment": "dam-flood-control", "version": "1.0.0"}


@app.post("/reset")
async def reset(request: ResetRequest = None) -> dict:
    """
    Initialize a new episode.
    task_name: 'level_management' | 'flood_prevention' | 'full_optimization'
    """
    if request is None:
        request = ResetRequest()
    task = request.task_name or "level_management"
    if task not in TASK_CONFIGS:
        raise HTTPException(
            status_code=400, detail=f"Unknown task: {task}. Valid: {list(TASK_CONFIGS.keys())}")
    result = env.reset(task_name=task, seed=request.seed)
    return result.model_dump()


@app.post("/step")
async def step(action: DamAction) -> dict:
    """Execute one simulation step (1 day) with the given gate/turbine action."""
    result = env.step(action)
    return result.model_dump()


@app.get("/state")
async def state() -> dict:
    """Return current episode state metadata."""
    return env.state().model_dump()


@app.get("/tasks")
async def list_tasks() -> dict:
    """List all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "name": cfg.name,
                "difficulty": cfg.difficulty,
                "description": cfg.description,
                "max_steps": cfg.max_steps,
                "success_threshold": cfg.success_threshold,
            }
            for cfg in TASK_CONFIGS.values()
        ]
    }


@app.get("/grade")
async def grade_current_episode() -> dict:
    """Grade the current episode (call after done=True)."""
    result = env.grade_episode()
    return result


@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple HTML landing page for HF Spaces."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dam Flood Control — OpenEnv</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #0f172a; color: #e2e8f0; }
            h1 { color: #38bdf8; }
            h2 { color: #7dd3fc; }
            code { background: #1e293b; padding: 2px 6px; border-radius: 4px; color: #a5f3fc; }
            .task { background: #1e293b; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #0ea5e9; }
            .easy { border-color: #22c55e; }
            .medium { border-color: #f59e0b; }
            .hard { border-color: #ef4444; }
            a { color: #38bdf8; }
        </style>
    </head>
    <body>
        <h1>🌊 Dam Flood Control — OpenEnv Environment</h1>
        <p>A real-world RL environment where an AI agent operates a hydroelectric dam to prevent flooding, generate electricity, and avoid drought.</p>

        <h2>API Endpoints</h2>
        <ul>
            <li><code>POST /reset</code> — Start a new episode</li>
            <li><code>POST /step</code> — Execute a gate/turbine action</li>
            <li><code>GET /state</code> — Current episode state</li>
            <li><code>GET /tasks</code> — List all tasks</li>
            <li><code>GET /grade</code> — Grade current episode</li>
            <li><a href="/docs">📖 Full API docs (Swagger)</a></li>
        </ul>

        <h2>Tasks</h2>
        <div class="task easy"><strong>🟢 Easy: Reservoir Level Management</strong><br>Keep the reservoir between 40-80% for 15 days during pre-monsoon season.</div>
        <div class="task medium"><strong>🟡 Medium: Flood Prevention During Monsoon</strong><br>Manage peak monsoon for 20 days. Prevent downstream flooding with heavy, variable inflow.</div>
        <div class="task hard"><strong>🔴 Hard: Full Multi-Objective Optimization</strong><br>30-day full simulation. Maximize power, prevent flooding AND drought, manage across seasons.</div>

        <h2>Action Space</h2>
        <p>5 spillway gates [0.0–1.0] + 3 turbines (on/off + flow rate)</p>

        <h2>Observation Space</h2>
        <p>Reservoir level, inflow forecast, gate status, downstream flow, power output, risk level, alerts</p>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
