"""
OpenEnv typed models for the Dam Flood Control environment.
All models use Pydantic for spec-compliant serialization.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ─── Action Model ──────────────────────────────────────────────────────────────

class DamAction(BaseModel):
    """
    Action space for the Dam Flood Control environment.

    The agent controls:
      - gate_positions: 5 spillway gates, each 0.0 (closed) to 1.0 (fully open)
      - turbine_active: which of 3 turbines to run
      - turbine_flow_fraction: flow rate for each turbine (0.0–1.0 of max capacity)

    Example (emergency release):
      gate_positions=[0.8, 0.8, 0.8, 0.5, 0.5]
      turbine_active=[True, True, True]
      turbine_flow_fraction=[1.0, 1.0, 0.8]
    """
    gate_positions: List[float] = Field(
        default=[0.0, 0.0, 0.0, 0.0, 0.0],
        description="Spillway gate openings [0.0=closed, 1.0=fully open] for each of 5 gates",
        min_length=5, max_length=5
    )
    turbine_active: List[bool] = Field(
        default=[False, False, False],
        description="Whether each of 3 turbines is operational",
        min_length=3, max_length=3
    )
    turbine_flow_fraction: List[float] = Field(
        default=[0.0, 0.0, 0.0],
        description="Flow fraction [0.0–1.0] through each turbine if active",
        min_length=3, max_length=3
    )
    action_rationale: Optional[str] = Field(
        default=None,
        description="Optional explanation of why this action was chosen (for logging)"
    )


# ─── Observation Model ─────────────────────────────────────────────────────────

class DamObservation(BaseModel):
    """
    Observation returned after each step.
    Contains full sensor readings from the dam system.
    """
    # Reservoir
    reservoir_level_mcm: float = Field(description="Current water volume in reservoir (Million Cubic Meters)")
    reservoir_fraction: float = Field(description="Reservoir fullness fraction [0.0–1.0]")
    reservoir_trend: str = Field(description="'rising', 'falling', or 'stable'")

    # Inflow
    current_inflow_mcm: float = Field(description="Water flowing into reservoir this step (MCM/day)")
    forecast_inflow_3day: List[float] = Field(description="Forecasted inflow for next 3 days (MCM/day)")
    season: str = Field(description="Current season: monsoon/pre_monsoon/post_monsoon/dry")
    is_extreme_weather_event: bool = Field(description="Whether an extreme rainfall event is occurring")

    # Gate and turbine status
    gate_positions: List[float] = Field(description="Current position of each spillway gate [0.0–1.0]")
    turbine_status: List[bool] = Field(description="Active/inactive status of each turbine")
    turbine_flow_mcm: List[float] = Field(description="Actual flow through each turbine (MCM)")

    # Outflow and downstream
    spillway_flow_mcm: float = Field(description="Total flow through spillway gates (MCM)")
    turbine_total_flow_mcm: float = Field(description="Total flow through all turbines (MCM)")
    downstream_flow_mcm: float = Field(description="Total downstream river flow (MCM)")
    downstream_status: str = Field(description="'safe', 'warning', 'danger', or 'catastrophe'")

    # Power
    power_output_mw: float = Field(description="Current electricity generation (MW)")
    cumulative_power_kwh: float = Field(description="Total energy generated this episode (kWh)")

    # Risk
    flood_risk_level: str = Field(description="'low', 'medium', 'high', or 'critical'")
    downstream_flooded: bool = Field(description="Whether downstream areas are currently flooded")
    drought_condition: bool = Field(description="Whether reservoir is critically low")

    # Episode info
    step: int = Field(description="Current simulation day")
    cumulative_flood_damage: float = Field(description="Accumulated flood damage score this episode")
    episode_reward_so_far: float = Field(description="Sum of rewards received so far")

    # Alerts
    alerts: List[str] = Field(default=[], description="Active system alerts")


# ─── State Model ──────────────────────────────────────────────────────────────

class DamState(BaseModel):
    """
    Internal episode state (returned by state() endpoint).
    """
    episode_id: str
    step_count: int
    reservoir_level_mcm: float
    reservoir_fraction: float
    season: str
    flood_risk_level: str
    downstream_flooded: bool
    drought_condition: bool
    cumulative_flood_damage: float
    cumulative_power_kwh: float
    total_water_wasted: float
    power_output_mw: float
    downstream_flow_mcm: float
    episode_done: bool
    task_name: Optional[str] = None


# ─── Step Result ───────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: DamObservation
    reward: float = Field(description="Step reward in [-1.0, 1.0]")
    done: bool
    info: dict = Field(default={})


# ─── Reset Result ──────────────────────────────────────────────────────────────

class ResetResult(BaseModel):
    observation: DamObservation
    episode_id: str
    task_name: str
