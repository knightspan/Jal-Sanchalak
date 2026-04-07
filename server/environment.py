"""
Dam Flood Control — OpenEnv Environment Implementation

Implements the full OpenEnv spec:
  - reset() → ResetResult
  - step(action) → StepResult
  - state() → DamState
"""

import uuid
from typing import Dict, List, Optional

from server.dam_physics import (
    DamPhysicsEngine, DamState as PhysicsState,
    DAM_MAX_CAPACITY_MCM, DOWNSTREAM_SAFE_FLOW_MCM,
    DOWNSTREAM_DANGER_FLOW_MCM, DOWNSTREAM_CATASTROPHE_MCM,
    NUM_GATES, NUM_TURBINES
)
from server.models import (
    DamAction, DamObservation, DamState as EnvState,
    StepResult, ResetResult
)
from server.tasks import TASK_CONFIGS, TaskGrader


class DamFloodControlEnvironment:
    """
    OpenEnv-compliant environment for dam flood control.

    Simulates a real hydroelectric dam with:
    - 5 controllable spillway gates
    - 3 turbine generators
    - Seasonal inflow variation
    - Random extreme weather events
    - Downstream flood monitoring
    """

    def __init__(self):
        self.engine = DamPhysicsEngine()
        self.grader = TaskGrader()
        self._episode_id: str = ""
        self._task_name: str = "level_management"
        self._episode_history: List[PhysicsState] = []
        self._cumulative_reward: float = 0.0
        self._step_count: int = 0
        self._physics_state: Optional[PhysicsState] = None

    def reset(self, task_name: str = "level_management", seed: Optional[int] = None) -> ResetResult:
        """Initialize a new episode. Returns initial observation."""
        if task_name not in TASK_CONFIGS:
            task_name = "level_management"

        config = TASK_CONFIGS[task_name]
        self._task_name = task_name
        self._episode_id = str(uuid.uuid4())
        self._episode_history = []
        self._cumulative_reward = 0.0
        self._step_count = 0

        # Initialize physics engine
        actual_seed = seed if seed is not None else config.seed
        self.engine = DamPhysicsEngine(seed=actual_seed)
        physics_state = self.engine.reset(
            seed=actual_seed,
            initial_level_fraction=config.initial_level_fraction
        )
        self._physics_state = physics_state
        self._episode_history.append(physics_state)

        obs = self._build_observation(physics_state)
        return ResetResult(
            observation=obs,
            episode_id=self._episode_id,
            task_name=task_name,
        )

    def step(self, action: DamAction) -> StepResult:
        """Execute an action and advance the simulation by 1 day."""
        config = TASK_CONFIGS[self._task_name]

        # Validate inputs
        gate_positions = action.gate_positions[:NUM_GATES]
        turbine_active = action.turbine_active[:NUM_TURBINES]
        turbine_flow = action.turbine_flow_fraction[:NUM_TURBINES]

        # Pad if short
        while len(gate_positions) < NUM_GATES:
            gate_positions.append(0.0)
        while len(turbine_active) < NUM_TURBINES:
            turbine_active.append(False)
        while len(turbine_flow) < NUM_TURBINES:
            turbine_flow.append(0.0)

        # Step physics
        physics_state, reward, done = self.engine.step(
            gate_positions=gate_positions,
            turbine_active=turbine_active,
            turbine_flow_fraction=turbine_flow,
        )

        # Override done with task max_steps
        self._step_count += 1
        if self._step_count >= config.max_steps:
            done = True

        self._physics_state = physics_state
        self._cumulative_reward += reward
        self._episode_history.append(physics_state)

        obs = self._build_observation(physics_state)

        info = {}
        if done:
            grade = self.grader.grade(self._task_name, self._episode_history)
            info["task_score"] = grade["score"]
            info["grade_breakdown"] = grade.get("breakdown", {})
            info["episode_summary"] = {
                "total_steps": self._step_count,
                "cumulative_reward": round(self._cumulative_reward, 3),
                "cumulative_flood_damage": round(physics_state.cumulative_flood_damage, 3),
                "power_generated_kwh": round(physics_state.cumulative_power_kwh, 1),
            }

        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=done,
            info=info,
        )

    def state(self) -> EnvState:
        """Return current episode state metadata."""
        ps = self._physics_state
        if ps is None:
            return EnvState(
                episode_id=self._episode_id,
                step_count=0,
                reservoir_level_mcm=0.0,
                reservoir_fraction=0.0,
                season="dry",
                flood_risk_level="low",
                downstream_flooded=False,
                drought_condition=False,
                cumulative_flood_damage=0.0,
                cumulative_power_kwh=0.0,
                total_water_wasted=0.0,
                power_output_mw=0.0,
                downstream_flow_mcm=0.0,
                episode_done=False,
                task_name=self._task_name,
            )

        return EnvState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            reservoir_level_mcm=round(ps.reservoir_level_mcm, 2),
            reservoir_fraction=round(ps.reservoir_fraction, 4),
            season=ps.season,
            flood_risk_level=ps.flood_risk_level,
            downstream_flooded=ps.downstream_flooded,
            drought_condition=ps.drought_condition,
            cumulative_flood_damage=round(ps.cumulative_flood_damage, 4),
            cumulative_power_kwh=round(ps.cumulative_power_kwh, 2),
            total_water_wasted=round(ps.total_water_wasted, 2),
            power_output_mw=round(ps.power_output_mw, 2),
            downstream_flow_mcm=round(ps.downstream_flow_mcm, 2),
            episode_done=ps.episode_done,
            task_name=self._task_name,
        )

    def grade_episode(self) -> Dict:
        """Grade the completed episode."""
        return self.grader.grade(self._task_name, self._episode_history)

    def _build_observation(self, ps: PhysicsState) -> DamObservation:
        """Convert physics state to typed observation."""
        # Generate 3-day inflow forecast (with noise)
        forecast = []
        for i in range(1, 4):
            future_inflow = self.engine._generate_inflow(ps.step + i)
            # Add forecast uncertainty (±20%)
            noisy = future_inflow * (0.8 + 0.4 * self.engine.rng.random())
            forecast.append(round(noisy, 2))

        # Reservoir trend
        if len(self._episode_history) >= 2:
            prev = self._episode_history[-2].reservoir_level_mcm
            curr = ps.reservoir_level_mcm
            if curr > prev + 1.0:
                trend = "rising"
            elif curr < prev - 1.0:
                trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Downstream status
        df = ps.downstream_flow_mcm
        if df >= DOWNSTREAM_CATASTROPHE_MCM:
            ds_status = "catastrophe"
        elif df >= DOWNSTREAM_DANGER_FLOW_MCM:
            ds_status = "danger"
        elif df >= DOWNSTREAM_SAFE_FLOW_MCM * 0.8:
            ds_status = "warning"
        else:
            ds_status = "safe"

        # Build alerts
        alerts = []
        if ps.flood_risk_level in ["high", "critical"]:
            alerts.append(f"ALERT: Reservoir at {ps.reservoir_fraction:.0%} capacity — {ps.flood_risk_level} flood risk")
        if ps.downstream_flooded:
            alerts.append(f"ALERT: Downstream flooding! Flow={ps.downstream_flow_mcm:.1f} MCM/day")
        if ps.is_extreme_event:
            alerts.append("ALERT: Extreme weather event — inflow tripled this step")
        if ps.drought_condition:
            alerts.append("ALERT: Drought condition — reservoir critically low")

        spillway_flow = sum(g * 20.0 for g in ps.gate_positions)

        return DamObservation(
            reservoir_level_mcm=round(ps.reservoir_level_mcm, 2),
            reservoir_fraction=round(ps.reservoir_fraction, 4),
            reservoir_trend=trend,
            current_inflow_mcm=round(ps.inflow_mcm, 2),
            forecast_inflow_3day=forecast,
            season=ps.season,
            is_extreme_weather_event=ps.is_extreme_event,
            gate_positions=[round(g, 3) for g in ps.gate_positions],
            turbine_status=ps.turbine_active,
            turbine_flow_mcm=[round(f, 2) for f in ps.turbine_flow],
            spillway_flow_mcm=round(spillway_flow, 2),
            turbine_total_flow_mcm=round(sum(ps.turbine_flow), 2),
            downstream_flow_mcm=round(ps.downstream_flow_mcm, 2),
            downstream_status=ds_status,
            power_output_mw=round(ps.power_output_mw, 2),
            cumulative_power_kwh=round(ps.cumulative_power_kwh, 1),
            flood_risk_level=ps.flood_risk_level,
            downstream_flooded=ps.downstream_flooded,
            drought_condition=ps.drought_condition,
            step=ps.step,
            cumulative_flood_damage=round(ps.cumulative_flood_damage, 4),
            episode_reward_so_far=round(self._cumulative_reward, 3),
            alerts=alerts,
        )
