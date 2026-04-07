"""
Dam Physics Engine — realistic simulation of a hydroelectric dam system.
Models water inflow, reservoir level, gate control, turbine output, and downstream flooding.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─── Physical Constants ────────────────────────────────────────────────────────

DAM_MAX_CAPACITY_MCM = 500.0       # Million cubic meters (MCM)
DAM_SAFE_LEVEL_FRACTION = 0.85     # Above this → danger zone
DAM_CRITICAL_LEVEL_FRACTION = 0.95 # Above this → imminent flood risk
DAM_MIN_LEVEL_FRACTION = 0.10      # Below this → drought risk

NUM_GATES = 5                       # Spillway gates
GATE_MAX_FLOW_MCM_PER_STEP = 20.0  # Each gate can release up to 20 MCM/step
NUM_TURBINES = 3
TURBINE_MAX_FLOW_MCM_PER_STEP = 8.0
TURBINE_MIN_FLOW_MCM_PER_STEP = 1.0

DOWNSTREAM_SAFE_FLOW_MCM = 60.0    # Safe downstream channel capacity
DOWNSTREAM_DANGER_FLOW_MCM = 90.0  # Flooding begins
DOWNSTREAM_CATASTROPHE_MCM = 120.0 # Severe flooding

EVAPORATION_RATE = 0.002            # Fraction per step
SEEPAGE_RATE = 0.001


# ─── Seasonal Inflow Profiles ─────────────────────────────────────────────────

SEASONAL_INFLOW = {
    # step → base inflow MCM/step (24hr period)
    "monsoon":    {"base": 45.0, "variance": 20.0},   # Heavy rain
    "pre_monsoon":{"base": 15.0, "variance": 8.0},
    "post_monsoon":{"base": 20.0, "variance": 10.0},
    "dry":        {"base": 5.0,  "variance": 2.0},
}

EXTREME_EVENT_PROBABILITY = 0.08   # 8% chance each step


@dataclass
class DamState:
    """Complete physical state of the dam system."""
    step: int = 0
    reservoir_level_mcm: float = 250.0      # Current water volume
    inflow_mcm: float = 15.0               # Inflow this step
    gate_positions: List[float] = field(default_factory=lambda: [0.0]*NUM_GATES)
    turbine_active: List[bool] = field(default_factory=lambda: [False]*NUM_TURBINES)
    turbine_flow: List[float] = field(default_factory=lambda: [0.0]*NUM_TURBINES)
    downstream_flow_mcm: float = 0.0
    power_output_mw: float = 0.0
    season: str = "pre_monsoon"
    is_extreme_event: bool = False
    flood_risk_level: str = "low"          # low / medium / high / critical
    downstream_flooded: bool = False
    drought_condition: bool = False
    cumulative_flood_damage: float = 0.0
    cumulative_power_kwh: float = 0.0
    total_water_wasted: float = 0.0        # spilled without generation
    episode_done: bool = False

    @property
    def reservoir_fraction(self) -> float:
        return self.reservoir_level_mcm / DAM_MAX_CAPACITY_MCM

    @property
    def safe_level_mcm(self) -> float:
        return DAM_MAX_CAPACITY_MCM * DAM_SAFE_LEVEL_FRACTION

    @property
    def critical_level_mcm(self) -> float:
        return DAM_MAX_CAPACITY_MCM * DAM_CRITICAL_LEVEL_FRACTION


class DamPhysicsEngine:
    """
    Simulates dam hydrology and flood dynamics.
    Step = 1 day in simulation time.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.state = DamState()
        self._step_count = 0

    def reset(self, seed: Optional[int] = None, initial_level_fraction: float = 0.5) -> DamState:
        if seed is not None:
            self.rng = random.Random(seed)
        self._step_count = 0
        self.state = DamState(
            reservoir_level_mcm=DAM_MAX_CAPACITY_MCM * initial_level_fraction,
            season=self._get_season(0),
        )
        self.state.inflow_mcm = self._generate_inflow(0)
        return self.state

    def step(
        self,
        gate_positions: List[float],   # 0.0–1.0 each
        turbine_active: List[bool],
        turbine_flow_fraction: List[float],  # 0.0–1.0 each turbine
    ) -> Tuple[DamState, float, bool]:
        """
        Advance simulation by one step (1 day).
        Returns: (new_state, reward, done)
        """
        s = self.state
        self._step_count += 1

        # ── Clamp inputs ──────────────────────────────────────────────────────
        gate_positions = [max(0.0, min(1.0, g)) for g in gate_positions]
        turbine_flow_fraction = [max(0.0, min(1.0, f)) for f in turbine_flow_fraction]

        # ── Calculate outflows ────────────────────────────────────────────────
        spillway_flow = sum(g * GATE_MAX_FLOW_MCM_PER_STEP for g in gate_positions)

        turbine_flows = []
        for i, (active, frac) in enumerate(zip(turbine_active, turbine_flow_fraction)):
            if active:
                flow = max(TURBINE_MIN_FLOW_MCM_PER_STEP,
                           frac * TURBINE_MAX_FLOW_MCM_PER_STEP)
                turbine_flows.append(flow)
            else:
                turbine_flows.append(0.0)

        turbine_total = sum(turbine_flows)
        total_outflow = spillway_flow + turbine_total

        # ── Update reservoir ──────────────────────────────────────────────────
        evaporation = s.reservoir_level_mcm * EVAPORATION_RATE
        seepage = s.reservoir_level_mcm * SEEPAGE_RATE
        new_level = s.reservoir_level_mcm + s.inflow_mcm - total_outflow - evaporation - seepage

        # Overflow check — uncontrolled spill
        uncontrolled_spill = 0.0
        if new_level > DAM_MAX_CAPACITY_MCM:
            uncontrolled_spill = new_level - DAM_MAX_CAPACITY_MCM
            new_level = DAM_MAX_CAPACITY_MCM

        # Dam can't go negative
        if new_level < 0:
            # Demand exceeds supply — truncate outflow
            actual_total_outflow = max(0.0, s.reservoir_level_mcm + s.inflow_mcm - evaporation - seepage)
            spillway_flow = spillway_flow / (total_outflow + 1e-9) * actual_total_outflow
            turbine_total = turbine_total / (total_outflow + 1e-9) * actual_total_outflow
            total_outflow = actual_total_outflow
            new_level = 0.0

        # ── Power generation ──────────────────────────────────────────────────
        # Power ∝ flow × head (water height above turbine)
        head_m = (new_level / DAM_MAX_CAPACITY_MCM) * 80.0  # Max 80m head
        power_mw = turbine_total * head_m * 0.85 * 9.81 / 3.6  # simplified
        power_mw = min(power_mw, 150.0)  # rated capacity cap

        # ── Downstream flow ───────────────────────────────────────────────────
        downstream_flow = spillway_flow + turbine_total + uncontrolled_spill
        flooded = downstream_flow > DOWNSTREAM_DANGER_FLOW_MCM

        # ── Flood risk classification ─────────────────────────────────────────
        frac = new_level / DAM_MAX_CAPACITY_MCM
        if frac >= DAM_CRITICAL_LEVEL_FRACTION:
            risk = "critical"
        elif frac >= DAM_SAFE_LEVEL_FRACTION:
            risk = "high"
        elif frac >= 0.60:
            risk = "medium"
        else:
            risk = "low"

        drought = frac < DAM_MIN_LEVEL_FRACTION

        # ── Flood damage accumulation ─────────────────────────────────────────
        flood_damage = 0.0
        if flooded:
            excess = downstream_flow - DOWNSTREAM_DANGER_FLOW_MCM
            flood_damage = min(1.0, excess / (DOWNSTREAM_CATASTROPHE_MCM - DOWNSTREAM_DANGER_FLOW_MCM))

        # ── Generate next step inflow ─────────────────────────────────────────
        next_season = self._get_season(self._step_count)
        next_inflow = self._generate_inflow(self._step_count)
        extreme = self._is_extreme_event()
        if extreme:
            next_inflow *= 3.0  # Triple inflow during extreme event

        # ── Build new state ───────────────────────────────────────────────────
        new_state = DamState(
            step=self._step_count,
            reservoir_level_mcm=new_level,
            inflow_mcm=next_inflow,
            gate_positions=gate_positions,
            turbine_active=turbine_active,
            turbine_flow=turbine_flows,
            downstream_flow_mcm=downstream_flow,
            power_output_mw=power_mw,
            season=next_season,
            is_extreme_event=extreme,
            flood_risk_level=risk,
            downstream_flooded=flooded,
            drought_condition=drought,
            cumulative_flood_damage=s.cumulative_flood_damage + flood_damage,
            cumulative_power_kwh=s.cumulative_power_kwh + power_mw * 24,
            total_water_wasted=s.total_water_wasted + uncontrolled_spill + spillway_flow * 0.3,
            episode_done=False,
        )

        # ── Compute reward ────────────────────────────────────────────────────
        reward = self._compute_reward(new_state, downstream_flow, power_mw, flood_damage, uncontrolled_spill, drought)

        # ── Episode termination ───────────────────────────────────────────────
        max_steps = 30  # 30-day episode
        done = (
            self._step_count >= max_steps or
            new_state.cumulative_flood_damage >= 5.0 or  # catastrophic
            drought and frac < 0.02                       # total drought
        )
        new_state.episode_done = done

        self.state = new_state
        return new_state, reward, done

    def _compute_reward(
        self,
        state: DamState,
        downstream_flow: float,
        power_mw: float,
        flood_damage: float,
        uncontrolled_spill: float,
        drought: bool,
    ) -> float:
        """
        Multi-objective reward:
        +reward for power generation (economic benefit)
        +reward for safe reservoir level
        -penalty for flooding
        -penalty for drought
        -penalty for uncontrolled spill
        All normalized to approximately [-1, +1] per step.
        """
        r = 0.0

        # 1. Power generation reward (0 to +0.4)
        power_norm = min(power_mw / 150.0, 1.0)
        r += 0.4 * power_norm

        # 2. Reservoir level reward — prefer 40–80% range (+0.3)
        frac = state.reservoir_fraction
        if 0.40 <= frac <= 0.80:
            level_score = 1.0
        elif 0.25 <= frac < 0.40 or 0.80 < frac <= 0.90:
            level_score = 0.5
        elif frac < 0.25 or 0.90 < frac <= 0.95:
            level_score = 0.1
        else:
            level_score = 0.0
        r += 0.3 * level_score

        # 3. Downstream flow penalty (-0.0 to -0.5)
        if downstream_flow <= DOWNSTREAM_SAFE_FLOW_MCM:
            r += 0.0  # no penalty
        elif downstream_flow <= DOWNSTREAM_DANGER_FLOW_MCM:
            excess_frac = (downstream_flow - DOWNSTREAM_SAFE_FLOW_MCM) / (DOWNSTREAM_DANGER_FLOW_MCM - DOWNSTREAM_SAFE_FLOW_MCM)
            r -= 0.2 * excess_frac
        else:
            excess_frac = min(1.0, (downstream_flow - DOWNSTREAM_DANGER_FLOW_MCM) / (DOWNSTREAM_CATASTROPHE_MCM - DOWNSTREAM_DANGER_FLOW_MCM))
            r -= 0.5 * (1 + excess_frac)

        # 4. Drought penalty (-0.3)
        if drought:
            r -= 0.3

        # 5. Uncontrolled spill penalty (-0.2 per MCM)
        r -= min(0.2, uncontrolled_spill / 50.0)

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, r))

    def _get_season(self, step: int) -> str:
        month = (step // 5) % 12  # 5 steps per "month" 
        if 5 <= month <= 9:
            return "monsoon"
        elif month in [3, 4]:
            return "pre_monsoon"
        elif month in [10, 11]:
            return "post_monsoon"
        else:
            return "dry"

    def _generate_inflow(self, step: int) -> float:
        season = self._get_season(step)
        profile = SEASONAL_INFLOW[season]
        base = profile["base"]
        var = profile["variance"]
        # Lognormal inflow (realistic)
        mu = math.log(base)
        sigma = var / base * 0.5
        return max(0.5, self.rng.lognormvariate(mu, sigma))

    def _is_extreme_event(self) -> bool:
        return self.rng.random() < EXTREME_EVENT_PROBABILITY
