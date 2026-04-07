"""
Three graded tasks for the Dam Flood Control environment.

Task 1 (Easy):   Reservoir Level Management
Task 2 (Medium): Flood Prevention During Monsoon
Task 3 (Hard):   Multi-Objective Optimization (flood safety + power + drought)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from server.dam_physics import DamState, DAM_MAX_CAPACITY_MCM, DOWNSTREAM_DANGER_FLOW_MCM


@dataclass
class TaskConfig:
    name: str
    description: str
    difficulty: str
    max_steps: int
    initial_level_fraction: float
    season_override: Optional[str]
    seed: int
    success_threshold: float


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "level_management": TaskConfig(
        name="level_management",
        description=(
            "Keep the reservoir between 40% and 80% capacity for 15 days during "
            "pre-monsoon season. Avoid overflow and dangerous low levels. "
            "Inflow is moderate and predictable."
        ),
        difficulty="easy",
        max_steps=15,
        initial_level_fraction=0.75,
        season_override="pre_monsoon",
        seed=42,
        success_threshold=0.6,
    ),
    "flood_prevention": TaskConfig(
        name="flood_prevention",
        description=(
            "Manage a dam during peak monsoon for 20 days. Prevent downstream "
            "flooding while keeping the dam from overflowing. Heavy, variable "
            "inflow with one extreme weather event. "
            "Partial credit for limiting flood duration and severity."
        ),
        difficulty="medium",
        max_steps=20,
        initial_level_fraction=0.65,
        season_override="monsoon",
        seed=77,
        success_threshold=0.5,
    ),
    "full_optimization": TaskConfig(
        name="full_optimization",
        description=(
            "Operate the dam across a full 30-day simulation spanning multiple "
            "seasons. Maximize hydroelectric generation while preventing both "
            "flooding AND drought. Multiple extreme events possible. "
            "Requires planning ahead using inflow forecasts."
        ),
        difficulty="hard",
        max_steps=30,
        initial_level_fraction=0.55,
        season_override=None,  # dynamic seasonal change
        seed=13,
        success_threshold=0.45,
    ),
}


class TaskGrader:
    """
    Grades agent performance on each task.
    All scores are in [0.0, 1.0].
    """

    def grade_level_management(self, episode_history: List[DamState]) -> Dict:
        """
        Easy task grader.

        Scoring:
        - 60% of score: fraction of steps where reservoir is in safe zone (40–80%)
        - 20%: no overflow events
        - 20%: no uncontrolled downstream flooding
        """
        if not episode_history:
            return {"score": 0.0, "breakdown": {}}

        steps = len(episode_history)
        safe_zone_steps = sum(
            1 for s in episode_history
            if 0.40 <= s.reservoir_fraction <= 0.80
        )
        overflow_steps = sum(
            1 for s in episode_history
            if s.reservoir_fraction >= 0.95
        )
        flood_steps = sum(
            1 for s in episode_history
            if s.downstream_flooded
        )

        safe_score = safe_zone_steps / steps
        overflow_score = max(0.0, 1.0 - overflow_steps / max(steps, 1))
        flood_score = max(0.0, 1.0 - flood_steps / max(steps, 1))

        total = 0.60 * safe_score + 0.20 * overflow_score + 0.20 * flood_score
        return {
            "score": round(min(1.0, max(0.0, total)), 4),
            "breakdown": {
                "safe_zone_fraction": round(safe_score, 3),
                "overflow_free_score": round(overflow_score, 3),
                "flood_free_score": round(flood_score, 3),
            },
        }

    def grade_flood_prevention(self, episode_history: List[DamState]) -> Dict:
        """
        Medium task grader.

        Scoring:
        - 50%: minimize downstream flooding (severity + duration)
        - 25%: prevent dam overflow (uncontrolled spill)
        - 25%: maintain safe reservoir level during event
        """
        if not episode_history:
            return {"score": 0.0, "breakdown": {}}

        steps = len(episode_history)
        total_flood_damage = episode_history[-1].cumulative_flood_damage
        overflow_fraction = sum(
            1 for s in episode_history
            if s.reservoir_fraction >= 0.98
        ) / max(steps, 1)

        # Flood damage normalized — target is keeping below 0.5 total damage
        flood_score = max(0.0, 1.0 - total_flood_damage / 3.0)

        # Overflow score
        overflow_score = max(0.0, 1.0 - overflow_fraction * 3.0)

        # Level management during monsoon — ideal is staying 50–85%
        managed_steps = sum(
            1 for s in episode_history
            if 0.50 <= s.reservoir_fraction <= 0.85
        )
        level_score = managed_steps / max(steps, 1)

        total = 0.50 * flood_score + 0.25 * overflow_score + 0.25 * level_score
        return {
            "score": round(min(1.0, max(0.0, total)), 4),
            "breakdown": {
                "flood_mitigation_score": round(flood_score, 3),
                "overflow_prevention_score": round(overflow_score, 3),
                "level_management_score": round(level_score, 3),
                "total_flood_damage": round(total_flood_damage, 3),
            },
        }

    def grade_full_optimization(self, episode_history: List[DamState]) -> Dict:
        """
        Hard task grader.

        Scoring:
        - 35%: flood prevention (damage score)
        - 25%: power generation (normalized to theoretical max)
        - 20%: drought prevention
        - 20%: operational efficiency (no wasteful spills)

        This task genuinely challenges even strong models because you must
        balance competing objectives across seasonal changes.
        """
        if not episode_history:
            return {"score": 0.0, "breakdown": {}}

        steps = len(episode_history)
        final = episode_history[-1]

        # Flood score
        total_flood_damage = final.cumulative_flood_damage
        flood_score = max(0.0, 1.0 - total_flood_damage / 5.0)

        # Power generation score (realistic baseline: 100 MWh/day × 30 days = 72,000 kWh)
        target_power_kwh = 72_000.0
        actual_power = final.cumulative_power_kwh
        power_score = min(1.0, actual_power / target_power_kwh)

        # Drought score — penalize steps where level < 15%
        drought_steps = sum(1 for s in episode_history if s.drought_condition)
        drought_score = max(0.0, 1.0 - drought_steps / max(steps, 1) * 2.0)

        # Efficiency — penalize water wasted (uncontrolled spill)
        # Target: waste less than 20% of total inflow
        total_wasted = final.total_water_wasted
        total_inflow_approx = 20.0 * steps  # rough estimate
        waste_fraction = total_wasted / max(total_inflow_approx, 1.0)
        efficiency_score = max(0.0, 1.0 - waste_fraction)

        total = (0.35 * flood_score + 0.25 * power_score +
                 0.20 * drought_score + 0.20 * efficiency_score)

        return {
            "score": round(min(1.0, max(0.0, total)), 4),
            "breakdown": {
                "flood_prevention_score": round(flood_score, 3),
                "power_generation_score": round(power_score, 3),
                "drought_prevention_score": round(drought_score, 3),
                "efficiency_score": round(efficiency_score, 3),
                "total_flood_damage": round(total_flood_damage, 3),
                "power_generated_kwh": round(actual_power, 1),
            },
        }

    def grade(self, task_name: str, episode_history: List[DamState]) -> Dict:
        """Grade any task by name. Returns dict with 'score' in [0.0, 1.0]."""
        graders = {
            "level_management": self.grade_level_management,
            "flood_prevention": self.grade_flood_prevention,
            "full_optimization": self.grade_full_optimization,
        }
        if task_name not in graders:
            return {"score": 0.0, "error": f"Unknown task: {task_name}"}
        return graders[task_name](episode_history)
