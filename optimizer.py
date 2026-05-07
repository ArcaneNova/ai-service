"""
optimizer.py — Genetic Algorithm based bus headway optimizer.
Finds optimal bus dispatch times for a route given demand curve and fleet size.
"""

import logging
import random
import math
from typing import List, Dict

try:
    import predictors as _pred
except Exception:
    _pred = None

logger = logging.getLogger(__name__)


def _demand_at_hour(hour: int, is_weekend: bool = False) -> float:
    """Simple demand model: morning + evening peaks."""
    # Morning peak 8-10, evening peak 17-20
    base = 60
    morning_peak = 100 * math.exp(-0.5 * ((hour - 9) / 1.5) ** 2)
    evening_peak = 120 * math.exp(-0.5 * ((hour - 18) / 1.5) ** 2)
    weekend_factor = 0.7 if is_weekend else 1.0
    return max(10, (base + morning_peak + evening_peak) * weekend_factor)


def _total_wait_time(schedule_minutes: List[int], demand_curve: List[float],
                     start_min: int = 300, end_min: int = 1380) -> float:
    """
    Fitness function: total passenger wait time across the full service window.
    Uses service-window sentinels so the GA must spread buses across operating hours.
    """
    if len(schedule_minutes) < 1:
        return 1e9

    total = 0.0
    # Bookend with service window boundaries so pre-first and post-last gaps are penalised
    times = [start_min] + sorted(schedule_minutes) + [end_min]

    for i in range(len(times) - 1):
        gap    = times[i + 1] - times[i]          # minutes between events
        hour   = min(times[i] // 60, 23)
        demand = demand_curve[hour]
        # Average wait in this gap = gap / 2; cost = demand × wait
        total += demand * (gap / 2)

    return total


def optimize_headway(
    route_id:        str,
    date:            str,
    fleet_size:      int,
    is_weekend:      bool = False,
    is_holiday:      bool = False,
    start_hour:      int  = 5,
    end_hour:        int  = 23,
    population_size: int  = 60,
    generations:     int  = 120,
) -> Dict:
    """
    Genetic algorithm to find optimal bus dispatch times.

    Returns:
        {
            route_id, date, fleet_size,
            slots: [{ departure_min, departure_time_str, hour, headway_min }],
            total_wait_score: float,
            optimization_info: { generations, population_size, convergence_gen }
        }
    """
    logger.info(f"Starting GA optimizer: route={route_id}, date={date}, fleet={fleet_size}")

    # Demand curve for 24 hours
    demand_curve = [_demand_at_hour(h, is_weekend or is_holiday) for h in range(24)]

    # Operational window in minutes from midnight
    start_min = start_hour * 60
    end_min   = end_hour   * 60
    window    = end_min - start_min

    # ── Genetic Algorithm ──────────────────────────────────────────────────

    def random_individual() -> List[int]:
        """Random departure times (minutes from midnight), sorted."""
        times = sorted(random.sample(range(start_min, end_min), min(fleet_size, window)))
        return times

    def crossover(parent1: List[int], parent2: List[int]) -> List[int]:
        """Single-point crossover — preserves fleet_size."""
        if len(parent1) < 2:
            return parent1[:]
        point = random.randint(1, len(parent1) - 1)
        child = sorted(set(parent1[:point] + parent2[point:]))
        # Pad back to fleet_size if set() removed duplicates
        while len(child) < fleet_size:
            new_t = random.randint(start_min, end_min - 1)
            if new_t not in child:
                child.append(new_t)
                child.sort()
        return child[:fleet_size]

    def mutate(individual: List[int], mutation_rate: float = 0.15) -> List[int]:
        """Randomly shift dispatch times — preserves fleet_size."""
        mutated = individual[:]
        used = set(mutated)
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                shift = random.randint(-30, 30)
                new_t = max(start_min, min(end_min - 1, mutated[i] + shift))
                if new_t not in used:
                    used.discard(mutated[i])
                    mutated[i] = new_t
                    used.add(new_t)
        return sorted(mutated)

    def fitness(individual: List[int]) -> float:
        return -_total_wait_time(individual, demand_curve, start_min, end_min)  # higher = better

    # Initialize population
    population = [random_individual() for _ in range(population_size)]
    best        = min(population, key=lambda x: _total_wait_time(x, demand_curve, start_min, end_min))
    best_score  = _total_wait_time(best, demand_curve, start_min, end_min)
    convergence_gen = 0

    for gen in range(generations):
        # Evaluate fitness
        scored = [(fitness(ind), ind) for ind in population]
        scored.sort(reverse=True)

        # Elitism: keep top 10%
        elite_n   = max(2, population_size // 10)
        new_pop   = [ind for _, ind in scored[:elite_n]]

        # Tournament selection + crossover + mutation
        while len(new_pop) < population_size:
            # Tournament
            t1 = max(random.sample(scored, min(4, len(scored))), key=lambda x: x[0])[1]
            t2 = max(random.sample(scored, min(4, len(scored))), key=lambda x: x[0])[1]
            child = mutate(crossover(t1, t2))
            if len(child) == fleet_size:
                new_pop.append(child)

        population = new_pop

        current_best_score = _total_wait_time(scored[0][1], demand_curve, start_min, end_min)
        if current_best_score < best_score:
            best_score  = current_best_score
            best        = scored[0][1]
            convergence_gen = gen

    # Format output slots
    def min_to_time(m: int) -> str:
        h, mn = divmod(m, 60)
        return f"{h:02d}:{mn:02d}"

    slots = []
    prev = None
    for t in best:
        headway = t - prev if prev is not None else 0
        hour    = t // 60
        # Try ML demand prediction; fall back to static demand_curve value
        ml_demand: float = demand_curve[hour]
        if _pred is not None:
            try:
                ml_result = _pred.predict_demand(
                    route_id      = route_id,
                    date          = date,
                    hour          = hour,
                    is_weekend    = is_weekend,
                    is_holiday    = is_holiday,
                    weather       = "clear",
                    avg_temp_c    = 30.0,
                    special_event = False,
                    model_key     = "auto",
                )
                ml_demand = float(ml_result.get("predicted_count", ml_demand))
            except Exception:
                pass  # keep static fallback
        crowd = (
            "very_high" if ml_demand > 160 else
            "high"      if ml_demand > 100 else
            "medium"    if ml_demand > 60  else "low"
        )
        slots.append({
            "departure_min":      t,
            "departure_time_str": min_to_time(t),
            "hour":               hour,
            "headway_min":        headway,
            "demand_score":       round(ml_demand, 1),
            "crowd_level":        crowd,
        })
        prev = t

    logger.info(f"GA complete: {generations} gens, best score={best_score:.1f}, converged at gen {convergence_gen}")

    return {
        "route_id":          route_id,
        "date":              date,
        "fleet_size":        fleet_size,
        "slots":             slots,
        "total_wait_score":  round(best_score, 2),
        "optimization_info": {
            "generations":      generations,
            "population_size":  population_size,
            "convergence_gen":  convergence_gen,
            "algorithm":        "Genetic Algorithm (tournament selection + single-point crossover)",
        },
    }
