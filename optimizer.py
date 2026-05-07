"""
optimizer.py — Genetic Algorithm based bus headway optimizer.
Finds optimal bus dispatch times for a route given demand curve and fleet size.
"""

import logging
import random
import math
from typing import List, Dict

logger = logging.getLogger(__name__)


def _demand_at_hour(hour: int, is_weekend: bool = False) -> float:
    """Simple demand model: morning + evening peaks."""
    # Morning peak 8-10, evening peak 17-20
    base = 60
    morning_peak = 100 * math.exp(-0.5 * ((hour - 9) / 1.5) ** 2)
    evening_peak = 120 * math.exp(-0.5 * ((hour - 18) / 1.5) ** 2)
    weekend_factor = 0.7 if is_weekend else 1.0
    return max(10, (base + morning_peak + evening_peak) * weekend_factor)


def _total_wait_time(schedule_minutes: List[int], demand_curve: List[float]) -> float:
    """
    Fitness function: total passenger wait time.
    Passengers arrive uniformly; average wait = headway / 2 × demand.
    """
    if len(schedule_minutes) < 2:
        return 1e9

    total = 0.0
    # Add sentinel times
    times = sorted(schedule_minutes)
    times = [0] + times + [24 * 60]

    for i in range(1, len(times) - 1):
        headway = times[i + 1] - times[i]   # minutes between buses
        hour = times[i] // 60
        demand = demand_curve[min(hour, 23)]
        # Wait time cost = demand × (headway / 2)
        total += demand * (headway / 2)

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
        """Single-point crossover."""
        if len(parent1) < 2:
            return parent1[:]
        point = random.randint(1, len(parent1) - 1)
        child = sorted(set(parent1[:point] + parent2[point:]))
        # Ensure correct fleet size
        while len(child) < fleet_size and len(child) < window:
            new_t = random.randint(start_min, end_min - 1)
            if new_t not in child:
                child.append(new_t)
        child = sorted(child[:fleet_size])
        return child

    def mutate(individual: List[int], mutation_rate: float = 0.15) -> List[int]:
        """Randomly shift dispatch times."""
        mutated = individual[:]
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                shift = random.randint(-20, 20)
                mutated[i] = max(start_min, min(end_min - 1, mutated[i] + shift))
        return sorted(set(mutated))

    def fitness(individual: List[int]) -> float:
        return -_total_wait_time(individual, demand_curve)  # higher = better

    # Initialize population
    population = [random_individual() for _ in range(population_size)]
    best        = min(population, key=lambda x: _total_wait_time(x, demand_curve))
    best_score  = _total_wait_time(best, demand_curve)
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

        current_best_score = _total_wait_time(scored[0][1], demand_curve)
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
        slots.append({
            "departure_min":      t,
            "departure_time_str": min_to_time(t),
            "hour":               t // 60,
            "headway_min":        headway,
            "demand_score":       round(demand_curve[t // 60], 1),
            "crowd_level":        (
                "very_high" if demand_curve[t // 60] > 160 else
                "high"      if demand_curve[t // 60] > 100 else
                "medium"    if demand_curve[t // 60] > 60  else "low"
            ),
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
