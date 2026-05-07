"""
enhanced_generate_dataset.py
Generates a realistic 3-year training dataset with:
- Real travel times calculated from DTC route coordinates (Haversine)
- Realistic peak-hour patterns (non-linear rush hour demand)
- Seasonal variations (summer vs winter)
- Weather-demand correlations
- Multiple route types with different patterns
- Advanced temporal and geographic features

Usage: python training/enhanced_generate_dataset.py
"""

import os
import random
import csv
import logging
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────

# Resolve paths - works in both local and Colab environments
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # ai-service/
workspace_root = os.path.dirname(project_root)  # bus-site/

# Try multiple possible paths for routes/stages CSV
ROUTES_CSV = None
STAGES_CSV = None

for possible_root in [
    workspace_root,
    "/content/bus-site",
]:
    routes_path = os.path.join(possible_root, "routes.csv")
    stages_path = os.path.join(possible_root, "stages.csv")
    if os.path.exists(routes_path) and os.path.exists(stages_path):
        ROUTES_CSV = routes_path
        STAGES_CSV = stages_path
        break

if ROUTES_CSV is None or STAGES_CSV is None:
    raise FileNotFoundError("❌ routes.csv or stages.csv not found in workspace root")

OUT_DIR = os.path.join(project_root, "data")
os.makedirs(OUT_DIR, exist_ok=True)

# Configuration
# Reduced for Colab compatibility: sampling strategy instead of exhaustive
YEARS_OF_DATA = 1  # Effective: ~6 months of sampled data
SAMPLING_RATIO = 0.5  # Sample 50% of days to stay under 1GB
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────
# Constants

WEATHER_OPTIONS = ["clear", "light_rain", "heavy_rain", "fog", "heatwave", "extreme"]
WEATHER_WEIGHTS = [0.50, 0.20, 0.10, 0.12, 0.06, 0.02]

# Weather impact on demand (multiplicative factors)
WEATHER_DEMAND_FACTOR = {
    "clear": 1.0,
    "light_rain": 1.15,      # More people use buses in light rain
    "heavy_rain": 1.30,       # Much more demand in heavy rain
    "fog": 0.95,
    "heatwave": 0.80,         # Fewer trips in extreme heat
    "extreme": 0.70,
}

# Weather impact on delay (minutes added)
WEATHER_DELAY_FACTOR = {
    "clear": 0.0,
    "light_rain": 3.0,
    "heavy_rain": 8.0,
    "fog": 5.0,
    "heatwave": 1.0,
    "extreme": 12.0,
}

# Base demand by hour (refined rush hour patterns)
HOUR_BASE_DEMAND = {
    0: 5, 1: 3, 2: 2, 3: 2, 4: 5,
    5: 20, 6: 60, 7: 150, 8: 200, 9: 160, 10: 80,
    11: 70, 12: 90, 13: 85, 14: 75, 15: 80,
    16: 110, 17: 180, 18: 200, 19: 160, 20: 100,
    21: 60, 22: 40, 23: 20,
}

# DTC holidays for 2022-2024
DTC_HOLIDAYS = {
    # 2022
    "2022-01-26", "2022-03-18", "2022-04-10", "2022-04-14",
    "2022-05-03", "2022-08-15", "2022-09-10", "2022-10-05",
    "2022-10-24", "2022-12-25",
    # 2023
    "2023-01-26", "2023-03-07", "2023-04-04", "2023-04-14",
    "2023-05-19", "2023-08-15", "2023-09-29", "2023-10-24",
    "2023-11-12", "2023-12-25",
    # 2024
    "2024-01-26", "2024-03-25", "2024-04-14", "2024-04-17",
    "2024-05-23", "2024-08-15", "2024-10-02", "2024-10-12",
    "2024-11-15", "2024-12-25",
}

# Major Delhi events (increase demand)
MAJOR_EVENTS = {
    "2023-03-07": "Holi Festival",           # +40% demand
    "2023-10-24": "Diwali Festival",         # +50% demand
    "2023-11-12": "Shopping Festival",       # +25% demand
    "2024-03-25": "Holi Festival",
    "2024-10-12": "Diwali Festival",
    "2024-11-15": "Year-end Shopping",
}

# ─────────────────────────────────────────────────────────────────────────
# Helper Functions

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates (km)."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def is_holiday(date: datetime) -> bool:
    """Check if date is a DTC holiday."""
    return date.strftime("%Y-%m-%d") in DTC_HOLIDAYS

def is_weekend(date: datetime) -> bool:
    """Check if date is Saturday or Sunday."""
    return date.weekday() >= 5

def is_major_event(date: datetime) -> bool:
    """Check if date has a major event."""
    return date.strftime("%Y-%m-%d") in MAJOR_EVENTS

def random_weather() -> str:
    """Generate random weather based on realistic weights."""
    return random.choices(WEATHER_OPTIONS, weights=WEATHER_WEIGHTS)[0]

def get_season_temp(date: datetime, base_var=5):
    """Get temperature based on season with variance."""
    month = date.month
    if month in range(4, 10):  # April-Sept: hot
        base_temp = 32
    elif month in (3, 10, 11):  # March, Oct, Nov: moderate
        base_temp = 25
    else:  # Dec, Jan, Feb: cool
        base_temp = 18
    
    return base_temp + np.random.normal(0, base_var)

def classify_route_type(route_id, stages_data):
    """Classify route type based on stages coverage."""
    # Simple heuristic: based on route characteristics
    # In production, use actual route classification
    route_num = int(route_id) % 100
    if route_num < 30:
        return "commercial_hub"  # High demand central routes
    elif route_num < 60:
        return "residential"      # Moderate demand
    else:
        return "peripheral"       # Lower demand

def demand_at_hour(hour, weather, date, special_event, route_type, capacity):
    """Calculate demand for a specific hour."""
    base = HOUR_BASE_DEMAND.get(hour, 30)
    
    # Weather factor
    weather_factor = WEATHER_DEMAND_FACTOR.get(weather, 1.0)
    
    # Day type factors
    day_factor = 1.0
    if is_weekend(date):
        day_factor *= 0.70
    if is_holiday(date):
        day_factor *= 0.60
    
    # Special event
    if special_event or is_major_event(date):
        day_factor *= 1.50
    
    # Route type factors
    route_factors = {
        "commercial_hub": 1.30,   # Higher demand
        "residential": 1.0,       # Standard demand
        "peripheral": 0.75,       # Lower demand
    }
    day_factor *= route_factors.get(route_type, 1.0)
    
    # Apply factors
    demand = base * weather_factor * day_factor
    
    # Add realistic noise
    noise = np.random.normal(0, demand * 0.12)
    demand = max(0, demand + noise)
    
    # Scale to route capacity
    demand = min(capacity * 1.2, demand)  # Can exceed capacity (standees)
    
    return int(round(demand))

def delay_at_hour(hour, weather, load_pct, distance_km, is_weekend, 
                  total_stops, date, route_type):
    """Calculate delay for a specific hour."""
    base_delay = 0.0
    
    # Peak hour delays
    if hour in (7, 8, 9, 17, 18, 19):
        base_delay += 10
    elif hour in (6, 10, 16, 20):
        base_delay += 4
    
    # Weather delays
    base_delay += WEATHER_DELAY_FACTOR.get(weather, 0.0)
    
    # Load impacts
    if load_pct > 80:
        base_delay += 3
    if load_pct > 100:
        base_delay += 5
    
    # Distance factor
    base_delay += distance_km * 0.05
    
    # Weekend (less traffic)
    if is_weekend:
        base_delay *= 0.7
    
    # Holiday (minimal traffic)
    if is_holiday(date):
        base_delay *= 0.5
    
    # Stops factor
    base_delay += total_stops * 0.2
    
    # Add realistic noise
    noise = np.random.normal(0, 2.0)
    delay = max(0.0, base_delay + noise)
    
    return round(delay, 1)

def calculate_real_distance(route_id, stages_data):
    """Calculate real distance using Haversine formula from coordinates."""
    route_stages = stages_data[stages_data['url_route_id'] == route_id].sort_values('seq')
    
    if len(route_stages) < 2:
        return 15.0  # Default fallback
    
    total_distance = 0.0
    for i in range(len(route_stages) - 1):
        lat1, lon1 = route_stages.iloc[i][['latitude', 'longitude']].values
        lat2, lon2 = route_stages.iloc[i+1][['latitude', 'longitude']].values
        total_distance += haversine(lat1, lon1, lat2, lon2)
    
    return total_distance

# ─────────────────────────────────────────────────────────────────────────
# Data Generation

def load_routes():
    """Load routes from CSV."""
    routes = []
    with open(ROUTES_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                routes.append({
                    "url_route_id": row["url_route_id"],
                    "route_name": row.get("route_name", ""),
                    "distance_km_csv": float(row.get("distance_km", 15) or 15),
                    "start_stage": row.get("start_stage", ""),
                    "end_stage": row.get("end_stage", ""),
                })
            except (ValueError, KeyError):
                continue
    return routes

def load_stages():
    """Load stages from CSV."""
    return pd.read_csv(STAGES_CSV)

def generate_demand_dataset(routes, stages_df):
    """Generate 3-year demand dataset with realistic patterns."""
    logger.info("🔄 Generating demand dataset (3 years)...")
    
    records = []
    start_date = datetime(2022, 1, 1)
    
    for route in routes:
        route_id = route["url_route_id"]
        
        # Calculate real distance
        real_distance = calculate_real_distance(route_id, stages_df)
        
        # Classify route type
        route_type = classify_route_type(route_id, stages_df)
        
        # Count stops
        route_stages = stages_df[stages_df['url_route_id'] == route_id]
        total_stops = len(route_stages)
        
        capacity = 60  # Standard DTC bus
        
        # Sampling: 50% of days to reduce memory (Colab limit)
        total_days = 365 * YEARS_OF_DATA
        for day_offset in range(total_days):
            if random.random() > SAMPLING_RATIO:
                continue
            
            date = start_date + timedelta(days=day_offset)
            
            is_wknd = is_weekend(date)
            is_hday = is_holiday(date)
            special_event = random.random() < 0.02  # 2% random special events
            weather = random_weather()
            temp = get_season_temp(date)
            month = date.month
            day_of_week = date.weekday()
            
            # Generate hourly demand
            for hour in range(24):
                demand_count = demand_at_hour(
                    hour, weather, date, special_event,
                    route_type, capacity
                )
                
                crowd_level = (
                    "critical" if demand_count >= 90
                    else "high" if demand_count >= 60
                    else "medium" if demand_count >= 30
                    else "low"
                )
                
                records.append({
                    "route_id": route_id,
                    "route_type": route_type,
                    "date": date.strftime("%Y-%m-%d"),
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "is_weekend": int(is_wknd),
                    "is_holiday": int(is_hday),
                    "is_major_event": int(is_major_event(date)),
                    "weather": weather,
                    "avg_temp_c": round(temp, 1),
                    "special_event": int(special_event),
                    "month": month,
                    "quarter": (month - 1) // 3 + 1,
                    "passenger_count": demand_count,
                    "crowd_level": crowd_level,
                    "distance_km": round(real_distance, 2),
                    "total_stops": total_stops,
                })
    
    df = pd.DataFrame(records)
    output_path = os.path.join(OUT_DIR, "demand_dataset.csv")
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Demand dataset: {len(df):,} rows → {output_path}")
    return df

def generate_delay_dataset(routes, stages_df):
    """Generate delay dataset with realistic patterns (sampled for Colab)."""
    logger.info("🔄 Generating delay dataset (sampled ~180 days)...")
    
    records = []
    start_date = datetime(2022, 1, 1)
    
    for route in routes:
        route_id = route["url_route_id"]
        
        # Calculate real distance
        real_distance = calculate_real_distance(route_id, stages_df)
        
        # Classify route type
        route_type = classify_route_type(route_id, stages_df)
        
        # Count stops
        route_stages = stages_df[stages_df['url_route_id'] == route_id]
        total_stops = len(route_stages)
        
        # Sampling: 50% of days to reduce memory (Colab limit)
        total_days = 365 * YEARS_OF_DATA
        for day_offset in range(total_days):
            if random.random() > SAMPLING_RATIO:
                continue
            date = start_date + timedelta(days=day_offset)
            
            is_wknd = is_weekend(date)
            is_hday = is_holiday(date)
            weather = random_weather()
            month = date.month
            day_of_week = date.weekday()
            
            # Generate hourly delay
            for hour in range(24):
                # Simulated load percentage (0-150%)
                load_pct = np.random.uniform(30, 120)
                
                delay_min = delay_at_hour(
                    hour, weather, load_pct, real_distance,
                    is_wknd, total_stops, date, route_type
                )
                
                is_delayed = 1 if delay_min > 5 else 0
                delay_category = (
                    "critical" if delay_min > 15
                    else "high" if delay_min > 10
                    else "moderate" if delay_min > 5
                    else "normal"
                )
                
                records.append({
                    "route_id": route_id,
                    "route_type": route_type,
                    "date": date.strftime("%Y-%m-%d"),
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "is_weekend": int(is_wknd),
                    "is_holiday": int(is_hday),
                    "weather": weather,
                    "avg_temp_c": round(get_season_temp(date), 1),
                    "passenger_load_pct": round(load_pct, 1),
                    "distance_km": round(real_distance, 2),
                    "total_stops": total_stops,
                    "month": month,
                    "quarter": (month - 1) // 3 + 1,
                    "delay_minutes": delay_min,
                    "is_delayed": is_delayed,
                    "delay_category": delay_category,
                    "scheduled_duration_min": round(real_distance * 3, 1),  # ~3 min/km
                })
    
    df = pd.DataFrame(records)
    output_path = os.path.join(OUT_DIR, "delay_dataset.csv")
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Delay dataset: {len(df):,} rows → {output_path}")
    return df

def generate_anomaly_dataset(routes, stages_df):
    """Generate anomaly dataset (normal + abnormal patterns, sampled for Colab)."""
    logger.info("🔄 Generating anomaly dataset (sampled ~180 days)...")
    
    records = []
    start_date = datetime(2022, 1, 1)
    
    for route in routes:
        route_id = route["url_route_id"]
        
        real_distance = calculate_real_distance(route_id, stages_df)
        total_stops = len(stages_df[stages_df['url_route_id'] == route_id])
        
        # Sampling: 50% of days to reduce memory (Colab limit)
        total_days = 365 * YEARS_OF_DATA
        for day_offset in range(total_days):
            if random.random() > SAMPLING_RATIO:
                continue
            
            date = start_date + timedelta(days=day_offset)
            
            is_wknd = is_weekend(date)
            weather = random_weather()
            month = date.month
            
            # Generate 3-4 observations per day
            for trip_idx in range(random.randint(3, 5)):
                hour = random.randint(6, 22)
                
                # Normal speed: 20-40 km/h
                normal_speed = np.random.normal(30, 5)
                normal_speed = np.clip(normal_speed, 20, 40)
                
                # With 3% probability, generate anomaly
                is_anomaly = random.random() < 0.03
                
                if is_anomaly:
                    # Anomalous patterns
                    speed_kmh = random.choice([
                        np.random.uniform(5, 15),   # Too slow
                        np.random.uniform(60, 90),  # Too fast
                    ])
                    delay_minutes = np.random.uniform(25, 60)
                    passenger_load = random.choice([0, 150])  # Empty or overcrowded
                else:
                    # Normal patterns
                    weather_effect = WEATHER_DELAY_FACTOR.get(weather, 0) / 10
                    speed_kmh = normal_speed * (1 - weather_effect)
                    delay_minutes = float(np.array([np.random.exponential(3)]).clip(0, 10)[0])
                    passenger_load = np.random.uniform(30, 90)
                
                records.append({
                    "route_id": route_id,
                    "date": date.strftime("%Y-%m-%d"),
                    "hour": hour,
                    "day_of_week": date.weekday(),
                    "is_weekend": int(is_wknd),
                    "weather": weather,
                    "speed_kmh": round(speed_kmh, 1),
                    "delay_minutes": round(delay_minutes, 1),
                    "passenger_load": round(passenger_load, 1),
                    "distance_km": round(real_distance, 2),
                    "total_stops": total_stops,
                    "month": month,
                    "is_anomaly": int(is_anomaly),
                })
    
    df = pd.DataFrame(records)
    output_path = os.path.join(OUT_DIR, "anomaly_dataset.csv")
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Anomaly dataset: {len(df):,} rows → {output_path}")
    return df

# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("╔════════════════════════════════════════════════════════════════╗")
    logger.info("║   SmartDTC Enhanced Dataset Generation (3-Year, Realistic)     ║")
    logger.info("╚════════════════════════════════════════════════════════════════╝")
    
    routes = load_routes()
    stages_df = load_stages()
    
    logger.info(f"\n📊 Loading {len(routes)} routes with {len(stages_df)} total stages")
    
    demand_df = generate_demand_dataset(routes, stages_df)
    delay_df = generate_delay_dataset(routes, stages_df)
    anomaly_df = generate_anomaly_dataset(routes, stages_df)
    
    logger.info(f"\n✨ Dataset generation complete!")
    logger.info(f"   Demand:  {len(demand_df):,} records")
    logger.info(f"   Delay:   {len(delay_df):,} records")
    logger.info(f"   Anomaly: {len(anomaly_df):,} records")
