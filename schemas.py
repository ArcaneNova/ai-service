"""
schemas.py — Pydantic request/response models for the SmartDTC AI service.
Updated for multi-model support: all responses include model, is_best_model, metrics.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ── Shared ──────────────────────────────────────────────────────────────────────

class ModelMetrics(BaseModel):
    mae:   Optional[float] = None
    rmse:  Optional[float] = None
    mape:  Optional[float] = None
    r2:    Optional[float] = None


# ── Demand Prediction ────────────────────────────────────────────────────────────

class DemandRequest(BaseModel):
    route_id:      str
    date:          str                              # YYYY-MM-DD
    hour:          int   = Field(..., ge=0, le=23)
    is_weekend:    bool  = False
    is_holiday:    bool  = False
    weather:       str   = "clear"                  # clear | rain | fog | heatwave
    avg_temp_c:    float = 25.0
    special_event: bool  = False

class DemandResponse(BaseModel):
    route_id:        str
    date:            str
    hour:            int
    predicted_count: int
    crowd_level:     str                            # low | medium | high | critical
    confidence:      float
    model:           str              = "unknown"
    is_best_model:   bool             = False
    metrics:         Optional[ModelMetrics] = None
    peak_factor:     Optional[float]  = None


# ── Delay Prediction ─────────────────────────────────────────────────────────────

class DelayRequest(BaseModel):
    route_id:               str
    hour:                   int   = Field(..., ge=0, le=23)
    day_of_week:            int   = Field(..., ge=0, le=6)  # 0=Mon
    is_weekend:             bool  = False
    is_holiday:             bool  = False
    weather:                str   = "clear"
    avg_temp_c:             float = 25.0
    passenger_load_pct:     float = 50.0                    # 0-200
    scheduled_duration_min: float = 60.0
    distance_km:            float = 10.0
    total_stops:            int   = 20

class DelayMetrics(BaseModel):
    mae:  Optional[float] = None
    rmse: Optional[float] = None
    r2:   Optional[float] = None

class DelayResponse(BaseModel):
    route_id:                str
    predicted_delay_minutes: float
    is_delayed:              bool                  # True if delay > 5 min
    delay_probability:       float                 # 0-1
    model:                   str  = "unknown"
    is_best_model:           bool = False
    metrics:                 Optional[DelayMetrics] = None


# ── Schedule Generation ───────────────────────────────────────────────────────────

class ScheduleSlot(BaseModel):
    hour:               int
    minute:             int
    frequency_minutes:  int
    bus_count:          int
    type:               str                       # regular | peak | express
    predicted_demand:   Optional[int]   = None

class ScheduleRequest(BaseModel):
    route_id:               str
    date:                   str
    total_buses_available:  int = 5

class ScheduleResponse(BaseModel):
    route_id:     str
    date:         str
    slots:        List[ScheduleSlot]
    total_trips:  int
    ai_generated: bool         = True
    demand_model: Optional[str] = None


# ── Anomaly Detection ─────────────────────────────────────────────────────────────

class AnomalyMetrics(BaseModel):
    precision: Optional[float] = None
    recall:    Optional[float] = None
    f1:        Optional[float] = None
