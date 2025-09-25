"""
LEARNING NOTE: Schemas actualizados para sorteos
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date
from backend.core.constants import SorteoType

class PredictionRequest(BaseModel):
    """Request para crear predicciones de sorteos"""
    
    sorteo_types: List[str] = Field(
        ..., 
        description="Tipos de sorteo a procesar (TST, SMS, etc.)",
        min_items=1,
        max_items=5,
        example=["TST", "SMS"]
    )
    
    force_recalculation: bool = Field(
        False,
        description="Forzar recálculo aunque ya exista predicción del día"
    )
    
    apply_smoothing: bool = Field(
        True,
        description="Aplicar suavizado automático"
    )
    
    save_to_bq: bool = Field(
        True,
        description="Guardar resultados en BigQuery"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "sorteo_types": ["TST", "SMS", "AVT"],
                "apply_smoothing": True,
                "save_to_bq": True
            }
        }

class HistoricalRequest(BaseModel):
    """Request para obtener históricos de un sorteo"""
    
    nombre_sorteo: str = Field(..., description="Nombre del sorteo")
    days: int = Field(10, ge=1, le=30, description="Días de historia")
    
    include_predictions: bool = Field(
        True,
        description="Incluir predicciones además de valores reales"
    )
    
    include_chart_data: bool = Field(
        True,
        description="Incluir datos formateados para gráficas"
    )

class ComparisonRequest(BaseModel):
    """Request para comparar sorteos"""
    
    sorteo_names: List[str] = Field(
        ...,
        description="Lista de sorteos a comparar",
        min_items=2,
        max_items=10
    )
    
    metric: str = Field(
        "PORCENTAJE_DE_AVANCE",
        description="Métrica a comparar"
    )
    
    normalize: bool = Field(
        True,
        description="Normalizar por DNAS para comparación justa"
    )