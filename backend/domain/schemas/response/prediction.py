"""
LEARNING NOTE: Response schemas actualizados
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class SorteoPrediction(BaseModel):
    """Predicción individual de un sorteo"""
    
    nombre_sorteo: str
    id_sorteo: int
    tipo: str
    fecha_cierre: str
    
    # Valores de predicción
    talones_estimados_total: float
    talones_estimados_hoy: float
    porcentaje_avance: float
    dias_restantes: int
    
    # Metadata
    was_smoothed: bool
    change_percentage: Optional[float]
    confidence_score: float
    
    # Datos para gráfica
    chart_data: Optional[Dict[str, List]]

class PredictionResponse(BaseModel):
    """Response de predicciones"""
    
    success: bool = Field(..., description="Si la predicción fue exitosa")
    predictions: Dict[str, Any] = Field(..., description="Predicciones por sorteo")
    
    # Metadata
    processing_time: float = Field(..., description="Tiempo de procesamiento en segundos")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Para debugging
    debug_info: Optional[Dict[str, Any]] = Field(None)
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "predictions": {
                    "TST_Sorteo_Tradicional_219": {
                        "talones_estimados_total": 850000,
                        "porcentaje_avance": 85.0,
                        "was_smoothed": True
                    }
                },
                "processing_time": 2.34,
                "metadata": {
                    "sorteos_procesados": 5,
                    "tipos_procesados": ["TST", "SMS"]
                }
            }
        }

class DashboardSummaryResponse(BaseModel):
    """Response para el resumen del dashboard"""
    
    kpis: Dict[str, Any] = Field(..., description="KPIs principales")
    sorteos_por_tipo: Dict[str, int] = Field(..., description="Conteo por tipo")
    proximos_cierres: List[Dict[str, Any]] = Field(..., description="Próximos sorteos a cerrar")
    last_update: str = Field(..., description="Última actualización")
    
    class Config:
        schema_extra = {
            "example": {
                "kpis": {
                    "total_sorteos_activos": 6,
                    "prediccion_total_hoy": 1500000,
                    "precision_promedio": 94.2,
                    "sorteos_suavizados_hoy": 2
                },
                "sorteos_por_tipo": {
                    "TST": 1,
                    "SMS": 1,
                    "AVT": 2,
                    "SOE": 1,
                    "DXV": 1
                },
                "proximos_cierres": [
                    {
                        "nombre": "Sorteo Tradicional 219",
                        "fecha_cierre": "2025-02-15",
                        "dias_restantes": 45
                    }
                ],
                "last_update": "2025-01-01T10:30:00"
            }
        }