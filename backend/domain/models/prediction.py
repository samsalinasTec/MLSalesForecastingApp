"""
LEARNING NOTE: Domain Models = Entidades del negocio
Representan conceptos del mundo real en código
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class Prediction(BaseModel):
    """
    LEARNING NOTE: Este es el modelo central del negocio.
    Field() añade validaciones y documentación.
    """
    
    id: Optional[str] = Field(None, description="ID único de la predicción")
    product_id: str = Field(..., description="ID del producto")
    
    # Valores de predicción
    original_value: float = Field(..., description="Valor crudo del modelo ML")
    smoothed_value: Optional[float] = Field(None, description="Valor después de suavizar")
    final_value: float = Field(..., description="Valor final a usar")
    
    # Metadata
    prediction_date: datetime = Field(default_factory=datetime.now)
    was_smoothed: bool = Field(False, description="Si se aplicó suavizado")
    change_percentage: Optional[float] = Field(None, description="% de cambio vs día anterior")
    
    # Contexto
    historical_context: Optional[List[float]] = Field(
        None, 
        description="Últimos N valores para contexto"
    )
    
    # LLM Insights
    llm_explanation: Optional[str] = Field(None, description="Explicación del LLM")
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    
    class Config:
        # LEARNING NOTE: Permite usar este modelo con ORMs
        orm_mode = True
        schema_extra = {
            "example": {
                "product_id": "product_1",
                "original_value": 265000,
                "smoothed_value": 272000,
                "final_value": 272000,
                "was_smoothed": True,
                "change_percentage": -3.2
            }
        }