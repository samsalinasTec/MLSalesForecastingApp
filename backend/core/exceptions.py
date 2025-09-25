"""
LEARNING NOTE: Excepciones custom para errores específicos del negocio.
Mejor que usar Exception genérica porque son más descriptivas.
"""

from fastapi import HTTPException
from typing import Any, Dict, Optional

class BusinessException(HTTPException):
    """Base para todas las excepciones de negocio"""
    def __init__(
        self,
        status_code: int = 400,
        detail: str = "Business logic error",
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)

class DataNotFoundException(BusinessException):
    """Cuando no hay datos en BigQuery"""
    def __init__(self, product_id: str):
        super().__init__(
            status_code=404,
            detail=f"No se encontraron datos para el producto {product_id}"
        )

class PredictionException(BusinessException):
    """Cuando falla el modelo de ML"""
    def __init__(self, detail: str):
        super().__init__(
            status_code=500,
            detail=f"Error en predicción: {detail}"
        )

class SmoothingException(BusinessException):
    """Cuando falla el suavizado"""
    def __init__(self, detail: str):
        super().__init__(
            status_code=500,
            detail=f"Error en suavizado: {detail}"
        )

class VertexAIException(BusinessException):
    """Cuando falla Vertex AI"""
    def __init__(self, detail: str):
        super().__init__(
            status_code=503,
            detail=f"Error con Vertex AI: {detail}"
        )