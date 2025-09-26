"""
LEARNING NOTE: Modelo simple para productos
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class Product(BaseModel):
    """Información del producto"""
    
    id: str = Field(..., description="ID único del producto")
    name: str = Field(..., description="Nombre del producto")
    category: Optional[str] = Field(None, description="Categoría")
    
    # PREGUNTA: ¿Qué otros atributos tienen tus productos?
    # Por ejemplo: precio_unitario, margen, estacionalidad?
    
    # Configuración específica del producto
    smoothing_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Configuración custom de suavizado por producto"
    )
    
    is_active: bool = Field(True, description="Si está activo para predicciones")
    
    class Config:
        orm_mode = True