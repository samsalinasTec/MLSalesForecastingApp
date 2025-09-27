"""
LEARNING NOTE: Modelo simple para productos
"""
from datetime import date
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class Product(BaseModel):
    """Información del producto/sorteo (espejo de la tabla de BigQuery)."""

    # ===== REQUERIDOS =====
    nombre: str = Field(..., description="Nombre del producto/sorteo")
    numero_edicion: int = Field(..., description="Número de edición")
    id_producto: int = Field(..., description="ID del producto")

    # ===== OPCIONALES =====
    id_sorteo: Optional[int] = Field(None, description="ID del sorteo")
    emision: Optional[int] = Field(None, description="Número de emisión")
    precio_unitario: Optional[float] = Field(None, description="Precio unitario")
    fecha_inicio: Optional[date] = Field(None, description="Fecha de inicio")
    fecha_cierre: Optional[date] = Field(None, description="Fecha de cierre")
    permiso_gobernacion: Optional[str] = Field(None, description="Permiso de gobernación")
    sorteo_grupo: Optional[str] = Field(None, description="Grupo del sorteo")
    sorteo_siglas: Optional[str] = Field(None, description="Siglas del sorteo")

    # === Dejar igual ===
    smoothing_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Configuración custom de suavizado por producto"
    )
    is_active: bool = Field(True, description="Si está activo para predicciones")

    class Config:
        orm_mode = True