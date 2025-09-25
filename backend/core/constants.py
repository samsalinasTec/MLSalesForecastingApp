"""
LEARNING NOTE: Actualizado para reflejar tipos de sorteo en lugar de productos
"""

from enum import Enum

# Tipos de sorteo
class SorteoType(str, Enum):
    """Enum para tipos de sorteo"""
    TST = "TST"  # Sorteo Tradicional
    SMS = "SMS"  # Sorteo SMS
    AVT = "AVT"  # Sorteo AVT
    SOE = "SOE"  # Sorteo SOE  
    DXV = "DXV"  # Sorteo DXV

# Status de predicciones
class PredictionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    SMOOTHED = "smoothed"
    ERROR = "error"

# Canales de venta
class SalesChannel(str, Enum):
    FISICO = "fisico"
    DIGITAL = "digital"
    MEMBRESIAS = "membresias"

# Mensajes comunes
MESSAGES = {
    "prediction_started": "Predicción iniciada para sorteo {sorteo}",
    "prediction_completed": "Predicción completada",
    "smoothing_applied": "Suavizado aplicado: cambio de {change:.1f}%",
    "no_historical_data": "Sin datos históricos suficientes",
    "multiple_active": "Múltiples sorteos activos del tipo {tipo}"
}