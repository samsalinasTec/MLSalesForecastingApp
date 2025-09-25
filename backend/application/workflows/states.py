"""
LEARNING NOTE: Estados del workflow para sorteos
TypedDict define la estructura del estado que pasa entre nodos
"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

class PredictionState(TypedDict):
    """
    Estado del workflow de predicción de sorteos
    
    LEARNING NOTE: Este estado se va actualizando
    conforme pasa por cada nodo del workflow
    """
    
    # Inputs iniciales
    sorteo_types: List[str]  # ["TST", "SMS", etc.]
    apply_smoothing: bool
    force_recalculation: bool
    save_to_bq: bool
    
    # Datos cargados
    data_fetched: bool
    sorteo_data: Dict[str, Any]  # DataFrames convertidos
    
    # Predicciones
    predictions: Dict[str, Any]  # Predicciones por sorteo
    raw_predictions: Dict[str, Any]  # Sin suavizar
    
    # Suavizado
    smoothed_predictions: Dict[str, Any]
    was_smoothed: Dict[str, bool]
    
    # Explicaciones LLM
    needs_llm_explanation: bool
    explanations: Dict[str, str]
    market_insights: Optional[Dict[str, Any]]
    
    # Resultados finales
    final_predictions: Dict[str, Dict[str, Any]]
    summary_df: Optional[pd.DataFrame]
    
    # Control y metadata
    processing_stage: str
    prediction_count: int
    total_sorteos_processed: int
    
    # Timing
    start_time: datetime
    end_time: Optional[datetime]
    processing_time: Optional[float]
    
    # Errores
    errors: List[str]