"""
LEARNING NOTE: Este archivo centraliza TODA la configuración.
Patrón: "Single Source of Truth" para configuraciones.
"""

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    """
    LEARNING NOTE: Pydantic Settings valida automáticamente las variables de entorno
    y las convierte al tipo correcto (int, float, bool, etc.)
    """
    
    # API Settings
    api_version: str = "v1"
    debug_mode: bool = False
    port: int = 8000
    
    # Google Cloud Settings
    gcp_project_id: str
    gcp_region: str = "us-central1"
    
    # BigQuery Settings
    bigquery_dataset: str
    bigquery_table: Optional[str] = None
    save_predictions_to_bq: bool = False
    
    # Vertex AI Settings  
    vertex_ai_location: str = "us-central1"
    model_name: str = "gemini-1.5-flash-002"
    
    # Business Logic Settings
    max_daily_change_percent: float = 2.0
    smoothing_alpha: float = 0.25
    history_days: int = 10
    
    # PREGUNTA: ¿Necesitas autenticación por ahora o lo dejamos para después?
    # jwt_secret_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """
    LEARNING NOTE: @lru_cache hace que solo se cree una instancia (Singleton Pattern)
    Esto evita leer el .env múltiples veces
    """
    return Settings()

# Instancia global para importar fácilmente
settings = get_settings()