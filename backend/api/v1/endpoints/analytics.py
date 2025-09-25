"""
LEARNING NOTE: Endpoints para análisis y comparaciones
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, date, timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analytics",
    tags=["analytics"],
    responses={404: {"description": "Not found"}}
)

@router.get("/comparison")
async def compare_sorteos(
    sorteo_names: List[str] = Query(..., description="Sorteos a comparar"),
    metric: str = Query("PORCENTAJE_DE_AVANCE", description="Métrica a comparar")
):
    """
    Compara curvas de venta entre sorteos
    
    LEARNING NOTE: Útil para ver si un sorteo va mejor o peor que otros similares
    """
    
    try:
        # TODO: Implementar comparación
        
        return {
            "sorteos": sorteo_names,
            "metric": metric,
            "comparison_data": {
                "labels": [],  # DNAS o fechas
                "datasets": []  # Un dataset por sorteo
            },
            "insights": {
                "best_performer": None,
                "worst_performer": None,
                "average_at_current_dnas": 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error en comparación: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical-accuracy")
async def get_historical_accuracy(
    days_back: int = Query(30, description="Días hacia atrás para analizar"),
    sorteo_type: Optional[str] = Query(None, description="Filtrar por tipo")
):
    """
    Analiza la precisión histórica de las predicciones
    
    LEARNING NOTE: Compara predicciones pasadas vs valores reales
    """
    
    try:
        # TODO: Implementar análisis de precisión
        
        return {
            "period_analyzed": f"Last {days_back} days",
            "metrics": {
                "mape": 0,  # Mean Absolute Percentage Error
                "rmse": 0,  # Root Mean Square Error
                "r2_score": 0,
                "smoothing_frequency": 0,  # % de veces que se aplicó suavizado
            },
            "by_sorteo_type": {},
            "recommendations": []
        }
        
    except Exception as e:
        logger.error(f"Error en análisis de precisión: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trends")
async def get_sales_trends(
    sorteo_type: str,
    period: str = Query("monthly", description="daily, weekly, monthly")
):
    """
    Obtiene tendencias de venta por tipo de sorteo
    """
    
    try:
        # TODO: Implementar análisis de tendencias
        
        return {
            "sorteo_type": sorteo_type,
            "period": period,
            "trends": {
                "current_trend": "increasing",  # increasing, decreasing, stable
                "change_percentage": 0,
                "seasonal_patterns": [],
                "forecast_next_period": 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error en análisis de tendencias: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard-summary")
async def get_dashboard_summary():
    """
    Obtiene resumen para el dashboard principal
    
    LEARNING NOTE: Este endpoint alimenta los KPIs del dashboard
    """
    
    try:
        from backend.application.services.sorteo_prediction_service import SorteoPredictionService
        from backend.infrastructure.database.bigquery import BigQueryRepository
        
        prediction_service = SorteoPredictionService()
        bq_repo = BigQueryRepository()
        
        # Obtener info básica
        df_info = bq_repo.get_sorteos_info()
        df_activos = df_info[
            pd.to_datetime(df_info["FECHA_CIERRE"]) >= pd.Timestamp.now()
        ]
        
        return {
            "kpis": {
                "total_sorteos_activos": len(df_activos),
                "prediccion_total_hoy": 0,  # TODO: Sumar todas las predicciones
                "precision_promedio": 94.2,  # TODO: Calcular real
                "sorteos_suavizados_hoy": 0,  # TODO: Contar
            },
            "sorteos_por_tipo": df_activos.groupby("SORTEO_GRUPO").size().to_dict(),
            "proximos_cierres": [
                {
                    "nombre": row["NOMBRE"],
                    "fecha_cierre": row["FECHA_CIERRE"].isoformat(),
                    "dias_restantes": (row["FECHA_CIERRE"] - pd.Timestamp.now()).days
                }
                for _, row in df_activos.nsmallest(5, "FECHA_CIERRE").iterrows()
            ],
            "last_update": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generando resumen: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))