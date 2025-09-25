"""
LEARNING NOTE: Endpoints actualizados para sorteos
Cambios principales: products → sorteos
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import List, Optional, Dict, Any
from datetime import date
import logging
import pandas as pd
from io import StringIO
from fastapi.responses import StreamingResponse

from backend.domain.schemas.request.prediction import PredictionRequest
from backend.domain.schemas.response.prediction import PredictionResponse
from backend.application.workflows.graphs import run_prediction_workflow
from backend.application.services.sorteo_prediction_service import SorteoPredictionService
from backend.core.config import settings
from backend.core.constants import SorteoType

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
    responses={404: {"description": "Not found"}}
)

# Instancia del servicio
prediction_service = SorteoPredictionService()

@router.post("/", response_model=PredictionResponse)
async def create_predictions(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
) -> PredictionResponse:
    """
    Genera predicciones para sorteos activos
    
    LEARNING NOTE: Cambiado para manejar tipos de sorteo
    en lugar de productos individuales
    """
    
    try:
        # Ejecutar workflow de predicción
        result = await run_prediction_workflow(
            sorteo_types=request.sorteo_types,
            apply_smoothing=request.apply_smoothing,
            force_recalculation=request.force_recalculation
        )
        
        # Verificar errores
        if result.get("processing_stage") == "failed":
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        # Opcionalmente guardar en BigQuery en background
        if settings.save_predictions_to_bq:
            background_tasks.add_task(
                save_predictions_to_bigquery,
                result["final_predictions"]
            )
        
        return PredictionResponse(
            success=True,
            predictions=result.get("final_predictions", {}),
            processing_time=result.get("processing_time", 0),
            metadata={
                "sorteos_procesados": len(result.get("final_predictions", {})),
                "tipos_procesados": request.sorteo_types
            },
            debug_info=result if settings.debug_mode else None
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/all")
async def predict_all_active_sorteos():
    """
    Genera predicciones para TODOS los sorteos activos
    
    LEARNING NOTE: Endpoint conveniente para ejecutar todo de una vez
    """
    
    try:
        # Procesar todos los tipos
        all_types = [t.value for t in SorteoType]
        
        predictions = await prediction_service.predict_all_active_sorteos(
            tipos_producto=all_types
        )
        
        return {
            "success": True,
            "total_sorteos": sum(
                1 for k in predictions.keys() 
                if k != 'resumen'
            ),
            "predictions": predictions,
            "message": "Predicciones generadas para todos los sorteos activos"
        }
        
    except Exception as e:
        logger.error(f"Error en predicción masiva: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary/download")
async def download_summary():
    """
    Descarga el resumen de predicciones como CSV
    
    LEARNING NOTE: Este es el CSV que mencionaste que querías descargar
    """
    
    try:
        # Obtener últimas predicciones
        # TODO: Implementar obtención desde caché o BD
        
        # Por ahora, generar nuevo
        predictions = await prediction_service.predict_all_active_sorteos()
        
        if 'resumen' not in predictions:
            raise HTTPException(
                status_code=404,
                detail="No hay resumen disponible"
            )
        
        df_resumen = predictions['resumen']
        
        # Convertir DataFrame a CSV
        stream = StringIO()
        df_resumen.to_csv(stream, index=False)
        stream.seek(0)
        
        # Retornar como descarga
        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=resumen_predicciones_{date.today()}.csv"
            }
        )
        
    except Exception as e:
        logger.error(f"Error generando CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sorteo/{nombre_sorteo}")
async def get_sorteo_prediction(
    nombre_sorteo: str,
    include_history: bool = Query(False, description="Incluir contexto histórico"),
    include_chart_data: bool = Query(True, description="Incluir datos para gráfica")
):
    """
    Obtiene la predicción para un sorteo específico
    
    LEARNING NOTE: Para mostrar en el dashboard individual
    """
    
    try:
        # TODO: Implementar obtención desde caché
        
        response = {
            "sorteo": nombre_sorteo,
            "prediction": {
                "total_estimado": 0,  # TODO
                "dias_restantes": 0,  # TODO
                "porcentaje_avance": 0,  # TODO
            }
        }
        
        if include_chart_data:
            # Datos para la gráfica
            response["chart_data"] = {
                "fechas": [],
                "valores_reales": [],
                "valores_predichos": [],
                "valores_suavizados": [],
                "sorteos_historicos": []  # Para mostrar en el legend
            }
        
        if include_history:
            response["historical_context"] = []
        
        return response
        
    except Exception as e:
        logger.error(f"Error obteniendo predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active")
async def get_active_sorteos():
    """
    Lista todos los sorteos activos agrupados por tipo
    """
    
    try:
        from backend.infrastructure.database.bigquery import BigQueryRepository
        
        bq_repo = BigQueryRepository()
        df_info = bq_repo.get_sorteos_info()
        
        # Filtrar activos
        df_activos = df_info[
            pd.to_datetime(df_info["FECHA_CIERRE"]) >= pd.Timestamp.now()
        ]
        
        # Agrupar por tipo
        result = {}
        for tipo in SorteoType:
            sorteos_tipo = df_activos[
                df_activos["SORTEO_GRUPO"] == tipo.value
            ]
            
            result[tipo.value] = [
                {
                    "nombre": row["NOMBRE"],
                    "id_sorteo": row["ID_SORTEO"],
                    "fecha_cierre": row["FECHA_CIERRE"].isoformat(),
                    "emision": row["EMISION"],
                    "numero_edicion": row["NUMERO_EDICION"]
                }
                for _, row in sorteos_tipo.iterrows()
            ]
        
        return {
            "success": True,
            "total_activos": len(df_activos),
            "sorteos_por_tipo": result
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo sorteos activos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/{nombre_sorteo}")
async def get_model_metrics(nombre_sorteo: str):
    """
    Obtiene métricas del modelo para un sorteo
    """
    
    metrics = prediction_service.get_model_metrics(nombre_sorteo)
    
    if "error" in metrics:
        raise HTTPException(status_code=404, detail=metrics["error"])
    
    return {
        "sorteo": nombre_sorteo,
        "metrics": metrics
    }

# Función helper para guardar en BigQuery
async def save_predictions_to_bigquery(predictions: Dict[str, Any]):
    """
    Guarda predicciones en BigQuery en background
    """
    try:
        from backend.infrastructure.database.bigquery import BigQueryRepository
        
        bq_repo = BigQueryRepository()
        
        for sorteo_name, pred_data in predictions.items():
            if isinstance(pred_data, pd.DataFrame):
                bq_repo.save_predictions(pred_data, "sorteo_predictions")
                
        logger.info(f"Predicciones guardadas en BigQuery")
        
    except Exception as e:
        logger.error(f"Error guardando en BigQuery: {e}")
        # No lanzar excepción porque es background task