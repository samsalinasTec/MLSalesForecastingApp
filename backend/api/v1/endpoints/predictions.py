"""
LEARNING NOTE: Endpoints actualizados para sorteos
Cambios principales: products → sorteos
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import List, Optional, Dict, Any
from datetime import date, datetime
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
from backend.infrastructure.database.bigquery import BigQueryRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
    responses={404: {"description": "Not found"}}
)

# Instancia del servicio
prediction_service = SorteoPredictionService()
bq_repo = BigQueryRepository()

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
        
        # Guardar en BigQuery en background si está configurado
        if settings.save_predictions_to_bq and request.save_to_bq:
            # Guardar predicciones detalladas
            background_tasks.add_task(
                bq_repo.save_predictions,
                result["final_predictions"],
                datetime.now()
            )
            
            # Guardar resumen si existe
            if "summary_data" in result and result["summary_data"]:
                background_tasks.add_task(
                    bq_repo.save_prediction_summary,
                    result["summary_data"],
                    datetime.now()
                )
                
        return PredictionResponse(
            success=True,
            predictions=result.get("final_predictions", {}),
            processing_time=result.get("processing_time", 0),
            metadata={
                "sorteos_procesados": len(result.get("final_predictions", {})),
                "tipos_procesados": request.sorteo_types,
                "summary_available": bool(result.get("summary_data"))
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
    Descarga el resumen de predicciones más reciente como CSV
    
    LEARNING NOTE: Recupera el último resumen guardado en BigQuery
    """
    
    try:
        # Obtener el resumen más reciente de BigQuery
        summary_data = await bq_repo.get_latest_summary()
        
        if not summary_data:
            raise HTTPException(
                status_code=404,
                detail="No hay resumen de predicciones disponible. Por favor ejecuta primero las predicciones."
            )
        
        # Convertir a DataFrame
        df_resumen = pd.DataFrame(summary_data)
        
        # Convertir a CSV
        stream = StringIO()
        df_resumen.to_csv(stream, index=False)
        stream.seek(0)
        
        # Preparar nombre del archivo con timestamp
        filename = f"resumen_predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error descargando resumen: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/details/download")
async def download_predictions_details(
    sorteo_nombre: Optional[str] = Query(None, description="Nombre específico del sorteo")
):
    """
    Descarga las predicciones detalladas como CSV
    
    Args:
        sorteo_nombre: Si se especifica, descarga solo ese sorteo.
                      Si no, descarga todas las predicciones recientes.
    """
    
    try:
        # Obtener predicciones detalladas más recientes
        predictions_data = await bq_repo.get_latest_predictions(sorteo_nombre)
        
        if not predictions_data:
            message = f"No hay predicciones disponibles"
            if sorteo_nombre:
                message += f" para el sorteo {sorteo_nombre}"
            message += ". Por favor ejecuta primero las predicciones."
            
            raise HTTPException(status_code=404, detail=message)
        
        # Preparar CSV con los datos detallados
        # La estructura dependerá de cómo guardes las predicciones
        df_details = pd.DataFrame(predictions_data)
        
        # Convertir a CSV
        stream = StringIO()
        df_details.to_csv(stream, index=False)
        stream.seek(0)
        
        # Preparar nombre del archivo
        if sorteo_nombre:
            filename = f"predicciones_{sorteo_nombre}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            filename = f"predicciones_detalladas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error descargando predicciones detalladas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active")
async def get_active_sorteos():
    """
    Obtiene lista de sorteos activos agrupados por tipo
    """
    
    try:
        from backend.infrastructure.database.bigquery import BigQueryRepository
        
        bq_repo = BigQueryRepository()
        df_info = bq_repo.get_sorteos_info()
        
        # Filtrar solo activos
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

