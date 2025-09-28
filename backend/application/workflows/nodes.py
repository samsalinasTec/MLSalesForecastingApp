"""
LEARNING NOTE: Nodos del workflow para predicción de sorteos
Cada nodo es una función que procesa el estado
"""

import logging
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime

from backend.application.services.sorteo_prediction_service import SorteoPredictionService
from backend.application.services.sorteo_selection_service import SorteoSelectionService
from backend.application.services.smoothing_service import SmoothingService
from backend.infrastructure.external.vertex_ai import VertexAIClient
from backend.infrastructure.database.bigquery import BigQueryRepository
from backend.infrastructure.ml.preprocessing import (
    process_sorteo_sales_data,
    aggregate_daily_sales,
    normalize_sales_percentages
)

logger = logging.getLogger(__name__)

# Instancias de servicios
prediction_service = SorteoPredictionService()
smoothing_service = SmoothingService()
vertex_client = VertexAIClient()
bq_repo = BigQueryRepository()
selection_service = SorteoSelectionService()

async def fetch_data_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo 1: Obtiene y procesa datos de BigQuery
    
    LEARNING NOTE: Este nodo carga TODOS los datos necesarios
    y los preprocesa para los siguientes nodos
    """
    
    logger.info("Fetching sorteo data from BigQuery")
    state["processing_stage"] = "fetching"
    
    errors = []
    
    try:
        # Obtener datos de las tres tablas
        logger.info("Obteniendo datos de ventas físicas...")
        df_fisico = bq_repo.get_sales_data_fisico()
        
        logger.info("Obteniendo datos de ventas digitales...")
        df_digital = bq_repo.get_sales_data_digital()
        
        logger.info("Obteniendo información de sorteos...")
        df_info = bq_repo.get_sorteos_info()
        
        # Procesar y preparar datos
        logger.info("Procesando datos de sorteos...")
        df_boletos = process_sorteo_sales_data(
            df_fisico, 
            df_digital, 
            df_info,
            years_limit=3  # Últimos 3 años
        )
        
        logger.info("Agregando ventas diarias...")
        df_agregado = aggregate_daily_sales(df_boletos)
        
        logger.info("Normalizando porcentajes...")
        df_escalado = normalize_sales_percentages(df_agregado, df_info)
        
        # Guardar en el estado para uso posterior
        return {
            "sorteo_data": {
                "df_boletos": df_boletos,
                "df_escalado": df_escalado,
                "df_info": df_info,
                "df_agregado": df_agregado
            },
            "data_fetched": True,
            "errors": errors
        }
        
    except Exception as e:
        error_msg = f"Error fetching sorteo data: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        
        return {
            "data_fetched": False,
            "errors": errors,
            "processing_stage": "failed"
        }

async def predict_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo 2: Ejecuta predicciones ML para sorteos activos
    
    LEARNING NOTE: Este nodo aplica el modelo de regresión polinomial
    para cada sorteo activo del tipo especificado
    """
    
    logger.info("Running sorteo predictions")
    state["processing_stage"] = "predicting"
    
    if not state.get("data_fetched"):
        return {
            "errors": state.get("errors", []) + ["No data available for prediction"],
            "processing_stage": "failed"
        }
    
    predictions = {}
    errors = []
    
    try:
        # Obtener datos del estado
        sorteo_data = state["sorteo_data"]
        df_escalado = sorteo_data["df_escalado"]
        df_info = sorteo_data["df_info"]
        df_boletos = sorteo_data["df_boletos"]
        
        # Resetear sorteos seleccionados para esta ejecución
        SorteoSelectionService.reset_selected_sorteos()
        
        # Procesar cada tipo de sorteo solicitado
        for tipo in state.get("sorteo_types", []):
            logger.info(f"Procesando predicciones para tipo: {tipo}")
            
            # Puede haber múltiples sorteos activos del mismo tipo
            while True:
                try:
                    # Seleccionar sorteo activo y sus históricos
                    sorteo_info = selection_service.select_sorteo_for_prediction(
                        df_info, 
                        tipo
                    )
                    
                    # Ejecutar predicción para este sorteo
                    df_pred = prediction_service._predict_single_sorteo(
                        sorteo_info,
                        df_escalado,
                        df_boletos
                    )
                    
                    # Guardar predicción
                    key = f"{tipo}_{sorteo_info['nombre_sorteo']}"
                    predictions[key] = {
                        "sorteo_info": sorteo_info,
                        "prediction_df": df_pred,
                        "tipo": tipo,
                        "nombre": sorteo_info['nombre_sorteo'],
                        "fecha_cierre": sorteo_info['fecha_cierre'],
                        "emision": sorteo_info['emision']
                    }
                    
                    logger.info(f"Predicción completada para: {sorteo_info['nombre_sorteo']}")
                    
                except ValueError as e:
                    # No hay más sorteos activos de este tipo
                    logger.info(f"No hay más sorteos activos para tipo {tipo}")
                    break
                    
                except Exception as e:
                    error_msg = f"Error prediciendo sorteo tipo {tipo}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    break
        
        return {
            "predictions": predictions,
            "raw_predictions": predictions,  # Para compatibilidad
            "prediction_count": len(predictions),
            "errors": state.get("errors", []) + errors
        }
        
    except Exception as e:
        error_msg = f"Error general en predicción: {str(e)}"
        logger.error(error_msg)
        return {
            "predictions": {},
            "errors": state.get("errors", []) + [error_msg],
            "processing_stage": "failed"
        }

async def smooth_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo 3: Aplica suavizado a las predicciones
    
    LEARNING NOTE: Este nodo detecta cambios bruscos y los suaviza
    para mantener consistencia día a día
    """
    
    logger.info("Applying smoothing to predictions")
    state["processing_stage"] = "smoothing"
    
    if not state.get("apply_smoothing", True):
        logger.info("Smoothing skipped by configuration")
        return {"needs_llm_explanation": False}
    
    predictions = state.get("predictions", {})
    smoothed_predictions = {}
    was_smoothed = {}
    needs_explanation = False
    errors = []
    
    try:
        for key, pred_data in predictions.items():
            try:
                sorteo_nombre = pred_data['nombre']
                df_pred = pred_data['prediction_df']
                
                # Obtener último valor estimado (más reciente)
                if not df_pred.empty:
                    latest_prediction = df_pred.iloc[0]['TALONES_ESTIMADOS']
                    
                    # Crear objeto Prediction compatible con el servicio
                    from backend.domain.models.prediction import Prediction
                    
                    temp_prediction = Prediction(
                        product_id=sorteo_nombre,  # Usamos nombre como ID
                        original_value=latest_prediction,
                        final_value=latest_prediction,
                        prediction_date=datetime.now(),
                        was_smoothed=False,
                        change_percentage=0.0,
                        historical_context=[]
                    )
                    
                    # Aplicar suavizado
                    smoothed_pred, applied, message = smoothing_service.apply_smoothing(
                        temp_prediction
                    )
                    
                    # Actualizar DataFrame si se aplicó suavizado
                    if applied:
                        # Ajustar todos los valores proporcionalmente
                        factor = smoothed_pred.final_value / latest_prediction
                        df_pred['TALONES_ESTIMADOS_FACTOR'] = df_pred['TALONES_ESTIMADOS'] * factor
                        df_pred['TALONES_DIARIOS_ESTIMADOS_FACTOR'] = df_pred['TALONES_DIARIOS_ESTIMADOS'] * factor
                        
                        needs_explanation = True
                        logger.info(f"Suavizado aplicado a {sorteo_nombre}: {message}")
                    
                    # Guardar resultados
                    smoothed_predictions[key] = {
                        **pred_data,
                        "prediction_df": df_pred,
                        "was_smoothed": applied,
                        "smoothing_message": message,
                        "original_value": temp_prediction.original_value,
                        "final_value": smoothed_pred.final_value
                    }
                    was_smoothed[key] = applied
                    
            except Exception as e:
                error_msg = f"Error smoothing {key}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                # Mantener predicción original si falla el suavizado
                smoothed_predictions[key] = pred_data
                was_smoothed[key] = False
        
        return {
            "predictions": smoothed_predictions,  # Actualizar con versiones suavizadas
            "smoothed_predictions": smoothed_predictions,
            "was_smoothed": was_smoothed,
            "needs_llm_explanation": needs_explanation,
            "errors": state.get("errors", []) + errors
        }
        
    except Exception as e:
        error_msg = f"Error general en suavizado: {str(e)}"
        logger.error(error_msg)
        return {
            "predictions": predictions,  # Mantener originales
            "needs_llm_explanation": False,
            "errors": state.get("errors", []) + [error_msg]
        }

async def explain_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo 4: Genera explicaciones con LLM para cambios significativos
    
    LEARNING NOTE: Solo se ejecuta si hubo suavizado
    Usa Vertex AI (Gemini) para generar insights
    """
    
    logger.info("Generating LLM explanations for smoothed predictions")
    state["processing_stage"] = "explaining"
    
    predictions = state.get("predictions", {})
    was_smoothed = state.get("was_smoothed", {})
    explanations = {}
    errors = []
    
    try:
        # Generar explicaciones solo para sorteos suavizados
        for key, was_smoothed_flag in was_smoothed.items():
            if was_smoothed_flag:
                try:
                    pred_data = predictions[key]
                    
                    # Obtener contexto histórico (últimos 10 valores si están disponibles)
                    historical_values = smoothing_service.history.get(
                        pred_data['nombre'], 
                        []
                    )
                    historical_list = list(historical_values) if historical_values else []
                    
                    # Calcular cambio porcentual
                    original = pred_data.get('original_value', 0)
                    final = pred_data.get('final_value', 0)
                    change_pct = ((original - final) / final * 100) if final else 0
                    
                    # Generar explicación con LLM
                    explanation = await vertex_client.analyze_prediction_change(
                        product_id=pred_data['nombre'],
                        original_value=original,
                        smoothed_value=final,
                        historical_values=historical_list[-10:],  # Últimos 10
                        change_percentage=change_pct
                    )
                    
                    explanations[key] = explanation
                    
                    # Actualizar predicción con explicación
                    predictions[key]['llm_explanation'] = explanation
                    
                    logger.info(f"Explicación generada para {pred_data['nombre']}")
                    
                except Exception as e:
                    error_msg = f"Error generando explicación para {key}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    explanations[key] = "No se pudo generar explicación"
        
        # Generar insights generales del mercado
        try:
            # Preparar datos agregados para análisis
            predictions_summary = {}
            for key, pred_data in predictions.items():
                if 'prediction_df' in pred_data and not pred_data['prediction_df'].empty:
                    predictions_summary[pred_data['nombre']] = pred_data['prediction_df'].iloc[0]['TALONES_ESTIMADOS']
            
            market_insights = await vertex_client.get_market_insights(
                product_id="todos",
                predictions=predictions_summary
            )
            
            logger.info("Insights de mercado generados")
            
        except Exception as e:
            logger.warning(f"No se pudieron generar insights de mercado: {str(e)}")
            market_insights = {
                "tendencia": "No disponible",
                "recomendaciones": []
            }
        
        return {
            "predictions": predictions,  # Ya actualizadas con explicaciones
            "explanations": explanations,
            "market_insights": market_insights,
            "errors": state.get("errors", []) + errors
        }
        
    except Exception as e:
        error_msg = f"Error general generando explicaciones: {str(e)}"
        logger.error(error_msg)
        return {
            "predictions": predictions,
            "explanations": {},
            "market_insights": {},
            "errors": state.get("errors", []) + [error_msg]
        }

async def compile_results_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo Final: Compila todos los resultados en formato final
    
    LEARNING NOTE: Este nodo prepara los datos para la respuesta final
    y genera el resumen descargable
    """
    
    logger.info("Compiling final results")
    state["processing_stage"] = "compiling"
    
    predictions = state.get("predictions", {})
    final_predictions = {}
    all_predictions_list = []
    
    try:
        # Compilar predicciones en formato final
        for key, pred_data in predictions.items():
            if 'prediction_df' in pred_data:
                df = pred_data['prediction_df']
                
                # Preparar datos para el frontend
                final_predictions[key] = {
                    "sorteo_info": {
                        "nombre": pred_data['nombre'],
                        "tipo": pred_data['tipo'],
                        "fecha_cierre": pred_data['fecha_cierre'].isoformat() if hasattr(pred_data['fecha_cierre'], 'isoformat') else str(pred_data['fecha_cierre']),
                        "emision": pred_data['emision']
                    },
                    "prediction_summary": {
                        "total_estimado": df['TALONES_ESTIMADOS'].iloc[0] if not df.empty else 0,
                        "dias_restantes": len(df),
                        "porcentaje_avance": (df['TALONES_ESTIMADOS'].iloc[0] / pred_data['emision'] * 100) if not df.empty and pred_data['emision'] > 0 else 0
                    },
                    "was_smoothed": pred_data.get('was_smoothed', False),
                    "smoothing_message": pred_data.get('smoothing_message', ''),
                    "llm_explanation": pred_data.get('llm_explanation', ''),
                    "chart_data": {
                        "dates": df['FECHA_MAPEADA'].dt.strftime('%Y-%m-%d').tolist() if 'FECHA_MAPEADA' in df.columns else [],
                        "values": df['TALONES_ESTIMADOS'].tolist(),
                        "daily_values": df['TALONES_DIARIOS_ESTIMADOS'].tolist() if 'TALONES_DIARIOS_ESTIMADOS' in df.columns else []
                    }
                }
                
                # Agregar a lista para resumen
                all_predictions_list.append(df)
        
        # Generar resumen (el CSV que quieres descargar)
        df_resumen = _generate_summary_dataframe(predictions, state.get("sorteo_data", {}))
        
        # Calcular tiempo de procesamiento
        end_time = datetime.now()
        start_time = state.get("start_time", end_time)
        processing_time = (end_time - start_time).total_seconds()
        
        # Compilar estado final
        return {
            "final_predictions": final_predictions,
            "summary_df": df_resumen,
            "market_insights": state.get("market_insights", {}),
            "end_time": end_time,
            "processing_time": processing_time,
            "processing_stage": "completed",
            "total_sorteos_processed": len(final_predictions),
            "errors": state.get("errors", [])
        }
        
    except Exception as e:
        error_msg = f"Error compilando resultados: {str(e)}"
        logger.error(error_msg)
        return {
            "final_predictions": {},
            "processing_stage": "failed",
            "errors": state.get("errors", []) + [error_msg]
        }

def _generate_summary_dataframe(predictions: Dict, sorteo_data: Dict) -> pd.DataFrame:
    """
    Genera el DataFrame de resumen estilo tu dfResumen original
    
    LEARNING NOTE: Esta función privada genera el CSV descargable
    con el resumen de todas las predicciones
    """
    
    try:
        summary_rows = []
        
        for key, pred_data in predictions.items():
            if 'prediction_df' in pred_data:
                df = pred_data['prediction_df']
                
                if not df.empty:
                    row = {
                        "NOMBRE": pred_data['nombre'],
                        "TIPO": pred_data['tipo'],
                        "EMISION": pred_data['emision'],
                        "FECHA_CIERRE": pred_data['fecha_cierre'],
                        "TALONES_ESTIMADOS_TOTAL": df['TALONES_ESTIMADOS'].iloc[0],
                        "PORCENTAJE_ESTIMADO": (df['TALONES_ESTIMADOS'].iloc[0] / pred_data['emision'] * 100) if pred_data['emision'] > 0 else 0,
                        "DIAS_RESTANTES": len(df),
                        "FUE_SUAVIZADO": "Sí" if pred_data.get('was_smoothed', False) else "No",
                        "FECHA_PREDICCION": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    summary_rows.append(row)
        
        if summary_rows:
            df_resumen = pd.DataFrame(summary_rows)
            # Ordenar por fecha de cierre
            df_resumen = df_resumen.sort_values('FECHA_CIERRE')
            return df_resumen
        else:
            # DataFrame vacío con estructura correcta
            return pd.DataFrame(columns=[
                "NOMBRE", "TIPO", "EMISION", "FECHA_CIERRE", 
                "TALONES_ESTIMADOS_TOTAL", "PORCENTAJE_ESTIMADO",
                "DIAS_RESTANTES", "FUE_SUAVIZADO", "FECHA_PREDICCION"
            ])
            
    except Exception as e:
        logger.error(f"Error generando resumen: {str(e)}")
        return pd.DataFrame()