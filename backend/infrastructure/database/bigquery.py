"""
LEARNING NOTE: Actualizado para trabajar con sorteos en lugar de productos
"""

from google.cloud import bigquery
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from backend.core.config import settings
from backend.core.exceptions import DataNotFoundException
import logging

logger = logging.getLogger(__name__)

class BigQueryRepository:
    """
    Repository para acceso a datos de sorteos en BigQuery
    """
    
    def __init__(self):
        self.client = bigquery.Client(project=settings.gcp_project_id)
        self.dataset_id = settings.bigquery_dataset
        
    def get_sorteos_info(self) -> pd.DataFrame:
        """
        Obtiene información de todos los sorteos (tabla DMSorteos)
        """
        query = f"""
        SELECT 
            ID_SORTEO,
            ID_PRODUCTO,
            NUMERO_EDICION,
            NOMBRE,
            EMISION,
            PRECIO_UNITARIO,
            FECHA_INICIO,
            FECHA_CIERRE,
            PERMISO_GOBERNACION,
            SORTEO_GRUPO,
            SORTEO_SIGLAS
        FROM 
            `{settings.gcp_project_id}.{self.dataset_id}.DMSorteos`
        ORDER BY 
            FECHA_CIERRE DESC
        """
        
        try:
            df = self.client.query(query).to_dataframe()
            logger.info(f"Recuperados {len(df)} sorteos de DMSorteos")
            return df
        except Exception as e:
            logger.error(f"Error obteniendo sorteos: {e}")
            raise
    
    def get_sales_data_digital(self) -> pd.DataFrame:
        """
        Obtiene datos de ventas digitales (FCVentas_digital)
        """
        query = f"""
        SELECT 
            CANAL_DIG,
            FECHAREGISTRO,
            INGRESO_POR_BOLETO,
            ID_CLIENTE,
            CANTIDAD_BOLETOS,
            ID_SORTEO,
            ID_SORTEO_DIA,
            ID_SORTEO_CLIENTE,
            ID_OFICINA,
            ERETAILER
        FROM 
            `{settings.gcp_project_id}.{self.dataset_id}.FCVentas_digital`
        WHERE
            FECHAREGISTRO >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 YEAR)
        """
        
        try:
            df = self.client.query(query).to_dataframe()
            logger.info(f"Recuperadas {len(df)} ventas digitales")
            return df
        except Exception as e:
            logger.error(f"Error obteniendo ventas digitales: {e}")
            raise
    
    def get_sales_data_fisico(self) -> pd.DataFrame:
        """
        Obtiene datos de ventas físicas (FCVentas_fisico)
        """
        query = f"""
        SELECT 
            ID_SORTEO,
            ID_COLAB,
            FECHAREGISTRO,
            CANTIDAD_BOLETOS,
            INGRESO_POR_BOLETOS,
            CANAL_TRADICIONAL,
            ID_SORTEO_DIA,
            ID_SORTEO_COLAB
        FROM 
            `{settings.gcp_project_id}.{self.dataset_id}.FCVentas_Fisico`
        WHERE
            FECHAREGISTRO >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 YEAR)
        """
        
        try:
            df = self.client.query(query).to_dataframe()
            logger.info(f"Recuperadas {len(df)} ventas físicas")
            return df
        except Exception as e:
            logger.error(f"Error obteniendo ventas físicas: {e}")
            raise
    
    def get_active_sorteos_by_type(self, tipo: str) -> pd.DataFrame:
        """
        Obtiene sorteos activos de un tipo específico
        """
        query = f"""
        SELECT 
            *
        FROM 
            `{settings.gcp_project_id}.{self.dataset_id}.DMSorteos`
        WHERE 
            SORTEO_GRUPO = @tipo
            AND FECHA_CIERRE >= CURRENT_DATE()
        ORDER BY 
            NUMERO_EDICION DESC
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("tipo", "STRING", tipo)
            ]
        )
        
        try:
            df = self.client.query(query, job_config=job_config).to_dataframe()
            return df
        except Exception as e:
            logger.error(f"Error obteniendo sorteos activos: {e}")
            raise
    
    def save_predictions(self, predictions: Dict[str, Any], timestamp: datetime):
        """
        Guarda predicciones detalladas en BigQuery
        
        Args:
            predictions: Diccionario con las predicciones por sorteo
            timestamp: Marca de tiempo de la predicción
        """
        table_id = f"{settings.gcp_project_id}.{self.dataset_id}.predictions_details"
        
        try:
            # Preparar datos para BigQuery
            rows_to_insert = []
            
            for sorteo_key, pred_data in predictions.items():
                # Extraer información del sorteo
                sorteo_info = pred_data.get("sorteo_info", {})
                summary = pred_data.get("prediction_summary", {})
                
                row = {
                    "sorteo_key": sorteo_key,
                    "nombre": sorteo_info.get("nombre"),
                    "tipo": sorteo_info.get("tipo"),
                    "fecha_cierre": sorteo_info.get("fecha_cierre"),
                    "emision": sorteo_info.get("emision"),
                    "total_estimado": summary.get("total_estimado"),
                    "porcentaje_avance": summary.get("porcentaje_avance"),
                    "dias_restantes": summary.get("dias_restantes"),
                    "was_smoothed": pred_data.get("was_smoothed", False),
                    "llm_explanation": pred_data.get("llm_explanation", ""),
                    "timestamp": timestamp,
                    "created_at": datetime.now()
                }
                
                rows_to_insert.append(row)
            
            if rows_to_insert:
                # Crear tabla si no existe
                self._ensure_predictions_table_exists()
                
                # Insertar datos
                table = self.client.get_table(table_id)
                errors = self.client.insert_rows_json(table, rows_to_insert)
                
                if errors:
                    logger.error(f"Error insertando predicciones: {errors}")
                else:
                    logger.info(f"Guardadas {len(rows_to_insert)} predicciones en BigQuery")
                    
        except Exception as e:
            logger.error(f"Error guardando predicciones: {e}")
            # No lanzar excepción porque es background task
    
    def save_prediction_summary(self, summary_data: List[Dict], timestamp: datetime):
        """
        Guarda el resumen de predicciones (df_resumen) en BigQuery
        
        Args:
            summary_data: Lista de diccionarios con el resumen
            timestamp: Marca de tiempo de la predicción
        """
        table_id = f"{settings.gcp_project_id}.{self.dataset_id}.predictions_summary"
        
        try:
            # Agregar timestamp a cada fila
            for row in summary_data:
                row["timestamp"] = timestamp
                row["created_at"] = datetime.now()
            
            if summary_data:
                # Crear tabla si no existe
                self._ensure_summary_table_exists()
                
                # Insertar datos
                table = self.client.get_table(table_id)
                errors = self.client.insert_rows_json(table, summary_data)
                
                if errors:
                    logger.error(f"Error insertando resumen: {errors}")
                else:
                    logger.info(f"Guardado resumen con {len(summary_data)} filas en BigQuery")
                    
        except Exception as e:
            logger.error(f"Error guardando resumen: {e}")
            # No lanzar excepción porque es background task
    
    async def get_latest_summary(self) -> Optional[List[Dict]]:
        """
        Obtiene el resumen más reciente de predicciones
        
        Returns:
            Lista de diccionarios con el resumen o None si no existe
        """
        query = f"""
        SELECT 
            * EXCEPT(timestamp, created_at)
        FROM 
            `{settings.gcp_project_id}.{self.dataset_id}.predictions_summary`
        WHERE 
            DATE(timestamp) = (
                SELECT MAX(DATE(timestamp))
                FROM `{settings.gcp_project_id}.{self.dataset_id}.predictions_summary`
            )
        ORDER BY 
            timestamp DESC
        """
        
        try:
            df = self.client.query(query).to_dataframe()
            
            if df.empty:
                return None
                
            # Convertir a lista de diccionarios
            return df.to_dict('records')
            
        except Exception as e:
            logger.warning(f"Error obteniendo resumen: {e}")
            return None
    
    async def get_latest_predictions(self, sorteo_nombre: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Obtiene las predicciones detalladas más recientes
        
        Args:
            sorteo_nombre: Filtrar por nombre de sorteo específico
            
        Returns:
            Lista de diccionarios con las predicciones o None si no existe
        """
        where_clause = ""
        if sorteo_nombre:
            where_clause = f"AND nombre = '{sorteo_nombre}'"
        
        query = f"""
        SELECT 
            * EXCEPT(timestamp, created_at)
        FROM 
            `{settings.gcp_project_id}.{self.dataset_id}.predictions_details`
        WHERE 
            DATE(timestamp) = (
                SELECT MAX(DATE(timestamp))
                FROM `{settings.gcp_project_id}.{self.dataset_id}.predictions_details`
            )
            {where_clause}
        ORDER BY 
            fecha_cierre
        """
        
        try:
            df = self.client.query(query).to_dataframe()
            
            if df.empty:
                return None
                
            return df.to_dict('records')
            
        except Exception as e:
            logger.warning(f"Error obteniendo predicciones: {e}")
            return None
    
    def _ensure_predictions_table_exists(self):
        """
        Crea la tabla de predicciones si no existe
        """
        table_id = f"{settings.gcp_project_id}.{self.dataset_id}.predictions_details"
        
        schema = [
            bigquery.SchemaField("sorteo_key", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("nombre", "STRING"),
            bigquery.SchemaField("tipo", "STRING"),
            bigquery.SchemaField("fecha_cierre", "STRING"),
            bigquery.SchemaField("emision", "FLOAT"),
            bigquery.SchemaField("total_estimado", "FLOAT"),
            bigquery.SchemaField("porcentaje_avance", "FLOAT"),
            bigquery.SchemaField("dias_restantes", "INTEGER"),
            bigquery.SchemaField("was_smoothed", "BOOLEAN"),
            bigquery.SchemaField("llm_explanation", "STRING"),
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
            bigquery.SchemaField("created_at", "TIMESTAMP"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        
        try:
            self.client.create_table(table)
            logger.info(f"Tabla {table_id} creada")
        except Exception as e:
            # La tabla ya existe, está bien
            pass
    
    def _ensure_summary_table_exists(self):
        """
        Crea la tabla de resumen si no existe
        """
        table_id = f"{settings.gcp_project_id}.{self.dataset_id}.predictions_summary"
        
        # El schema dependerá de tu df_resumen
        # Ajusta según los campos que generes
        schema = [
            bigquery.SchemaField("NOMBRE", "STRING"),
            bigquery.SchemaField("TIPO", "STRING"),
            bigquery.SchemaField("EMISION", "FLOAT"),
            bigquery.SchemaField("FECHA_CIERRE", "STRING"),
            bigquery.SchemaField("TALONES_ESTIMADOS_TOTAL", "FLOAT"),
            bigquery.SchemaField("PORCENTAJE_ESTIMADO", "FLOAT"),
            bigquery.SchemaField("DIAS_RESTANTES", "INTEGER"),
            bigquery.SchemaField("FUE_SUAVIZADO", "STRING"),
            bigquery.SchemaField("FECHA_PREDICCION", "STRING"),
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
            bigquery.SchemaField("created_at", "TIMESTAMP"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        
        try:
            self.client.create_table(table)
            logger.info(f"Tabla {table_id} creada")
        except Exception as e:
            # La tabla ya existe, está bien
            pass