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
            `{settings.gcp_project_id}.{self.dataset_id}.FCVentas_fisico`
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
    
    def save_predictions(self, predictions_df: pd.DataFrame, table_name: str = "predictions"):
        """
        Guarda predicciones en BigQuery
        
        TODO: Definir esquema de tabla de predicciones
        """
        table_id = f"{settings.gcp_project_id}.{self.dataset_id}.{table_name}"
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            schema_update_options=[
                bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
            ]
        )
        
        try:
            job = self.client.load_table_from_dataframe(
                predictions_df,
                table_id,
                job_config=job_config
            )
            job.result()
            logger.info(f"Predicciones guardadas en {table_id}")
        except Exception as e:
            logger.error(f"Error guardando predicciones: {e}")
            raise