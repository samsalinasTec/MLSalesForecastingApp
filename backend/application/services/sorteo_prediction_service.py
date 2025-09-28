"""
LEARNING NOTE: Servicio principal de predicción para sorteos
Orquesta todo el flujo de predicción
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from backend.infrastructure.ml.preprocessing import (
    process_sorteo_sales_data,
    aggregate_daily_sales,
    normalize_sales_percentages
)
from backend.infrastructure.ml.models.sorteo_regression import SorteoPolynomialRegressor
from backend.application.services.sorteo_selection_service import SorteoSelectionService
from backend.infrastructure.database.bigquery import BigQueryRepository

logger = logging.getLogger(__name__)

class SorteoPredictionService:
    """
    Servicio principal para predicciones de sorteos
    
    LEARNING NOTE: Este servicio reemplaza tu clase SorteosTecLRWM
    pero con mejor separación de responsabilidades
    """
    
    def __init__(self):
        self.bq_repo = BigQueryRepository()
        self.selection_service = SorteoSelectionService()
        self.regressors: Dict[str, SorteoPolynomialRegressor] = {}
        
    async def predict_all_active_sorteos(
        self,
        tipos_producto: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Genera predicciones para todos los sorteos activos
        
        Args:
            tipos_producto: Lista de tipos a procesar (TST, SMS, etc.)
                           Si None, procesa todos
        
        Returns:
            Dict con DataFrames de predicciones por tipo
        """
        
        if tipos_producto is None:
            tipos_producto = ["TST", "SMS", "AVT", "SOE", "DXV"]
        
        # Resetear sorteos seleccionados
        SorteoSelectionService.reset_selected_sorteos()
        
        # Obtener datos de BigQuery
        logger.info("Obteniendo datos de BigQuery...")
        df_fisico = self.bq_repo.get_sales_data_fisico()
        df_digital = self.bq_repo.get_sales_data_digital()
        df_info = self.bq_repo.get_sorteos_info()
        
        # Procesar datos
        logger.info("Procesando datos de ventas...")
        df_boletos = process_sorteo_sales_data(df_fisico, df_digital, df_info)
        df_agregado = aggregate_daily_sales(df_boletos)
        df_escalado = normalize_sales_percentages(df_agregado, df_info)
        
        # Generar predicciones por tipo
        predicciones = {}
        
        for tipo in tipos_producto:
            try:
                logger.info(f"Procesando tipo {tipo}...")
                
                # Puede haber múltiples sorteos activos del mismo tipo
                while True:
                    try:
                        # Seleccionar sorteo y entrenar
                        sorteo_info = self.selection_service.select_sorteo_for_prediction(
                            df_info, tipo
                        )
                        
                        # Generar predicción
                        df_pred = self._predict_single_sorteo(
                            sorteo_info,
                            df_escalado,
                            df_boletos
                        )
                        
                        # Guardar predicción
                        key = f"{tipo}_{sorteo_info['nombre_sorteo']}"
                        predicciones[key] = df_pred
                        
                    except ValueError as e:
                        # No hay más sorteos activos de este tipo
                        logger.info(f"Terminado con tipo {tipo}: {e}")
                        break
                        
            except Exception as e:
                logger.error(f"Error procesando tipo {tipo}: {e}")
                continue
        
        # Generar resumen
        df_resumen = self._generate_summary(predicciones, df_boletos, df_info)
        predicciones['resumen'] = df_resumen
        
        return predicciones
    
    def _predict_single_sorteo(
        self,
        sorteo_info: Dict,
        df_escalado: pd.DataFrame,
        df_boletos: pd.DataFrame # TODO: df_boletos podría usarse para métricas adicionales o validación pero no se usa por ahora
    ) -> pd.DataFrame:
        """
        Genera predicción para un sorteo específico
        
        LEARNING NOTE: Esta es tu lógica de predict() refactorizada
        """
        
        nombre_sorteo = sorteo_info['nombre_sorteo']
        sorteos_entrenamiento = sorteo_info['sorteos_entrenamiento']
        
        # Preparar datos de entrenamiento
        df_train = df_escalado[
            df_escalado["NOMBRE"].isin(sorteos_entrenamiento)
        ][["PORCENTAJE_DNAS", "PORCENTAJE_DE_AVANCE_SIN_MEMBRE", "NOMBRE"]]
        
        # Si el sorteo actual ya tiene datos, excluir el último punto
        if (df_escalado["NOMBRE"] == nombre_sorteo).any():
            df_entrena = df_escalado.drop(
                df_escalado[df_escalado["NOMBRE"] == nombre_sorteo].last_valid_index()
            )
        else:
            df_entrena = df_escalado
        
        # Entrenar modelo
        X_train = df_train[["PORCENTAJE_DNAS"]].values
        y_train = df_train["PORCENTAJE_DE_AVANCE_SIN_MEMBRE"].values
        
        # Crear y entrenar regresor
        regressor = SorteoPolynomialRegressor()
        regressor.train(X_train, y_train)
        
        # Guardar modelo para análisis posterior
        self.regressors[nombre_sorteo] = regressor
        
        # Calcular DNAS máximo
        try:
            max_dnas = int(df_entrena[df_entrena["NOMBRE"] == nombre_sorteo]["DNAS"].max())
        except:
            raise Exception(f"No hay datos de venta para {nombre_sorteo}")
        
        # Generar predicciones
        X_to_predict, y_sales = regressor.predict_sales_curve(
            max_dnas,
            sorteo_info['emision']
        )
        
        # Crear DataFrame de predicciones
        dnas_column = range(max_dnas, 0, -1)
        
        df_predicciones = pd.DataFrame({
            "ID_SORTEO": sorteo_info['id_sorteo'],
            "SORTEO": nombre_sorteo,
            "DNAS": dnas_column,
            "TALONES_ESTIMADOS": y_sales
        })
        
        # Ajustar con membresías
        df_predicciones = self._adjust_with_membresias(
            df_predicciones,
            df_escalado,
            nombre_sorteo,
            sorteos_entrenamiento
        )
        
        # Mapear fechas
        df_predicciones = self._map_dates(
            df_predicciones,
            sorteo_info['fecha_cierre'],
            max_dnas
        )
        
        return df_predicciones
    
    def _adjust_with_membresias(
        self,
        df_predicciones: pd.DataFrame,
        df_escalado: pd.DataFrame,
        nombre_sorteo: str,
        sorteos_entrenamiento: List[str]
    ) -> pd.DataFrame:
        """
        Ajusta predicciones con datos de membresías
        """
        # Obtener datos de membresías
        if (df_escalado["NOMBRE"] == nombre_sorteo).any():
            df_membresias = df_escalado[
                df_escalado["NOMBRE"] == nombre_sorteo
            ][["CANTIDAD_BOLETOS_MEMBRESIAS", "DNAS"]]
        else:
            # Usar último sorteo de entrenamiento
            df_membresias = df_escalado[
                df_escalado["NOMBRE"] == sorteos_entrenamiento[-1]
            ][["CANTIDAD_BOLETOS_MEMBRESIAS", "DNAS"]]
        
        # Merge con predicciones
        df_predicciones = pd.merge(
            df_predicciones,
            df_membresias,
            on="DNAS",
            how="left"
        ).fillna(0)
        
        # Calcular diferencias diarias
        df_predicciones['TALONES_DIARIOS_ESTIMADOS'] = df_predicciones['TALONES_ESTIMADOS'].diff()
        df_predicciones.iloc[0, 4] = df_predicciones["TALONES_ESTIMADOS"][0]
        
        # Ajustar con membresías
        df_predicciones["TALONES_DIARIOS_ESTIMADOS"] = (
            df_predicciones["CANTIDAD_BOLETOS_MEMBRESIAS"] +
            df_predicciones["TALONES_DIARIOS_ESTIMADOS"]
        )
        
        # Recalcular acumulados
        df_predicciones["TALONES_ESTIMADOS"] = df_predicciones.groupby("SORTEO")["TALONES_DIARIOS_ESTIMADOS"].cumsum()
        
        df_predicciones = df_predicciones.drop("CANTIDAD_BOLETOS_MEMBRESIAS", axis=1)
        
        return df_predicciones
    
    def _map_dates(
        self,
        df_predicciones: pd.DataFrame,
        fecha_cierre: datetime,
        max_dnas: int
    ) -> pd.DataFrame:
        """
        Mapea DNAS a fechas reales
        """
        fecha_inicio = fecha_cierre - timedelta(days=int(max_dnas))
        fecha_fin = fecha_cierre
        
        # Crear rango de fechas
        rango_fechas = pd.date_range(start=fecha_inicio, end=fecha_fin)
        
        df_predicciones["FECHA_MAPEADA"] = rango_fechas[1:]
        
        # Crear ID_SORTEO_DIA
        df_predicciones["FECHAAPOYO"] = (
            df_predicciones['FECHA_MAPEADA'] - pd.Timestamp('1899-12-30')
        ).dt.days
        
        df_predicciones["ID_SORTEO_DIA"] = (
            df_predicciones["FECHAAPOYO"].astype(str) +
            df_predicciones["ID_SORTEO"].astype(str)
        ).astype(np.int64)
        
        df_predicciones = df_predicciones.drop("FECHAAPOYO", axis=1)
        
        return df_predicciones
    
    def _generate_summary(
        self,
        predicciones: Dict[str, pd.DataFrame],
        df_boletos: pd.DataFrame,
        df_info: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Genera el resumen final estilo tu dfResumen
        
        LEARNING NOTE: Este es el CSV que quieres descargar
        """
        # TODO: Implementar generación de resumen
        # Por ahora retorno placeholder
        
        return pd.DataFrame({
            "NOMBRE": ["Implementar resumen"],
            "TALONES_ESTIMADOS": [0]
        })
    
    def get_model_metrics(self, nombre_sorteo: str) -> Dict:
        """
        Obtiene métricas del modelo para un sorteo
        """
        if nombre_sorteo in self.regressors:
            regressor = self.regressors[nombre_sorteo]
            return {
                "degree": regressor.best_degree,
                "r2_score": regressor.best_r2,
                "random_state": regressor.best_random_state
            }
        return {"error": "Modelo no encontrado"}