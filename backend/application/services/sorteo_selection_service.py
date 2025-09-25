"""
LEARNING NOTE: Servicio para selección inteligente de sorteos
Maneja la lógica de qué sorteos usar para entrenar
"""

import pandas as pd
import datetime as dt
from typing import List, Dict, Set, Optional
import logging

logger = logging.getLogger(__name__)

class SorteoSelectionService:
    """
    Maneja la selección de sorteos para entrenamiento y predicción
    
    LEARNING NOTE: Extraído de tu _set_sorteo_info
    """
    
    # Variable de clase para trackear sorteos ya procesados
    sorteos_seleccionados: Set[str] = set()
    
    @classmethod
    def reset_selected_sorteos(cls):
        """Reinicia la lista de sorteos seleccionados"""
        cls.sorteos_seleccionados = set()
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_active_sorteos(
        self,
        df_info: pd.DataFrame,
        tipo_producto: str
    ) -> pd.DataFrame:
        """
        Obtiene sorteos activos de un tipo específico
        
        LEARNING NOTE: Sorteo activo = fecha_cierre >= hoy
        """
        # Convertir fechas
        df_info["FECHA_CIERRE"] = pd.to_datetime(df_info["FECHA_CIERRE"], errors="coerce")
        
        # Filtrar por tipo y fecha
        df_activos = df_info[
            (df_info["SORTEO_GRUPO"] == tipo_producto) &
            (df_info["FECHA_CIERRE"] >= dt.datetime.now())
        ].copy()
        
        # Excluir ya procesados
        df_activos = df_activos[
            ~df_activos["NOMBRE"].isin(self.sorteos_seleccionados)
        ]
        
        if df_activos.empty:
            raise ValueError(f"No hay sorteos activos para {tipo_producto}")
        
        # Si hay múltiples, tomar el más próximo
        if len(df_activos) > 1:
            logger.info(f"Hay {len(df_activos)} sorteos activos de {tipo_producto}")
            df_activos = df_activos.sort_values("NUMERO_EDICION").head(1)
        
        return df_activos
    
    def find_similar_historical_sorteos(
        self,
        df_info: pd.DataFrame,
        sorteo_actual: pd.DataFrame,
        tipo_producto: str,
        n_sorteos: int = 4
    ) -> List[str]:
        """
        Encuentra sorteos históricos similares para entrenamiento
        
        LEARNING NOTE: Busca por cercanía de fechas normalizadas
        Similar a como se celebran año con año
        """
        # Filtrar sorteos del mismo tipo
        df_historicos = df_info[
            df_info["SORTEO_GRUPO"] == tipo_producto
        ].copy()
        
        # Normalizar fechas (mismo año para comparar)
        df_historicos["FECHA_NORMALIZADA"] = df_historicos["FECHA_CIERRE"].apply(
            lambda x: x.replace(year=2000) if not pd.isnull(x) else None
        )
        
        # Fecha del sorteo actual normalizada
        fecha_actual = sorteo_actual["FECHA_CIERRE"].iloc[0]
        fecha_actual_norm = fecha_actual.replace(year=2000)
        
        # Calcular diferencias en días
        df_historicos["DIAS_DIFF"] = abs(
            (fecha_actual_norm - df_historicos["FECHA_NORMALIZADA"]).dt.days
        )
        
        # Ordenar por cercanía y tomar top N
        df_historicos = df_historicos.sort_values("DIAS_DIFF").head(n_sorteos)
        
        # Ordenar por edición para mantener orden cronológico
        df_historicos = df_historicos.sort_values("NUMERO_EDICION", ascending=False)
        
        return df_historicos["NOMBRE"].tolist()
    
    def select_sorteo_for_prediction(
        self,
        df_info: pd.DataFrame,
        tipo_producto: str
    ) -> Dict[str, any]:
        """
        Selecciona un sorteo activo y sus históricos para predicción
        
        Returns:
            Dict con info del sorteo y sus históricos para entrenamiento
        """
        # Obtener sorteo activo
        df_activo = self.get_active_sorteos(df_info, tipo_producto)
        
        nombre_sorteo = df_activo["NOMBRE"].iloc[0]
        
        # Marcar como seleccionado
        self.sorteos_seleccionados.add(nombre_sorteo)
        
        # Buscar históricos similares
        sorteos_entrenamiento = self.find_similar_historical_sorteos(
            df_info, df_activo, tipo_producto
        )
        
        logger.info(f"Seleccionado sorteo: {nombre_sorteo}")
        logger.info(f"Sorteos para entrenamiento: {sorteos_entrenamiento}")
        
        return {
            "nombre_sorteo": nombre_sorteo,
            "id_sorteo": df_activo["ID_SORTEO"].iloc[0],
            "fecha_cierre": df_activo["FECHA_CIERRE"].iloc[0],
            "emision": df_activo["EMISION"].iloc[0],
            "sorteos_entrenamiento": sorteos_entrenamiento,
            "tipo_producto": tipo_producto
        }