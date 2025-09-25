"""
LEARNING NOTE: Preprocessing para sorteos
Mantiene tu lógica pero mejor organizada
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ========== TU PREPROCESSING ACTUAL ==========

def calculate_dnas(fecha_cierre: pd.Series, fecha_registro: pd.Series) -> pd.Series:
    """
    Calcula DNAS (Days Not Available for Sale)
    
    LEARNING NOTE: Esta es tu función DNASColumn original
    Días restantes hasta el cierre del sorteo
    """
    return (fecha_cierre - fecha_registro).dt.days

def process_sorteo_sales_data(
    df_fisico: pd.DataFrame,
    df_digital: pd.DataFrame,
    df_info: pd.DataFrame,
    years_limit: int = 3
) -> pd.DataFrame:
    """
    Procesa y combina datos de ventas físicas y digitales
    
    LEARNING NOTE: Extraído de tu script principal
    Esta es la lógica de preparación inicial
    """
    
    # Renombrar columnas para unificar
    df_digital_renamed = df_digital[
        ["ID_SORTEO", "ID_SORTEO_DIA", "FECHAREGISTRO", "CANTIDAD_BOLETOS", "CANAL_DIG"]
    ].rename({"CANAL_DIG": "CANAL"}, axis=1)
    
    df_fisico_renamed = df_fisico[
        ["ID_SORTEO", "ID_SORTEO_DIA", "FECHAREGISTRO", "CANTIDAD_BOLETOS", "CANAL_TRADICIONAL"]
    ].rename({"CANAL_TRADICIONAL": "CANAL"}, axis=1)
    
    # Combinar físico y digital
    df_boletos = pd.concat([df_fisico_renamed, df_digital_renamed])
    
    # Convertir fechas
    df_boletos["FECHAREGISTRO"] = pd.to_datetime(df_boletos["FECHAREGISTRO"], format='%Y-%m-%d')
    df_info["FECHA_CIERRE"] = pd.to_datetime(df_info["FECHA_CIERRE"], format='%Y-%m-%d')
    
    # Filtrar por años límite
    fecha_limite = datetime.now() - timedelta(days=years_limit * 365)
    df_info = df_info[df_info["FECHA_CIERRE"] >= fecha_limite]
    
    # Merge con información del sorteo
    df_boletos = pd.merge(
        df_boletos,
        df_info[["ID_SORTEO", "NOMBRE", "FECHA_CIERRE", "PRECIO_UNITARIO"]],
        on="ID_SORTEO",
        how="left"
    )
    
    # Calcular DNAS
    df_boletos["DNAS"] = calculate_dnas(df_boletos["FECHA_CIERRE"], df_boletos["FECHAREGISTRO"])
    
    # Ordenar por fecha
    df_boletos = df_boletos.sort_values("FECHAREGISTRO")
    
    # Separar membresías
    df_boletos['CANTIDAD_BOLETOS_MEMBRESIAS'] = df_boletos.apply(
        lambda row: row['CANTIDAD_BOLETOS'] if row['CANAL'] == 'Membresias' else 0,
        axis=1
    )
    df_boletos['CANTIDAD_BOLETOS_SIN_MEMBRE'] = df_boletos.apply(
        lambda row: row['CANTIDAD_BOLETOS'] - row['CANTIDAD_BOLETOS_MEMBRESIAS'],
        axis=1
    )
    
    df_boletos = df_boletos.drop("CANAL", axis=1)
    
    return df_boletos

def aggregate_daily_sales(df_boletos: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega ventas por día y maneja días negativos
    
    LEARNING NOTE: Esta lógica es crítica para tu negocio
    Los días negativos (antes del lanzamiento) se suman al día 1
    """
    
    # Agrupar por nombre y DNAS
    df_agregado = df_boletos.groupby(["NOMBRE", "DNAS"]).agg(
        CANTIDAD_BOLETOS=("CANTIDAD_BOLETOS", "sum"),
        CANTIDAD_BOLETOS_SIN_MEMBRE=("CANTIDAD_BOLETOS_SIN_MEMBRE", "sum"),
        CANTIDAD_BOLETOS_MEMBRESIAS=("CANTIDAD_BOLETOS_MEMBRESIAS", "sum")
    ).reset_index()
    
    # Manejar días negativos (ventas antes del lanzamiento oficial)
    sorteos = df_agregado["NOMBRE"].unique()
    
    for sorteo in sorteos:
        # Sumar días negativos al día 1
        mask_negativo = (df_agregado['DNAS'] <= 0) & (df_agregado["NOMBRE"] == sorteo)
        mask_dia1 = (df_agregado['DNAS'] == 1) & (df_agregado["NOMBRE"] == sorteo)
        
        for col in ['CANTIDAD_BOLETOS_SIN_MEMBRE', 'CANTIDAD_BOLETOS_MEMBRESIAS', 'CANTIDAD_BOLETOS']:
            boletos_negativos = df_agregado.loc[mask_negativo, col].sum()
            df_agregado.loc[mask_dia1, col] += boletos_negativos
    
    # Eliminar días negativos
    df_agregado = df_agregado[df_agregado['DNAS'] > 0]
    
    # Ordenar por DNAS descendente
    df_agregado = df_agregado.sort_values("DNAS", ascending=False)
    
    # Calcular acumulados
    df_agregado["BOLETOS_ACUMULADOS_SIN_MEMBRE"] = df_agregado.groupby("NOMBRE")["CANTIDAD_BOLETOS_SIN_MEMBRE"].cumsum()
    df_agregado["BOLETOS_ACUMULADOS_CON_MEMBRE"] = df_agregado.groupby("NOMBRE")["CANTIDAD_BOLETOS"].cumsum()
    
    return df_agregado

def normalize_sales_percentages(df_agregado: pd.DataFrame, df_info: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza ventas a porcentajes para comparación entre sorteos
    
    LEARNING NOTE: Normalización clave para comparar sorteos de diferente tamaño
    """
    
    # Merge con información del sorteo
    df_escalado = pd.merge(
        df_agregado,
        df_info[["SORTEO_GRUPO", "NOMBRE", "EMISION", "ID_SORTEO"]],
        on="NOMBRE",
        how="left"
    )
    
    # Calcular porcentajes de avance
    df_escalado["PORCENTAJE_DE_AVANCE_SIN_MEMBRE"] = (
        df_escalado["BOLETOS_ACUMULADOS_SIN_MEMBRE"] / df_escalado["EMISION"]
    )
    df_escalado["PORCENTAJE_DE_AVANCE_CON_MEMBRE"] = (
        df_escalado["BOLETOS_ACUMULADOS_CON_MEMBRE"] / df_escalado["EMISION"]
    )
    
    # Calcular porcentaje de DNAS (tiempo transcurrido)
    max_days = df_escalado.groupby('NOMBRE')['DNAS'].transform('max') - 1
    df_escalado["PORCENTAJE_DNAS"] = (max_days - (df_escalado["DNAS"] - 1)) / max_days
    
    # Ordenar
    df_escalado = df_escalado.sort_values(["NOMBRE", "DNAS"], ascending=False)
    
    return df_escalado

# ========== PREPROCESSING ADICIONAL SUGERIDO (PLACEHOLDER) ==========

def detect_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.Series:
    """
    Detecta outliers usando IQR
    
    LEARNING NOTE: Mantén esto como referencia para cuando lo necesites
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def handle_missing_values(df: pd.DataFrame, strategy: str = 'forward_fill') -> pd.DataFrame:
    """
    Maneja valores faltantes con diferentes estrategias
    
    PLACEHOLDER: Para cuando necesites manejar datos faltantes
    """
    df = df.copy()
    
    if strategy == 'forward_fill':
        df = df.fillna(method='ffill')
    elif strategy == 'interpolate':
        df = df.interpolate(method='linear')
    elif strategy == 'drop':
        df = df.dropna()
    
    return df

def validate_sorteo_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Valida que los datos del sorteo sean consistentes
    
    PLACEHOLDER: Para validación de datos
    """
    errors = []
    
    # Verificar columnas requeridas
    required_columns = ['ID_SORTEO', 'NOMBRE', 'FECHA_CIERRE', 'EMISION']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Columnas faltantes: {missing_cols}")
    
    # Verificar valores negativos donde no deberían
    if (df['EMISION'] < 0).any():
        errors.append("Emisión con valores negativos")
    
    is_valid = len(errors) == 0
    return is_valid, errors