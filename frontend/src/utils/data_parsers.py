"""Helpers to transform backend payloads into Streamlit-friendly tables."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def build_predictions_dataframe(predictions: Dict[str, Any]) -> pd.DataFrame:
    """Create a summary dataframe from the prediction payload."""

    rows: List[Dict[str, Any]] = []

    for key, payload in predictions.items():
        sorteo_info = payload.get("sorteo_info", {})
        summary = payload.get("prediction_summary", {})

        rows.append(
            {
                "ID": key,
                "Sorteo": sorteo_info.get("nombre"),
                "Tipo": sorteo_info.get("tipo"),
                "Fecha de cierre": sorteo_info.get("fecha_cierre"),
                "Emisión": sorteo_info.get("emision"),
                "Talones estimados": summary.get("total_estimado"),
                "% avance": summary.get("porcentaje_avance"),
                "Días restantes": summary.get("dias_restantes"),
                "Suavizado": "Sí" if payload.get("was_smoothed") else "No",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "ID",
                "Sorteo",
                "Tipo",
                "Fecha de cierre",
                "Emisión",
                "Talones estimados",
                "% avance",
                "Días restantes",
                "Suavizado",
            ]
        )

    df = pd.DataFrame(rows)
    df["Fecha de cierre"] = pd.to_datetime(df["Fecha de cierre"], errors="coerce")
    df = df.sort_values("Fecha de cierre", na_position="last").reset_index(drop=True)
    return df


def build_chart_dataframe(chart_data: Dict[str, Iterable[Any]]) -> pd.DataFrame:
    """Return a dataframe prepared for the line chart."""

    dates = list(chart_data.get("dates", []))
    values = list(chart_data.get("values", []))
    daily_values = list(chart_data.get("daily_values", []))

    df = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(dates, errors="coerce"),
            "Talones estimados": values,
            "Talones diarios": daily_values,
        }
    )

    df = df.dropna(subset=["Fecha"]).sort_values("Fecha")
    return df


def extract_training_list(debug_info: Optional[Dict[str, Any]], prediction_key: str) -> List[str]:
    """Get the list of training draws from the debug payload."""

    if not debug_info:
        return []

    predictions = debug_info.get("predictions") or debug_info.get("raw_predictions")
    if not predictions:
        return []

    target = predictions.get(prediction_key)
    if not target:
        return []

    sorteo_info = target.get("sorteo_info", {})
    training = sorteo_info.get("sorteos_entrenamiento")

    if isinstance(training, list):
        return training

    return []


def build_training_summary(
    debug_info: Optional[Dict[str, Any]],
    training_selection: Iterable[str],
) -> Optional[pd.DataFrame]:
    """Aggregate information for the selected training draws when available."""

    if not debug_info:
        return None

    sorteo_data = debug_info.get("sorteo_data")
    if not sorteo_data:
        return None

    df_escalado = _as_dataframe(sorteo_data.get("df_escalado"))
    if df_escalado is None or df_escalado.empty:
        return None

    selection = list(training_selection)
    if not selection:
        return None

    filtered = df_escalado[df_escalado["NOMBRE"].isin(selection)]
    if filtered.empty:
        return None

    aggregated = (
        filtered.groupby("NOMBRE")
        .agg(
            {
                "DNAS": "max",
                "PORCENTAJE_DE_AVANCE_SIN_MEMBRE": "max",
                "PORCENTAJE_DNAS": "max",
            }
        )
        .reset_index()
    )

    aggregated = aggregated.rename(
        columns={
            "NOMBRE": "Sorteo",
            "DNAS": "DNAS máximos",
            "PORCENTAJE_DE_AVANCE_SIN_MEMBRE": "% avance máximo",
            "PORCENTAJE_DNAS": "% DNAS cubierto",
        }
    )

    aggregated["% avance máximo"] = (aggregated["% avance máximo"].astype(float).round(2))
    aggregated["% DNAS cubierto"] = (aggregated["% DNAS cubierto"].astype(float).round(2))

    return aggregated


def _as_dataframe(payload: Any) -> Optional[pd.DataFrame]:
    """Best-effort conversion from JSON-serializable payloads to DataFrame."""

    if payload is None:
        return None

    if isinstance(payload, pd.DataFrame):
        return payload

    try:
        if isinstance(payload, dict):
            if {"data", "columns"}.issubset(payload.keys()):
                return pd.DataFrame(payload["data"], columns=payload["columns"])
            return pd.DataFrame(payload)

        if isinstance(payload, list):
            return pd.DataFrame(payload)

    except Exception:  # pragma: no cover - defensive
        return None

    return None