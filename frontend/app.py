"""Streamlit dashboard tailored to the MLSalesForecasting backend."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import httpx
import plotly.graph_objects as go
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from src.utils.api_client import APIClient  # noqa: E402
from src.utils.data_parsers import (  # noqa: E402
    build_chart_dataframe,
    build_predictions_dataframe,
    build_training_summary,
    extract_training_list,
)

DEFAULT_API_URL = os.getenv("PREDICTIONS_API_URL", "http://localhost:8080/api/v1")


@st.cache_data(ttl=600, show_spinner=False)
def cached_active_sorteos(base_url: str) -> Dict:
    """Cache layer around the ``/predictions/active`` endpoint."""
    # Aumentamos el timeout para soportar respuestas lentas del backend (e.g., BigQuery frío)
    client = APIClient(base_url, timeout=300.0)
    return client.get_active_sorteos()


def initialize_session_state() -> None:
    """Populate default keys to avoid ``KeyError`` during reruns."""
    st.session_state.setdefault("api_base_url", DEFAULT_API_URL)
    st.session_state.setdefault("predictions_payload", None)
    st.session_state.setdefault("predictions", {})
    st.session_state.setdefault("debug_info", None)
    st.session_state.setdefault("summary_csv", None)
    st.session_state.setdefault("last_prediction_at", None)
    st.session_state.setdefault("apply_smoothing", True)
    st.session_state.setdefault("force_recalculation", False)
    st.session_state.setdefault("selected_sorteo_types", [])


def main() -> None:
    st.set_page_config(page_title="Predicción de sorteos", layout="wide")
    initialize_session_state()

    st.title("Panel de predicciones de sorteos")
    st.caption("Interactúa con el backend existente para generar, visualizar y descargar predicciones.")

    with st.sidebar:
        st.header("Configuración de la API")
        api_base = st.text_input(
            "URL base",
            st.session_state["api_base_url"],
            help="Incluye el prefijo /api/v1 de tu backend.",
        )
        if api_base != st.session_state["api_base_url"]:
            st.session_state["api_base_url"] = api_base
            cached_active_sorteos.clear()
            st.session_state["predictions_payload"] = None
            st.session_state["predictions"] = {}
            st.session_state["debug_info"] = None
            st.session_state["summary_csv"] = None

        refresh_active = st.button("Recargar sorteos activos", use_container_width=True, key="refresh_active")

        st.divider()
        st.header("Parámetros de ejecución")
        try:
            if refresh_active:
                cached_active_sorteos.clear()

            active_payload = (
                cached_active_sorteos(st.session_state["api_base_url"]) if st.session_state["api_base_url"] else {}
            )
            type_options = sorted((active_payload.get("sorteos_por_tipo") or {}).keys())

            if not type_options:
                st.warning("No se encontraron sorteos activos en la API.")
            default_selection = (
                type_options if not st.session_state["selected_sorteo_types"] else st.session_state["selected_sorteo_types"]
            )
            selected_types = st.multiselect(
                "Tipos de sorteo",
                options=type_options,
                default=default_selection,
                help="Selecciona los grupos que deseas procesar.",
                key="selected_sorteo_types",
            )
        except httpx.TimeoutException:
            st.error(
                "Se agotó el tiempo de espera al consultar los sorteos activos. "
                "Intenta de nuevo o incrementa el timeout del cliente."
            )
            active_payload = {}
            selected_types = []
        except httpx.HTTPStatusError as exc:
            st.error(f"HTTP {exc.response.status_code}: {exc.response.text}")
            active_payload = {}
            selected_types = []
        except httpx.RequestError as exc:
            st.error(f"Error de red al llamar al backend: {exc}")
            active_payload = {}
            selected_types = []
        except Exception as exc:  # pragma: no cover - defensivo
            st.error(f"Error inesperado al cargar sorteos activos: {exc}")
            active_payload = {}
            selected_types = []

        st.session_state["apply_smoothing"] = st.toggle(
            "Aplicar suavizado",
            value=st.session_state["apply_smoothing"],
            help="Activa o desactiva la lógica de smoothing del backend.",
            key="apply_smoothing_toggle",
        )
        st.session_state["force_recalculation"] = st.toggle(
            "Forzar recálculo",
            value=st.session_state["force_recalculation"],
            help="Ignora cachés previas y vuelve a correr el flujo completo.",
            key="force_recalculation_toggle",
        )

        trigger_prediction = st.button(
            "Generar predicciones",
            type="primary",
            use_container_width=True,
        )

        if trigger_prediction:
            if not selected_types:
                st.warning("Selecciona al menos un tipo de sorteo antes de continuar.")
            else:
                # Igual que arriba: elevamos timeout para operaciones potencialmente pesadas
                client = APIClient(st.session_state["api_base_url"], timeout=300.0)
                with st.spinner("Ejecutando workflow de predicción..."):
                    try:
                        response = client.create_predictions(
                            sorteo_types=selected_types,
                            apply_smoothing=st.session_state["apply_smoothing"],
                            force_recalculation=st.session_state["force_recalculation"],
                            save_to_bq=False,
                        )
                        st.session_state["predictions_payload"] = response
                        st.session_state["predictions"] = response.get("predictions", {})
                        st.session_state["debug_info"] = response.get("debug_info")
                        st.session_state["last_prediction_at"] = response.get("timestamp") or datetime.utcnow().isoformat()
                        st.session_state["summary_csv"] = None
                        st.success("Predicciones generadas correctamente.")
                    except httpx.TimeoutException:
                        st.error("Timeout esperando respuesta del backend al generar predicciones.")
                    except httpx.HTTPStatusError as exc:
                        st.error(f"HTTP {exc.response.status_code}: {exc.response.text}")
                    except httpx.RequestError as exc:
                        st.error(f"Error de red al llamar al backend: {exc}")
                    except Exception as exc:  # pragma: no cover - defensivo
                        st.error(f"Ocurrió un error inesperado al solicitar las predicciones: {exc}")

        if st.session_state.get("predictions"):
            st.divider()
            st.header("Resumen descargable")
            if st.button("Actualizar CSV de resumen", use_container_width=True, key="refresh_summary"):
                client = APIClient(st.session_state["api_base_url"], timeout=300.0)
                with st.spinner("Solicitando resumen al backend..."):
                    try:
                        csv_bytes, filename = client.download_prediction_summary()
                        st.session_state["summary_csv"] = (csv_bytes, filename)
                        st.success("Resumen actualizado.")
                    except httpx.TimeoutException:
                        st.error("Timeout esperando la generación/descarga del resumen (CSV).")
                    except httpx.HTTPStatusError as exc:
                        st.error(f"No fue posible generar el resumen. HTTP {exc.response.status_code}: {exc.response.text}")
                    except httpx.RequestError as exc:
                        st.error(f"Error de red al llamar al backend para el resumen: {exc}")
                    except Exception as exc:  # pragma: no cover - defensivo
                        st.error(f"Error inesperado al descargar el resumen: {exc}")

            if st.session_state.get("summary_csv"):
                csv_bytes, filename = st.session_state["summary_csv"]
                st.download_button(
                    "Descargar resumen (CSV)",
                    data=csv_bytes,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.caption("Genera el CSV para habilitar la descarga.")

    predictions = st.session_state.get("predictions") or {}

    if not predictions:
        st.info("Genera una predicción desde la barra lateral para visualizar los resultados.")
        return

    summary_df = build_predictions_dataframe(predictions)
    st.subheader("Resumen general de sorteos procesados")

    if not summary_df.empty:
        available_types = sorted([t for t in summary_df["Tipo"].dropna().unique()])
        selected_table_types = st.multiselect(
            "Filtrar por tipo",
            options=available_types,
            default=available_types,
            help="Selecciona los tipos de sorteo que quieres mostrar en la tabla.",
        )

        filtered_df = summary_df[summary_df["Tipo"].isin(selected_table_types)] if selected_table_types else summary_df
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Talones estimados": st.column_config.NumberColumn(format="%d"),
                "% avance": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    else:
        st.warning("El backend no devolvió información resumida para los sorteos procesados.")

    st.subheader("Detalle por sorteo")

    sorteo_keys = list(predictions.keys())
    selected_key = st.selectbox(
        "Selecciona un sorteo para explorar",
        options=sorteo_keys,
        format_func=lambda key: predictions[key].get("sorteo_info", {}).get("nombre", key),
    )

    sorteo_payload = predictions.get(selected_key, {})
    sorteo_info = sorteo_payload.get("sorteo_info", {})
    summary = sorteo_payload.get("prediction_summary", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("Talones estimados", f"{summary.get('total_estimado', 0):,.0f}")
    col2.metric("% avance", f"{summary.get('porcentaje_avance', 0):.2f}%")
    col3.metric("Días restantes", summary.get("dias_restantes", 0))

    extra_cols = st.columns(2)
    extra_cols[0].metric("Emisión", f"{sorteo_info.get('emision', 0):,}")
    extra_cols[1].metric("Suavizado aplicado", "Sí" if sorteo_payload.get("was_smoothed") else "No")

    if sorteo_payload.get("smoothing_message"):
        st.info(sorteo_payload["smoothing_message"])

    if sorteo_payload.get("llm_explanation"):
        with st.expander("Explicación generada por Vertex/LLM"):
            st.write(sorteo_payload["llm_explanation"])

    chart_df = build_chart_dataframe(sorteo_payload.get("chart_data", {}))
    if chart_df.empty:
        st.warning("No hay datos suficientes para construir la curva de ventas.")
    else:
        chart_mode = st.radio(
            "Datos a visualizar",
            options=["Acumulado", "Acumulado + Diario"],
            index=0,
            horizontal=True,
            help="Alterna entre la curva acumulada y la combinación con ventas diarias.",
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart_df["Fecha"],
                y=chart_df["Talones estimados"],
                mode="lines+markers",
                name="Talones acumulados",
                line=dict(color="#1f77b4", width=2),
            )
        )

        if chart_mode == "Acumulado + Diario" and "Talones diarios" in chart_df:
            fig.add_trace(
                go.Bar(
                    x=chart_df["Fecha"],
                    y=chart_df["Talones diarios"],
                    name="Talones diarios",
                    marker_color="#ff7f0e",
                    opacity=0.4,
                )
            )

        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Fecha",
            yaxis_title="Talones",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

    training_list = extract_training_list(st.session_state.get("debug_info"), selected_key)
    with st.expander("Sorteos históricos utilizados para el entrenamiento"):
        if training_list:
            selection = st.multiselect(
                "Selecciona los sorteos históricos a mostrar",
                options=training_list,
                default=training_list,
                key=f"{selected_key}_training_selection",
            )

            training_summary = build_training_summary(st.session_state.get("debug_info"), selection)
            if training_summary is not None:
                st.dataframe(training_summary, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "Activa el modo DEBUG en el backend para exponer la información detallada de los sorteos históricos."
                )
        else:
            st.info(
                "Los sorteos utilizados para el entrenamiento estarán disponibles cuando el backend envíe la información en el payload de debug."
            )

    if st.session_state.get("last_prediction_at"):
        st.caption(f"Última ejecución: {st.session_state['last_prediction_at']}")


if __name__ == "__main__":
    main()
