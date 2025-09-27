"""
LEARNING NOTE: Workflow actualizado para sorteos
"""

from langgraph.graph import StateGraph, END
from typing import Dict, Any, List
import logging
from datetime import datetime
from backend.application.workflows.states import PredictionState
from backend.application.workflows.nodes import (
    fetch_data_node,
    predict_node,
    smooth_node,
    explain_node,
    compile_results_node
)

logger = logging.getLogger(__name__)

def should_explain(state: Dict[str, Any]) -> str:
    """
    Decide si necesitamos explicación del LLM
    """
    if state.get("needs_llm_explanation", False):
        return "explain"
    else:
        return "compile"

def build_prediction_workflow():
    """
    Construye el workflow para predicción de sorteos
    """
    
    # Crear el grafo
    workflow = StateGraph(PredictionState)
    
    # Agregar nodos
    workflow.add_node("fetch_data", fetch_data_node)
    workflow.add_node("predict", predict_node) 
    workflow.add_node("smooth", smooth_node)
    workflow.add_node("explain", explain_node)
    workflow.add_node("compile", compile_results_node)
    
    # Definir flujo
    workflow.set_entry_point("fetch_data")
    
    # Edges lineales
    workflow.add_edge("fetch_data", "predict")
    workflow.add_edge("predict", "smooth")
    
    # Edge condicional
    workflow.add_conditional_edges(
        "smooth",
        should_explain,
        {
            "explain": "explain",
            "compile": "compile"
        }
    )
    
    # Edges finales
    workflow.add_edge("explain", "compile")
    workflow.add_edge("compile", END)
    
    # Compilar el grafo
    app = workflow.compile()
    
    logger.info("Prediction workflow compiled successfully")
    
    return app

# Crear instancia global
prediction_workflow = build_prediction_workflow()   

async def run_prediction_workflow(
    sorteo_types: List[str],
    apply_smoothing: bool = True,
    force_recalculation: bool = False
) -> Dict[str, Any]:
    """
    Ejecuta el workflow de predicción para sorteos
    
    LEARNING NOTE: Actualizado para manejar tipos de sorteo
    """
    
    initial_state = {
        "sorteo_types": sorteo_types,  # Cambiado de product_ids
        "apply_smoothing": apply_smoothing,
        "force_recalculation": force_recalculation,
        "start_time": datetime.now(),
        "predictions": {},
        "errors": [],
        "processing_stage": "starting"
    }
    
    try:
        # Ejecutar el workflow
        result = await prediction_workflow.ainvoke(initial_state)
        
        # Log resumen
        logger.info(
            f"Workflow completado: {len(result.get('final_predictions', {}))} "
            f"sorteos procesados en {result.get('processing_time', 0):.2f}s"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {str(e)}")
        return {
            "error": str(e),
            "processing_stage": "failed",
            "errors": initial_state.get("errors", []) + [str(e)]
        }