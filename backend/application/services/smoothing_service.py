"""
LEARNING NOTE: Servicio dedicado al suavizado
Separado para poder cambiar la estrategia fácilmente
"""

from typing import Dict, List, Tuple, Optional
from collections import deque
import logging
from backend.domain.models.prediction import Prediction
from backend.core.config import settings
from backend.core.exceptions import SmoothingException

logger = logging.getLogger(__name__)

class SmoothingService:
    """
    Maneja el suavizado de predicciones
    
    LEARNING NOTE: Estado en memoria por simplicidad
    En producción, usarías Redis o una BD
    """
    
    def __init__(self):
        # Historia por producto
        self.history: Dict[str, deque] = {}
        self.max_change = settings.max_daily_change_percent / 100
        self.alpha = settings.smoothing_alpha
        
    def apply_smoothing(
        self,
        prediction: Prediction
    ) -> Tuple[Prediction, bool, str]:
        """
        Aplica suavizado si es necesario
        
        Returns:
            - Prediction actualizada
            - Si se aplicó suavizado
            - Mensaje explicativo
        """
        
        product_id = prediction.product_id
        original_value = prediction.original_value
        
        # Inicializar historia si no existe
        if product_id not in self.history:
            self.history[product_id] = deque(maxlen=settings.history_days)
            
        history = self.history[product_id]
        
        # Si no hay historia, no suavizar
        if not history:
            self.history[product_id].append(original_value)
            return prediction, False, "Primera predicción - sin suavizado"
        
        # Calcular cambio vs último valor
        last_value = history[-1]
        change_rate = (original_value - last_value) / last_value
        
        # Verificar si necesita suavizado
        if abs(change_rate) <= self.max_change:
            # No necesita suavizado
            self.history[product_id].append(original_value)
            return prediction, False, f"Cambio de {change_rate*100:.1f}% dentro del límite"
        
        # APLICAR SUAVIZADO
        # LEARNING NOTE: Exponential smoothing simple
        # Puedes cambiar a otros métodos más sofisticados
        
        # Opción 1: Suavizado exponencial
        smoothed_value = self.alpha * original_value + (1 - self.alpha) * last_value
        
        # Opción 2: Limitar el cambio máximo
        # max_change_value = last_value * (1 + self.max_change if change_rate > 0 else 1 - self.max_change)
        # smoothed_value = max_change_value
        
        # Opción 3: Media ponderada con historia
        # weights = [0.5, 0.3, 0.2]  # Pesos para últimos 3 valores
        # weighted_history = sum(h * w for h, w in zip(history, weights))
        # smoothed_value = 0.3 * original_value + 0.7 * weighted_history
        
        # Actualizar predicción
        prediction.smoothed_value = smoothed_value
        prediction.final_value = smoothed_value
        prediction.was_smoothed = True
        
        # Guardar en historia
        self.history[product_id].append(smoothed_value)
        
        message = (
            f"Suavizado aplicado: {change_rate*100:.1f}% de cambio "
            f"(${original_value:,.0f} → ${smoothed_value:,.0f})"
        )
        
        logger.info(f"[{product_id}] {message}")
        
        return prediction, True, message
    
    def get_smoothing_stats(self, product_id: str) -> Dict:
        """Retorna estadísticas del suavizado"""
        
        if product_id not in self.history:
            return {"message": "Sin historia de suavizado"}
        
        history_list = list(self.history[product_id])
        
        if len(history_list) < 2:
            return {"message": "Historia insuficiente"}
        
        # Calcular estadísticas
        changes = []
        for i in range(1, len(history_list)):
            change = (history_list[i] - history_list[i-1]) / history_list[i-1]
            changes.append(change * 100)
        
        return {
            "total_predictions": len(history_list),
            "average_change": sum(changes) / len(changes),
            "max_change": max(changes),
            "min_change": min(changes),
            "last_values": history_list[-5:]  # Últimos 5 valores
        }
    
    def reset_history(self, product_id: Optional[str] = None):
        """Reinicia la historia de suavizado"""
        
        if product_id:
            if product_id in self.history:
                self.history[product_id].clear()
                logger.info(f"Historia reiniciada para {product_id}")
        else:
            self.history.clear()
            logger.info("Toda la historia de suavizado reiniciada")