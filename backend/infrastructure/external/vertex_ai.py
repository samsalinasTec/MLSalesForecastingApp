"""
LEARNING NOTE: Integración con Vertex AI para LLM
Separado para cambiar fácilmente de proveedor (OpenAI, Anthropic, etc.)
"""

from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, GenerationConfig
import vertexai
from typing import Dict, Any, List, Optional
import json
import logging
from backend.core.config import settings
from backend.core.exceptions import VertexAIException

logger = logging.getLogger(__name__)

class VertexAIClient:
    """
    Cliente para interactuar con Gemini en Vertex AI
    
    LEARNING NOTE: Facade Pattern - Simplifica una API compleja
    """
    
    def __init__(self):
        """
        Inicializa configuración pero NO la conexión
        LAZY LOADING: La conexión real se hace cuando se usa
        """
        self._initialized = False
        self._model: Optional[GenerativeModel] = None
        self._generation_config: Optional[GenerationConfig] = None
        logger.info("VertexAIClient creado (sin conectar aún)")
    
    def _ensure_initialized(self):
        """
        Lazy initialization - solo conecta cuando se necesita
        
        LEARNING NOTE: Este patrón evita conexiones costosas en el arranque
        """
        if not self._initialized:
            logger.info("🔄 Inicializando conexión con Vertex AI...")
            
            try:
                vertexai.init(
                    project=settings.gcp_project_id,
                    location=settings.vertex_ai_location
                )
                
                self._model = GenerativeModel(settings.model_name)
                
                # Configuración para respuestas consistentes
                self._generation_config = GenerationConfig(
                    temperature=0.3,  # Baja temperatura = respuestas más consistentes
                    max_output_tokens=500,
                    top_p=0.8,
                    top_k=40
                )
                
                self._initialized = True
                logger.info("✅ Vertex AI conectado exitosamente")
                
            except Exception as e:
                logger.error(f"❌ Error conectando con Vertex AI: {str(e)}")
                raise VertexAIException(f"Failed to initialize Vertex AI: {str(e)}")
    
    @property
    def model(self) -> GenerativeModel:
        """Acceso lazy al modelo"""
        self._ensure_initialized()
        return self._model
    
    @property
    def generation_config(self) -> GenerationConfig:
        """Acceso lazy a la configuración"""
        self._ensure_initialized()
        return self._generation_config
        
    async def analyze_prediction_change(
        self,
        product_id: str,
        original_value: float,
        smoothed_value: float,
        historical_values: List[float],
        change_percentage: float
    ) -> str:
        """
        Usa el LLM para explicar cambios en predicciones
        
        LEARNING NOTE: Prompt engineering estructurado
        """
        
        # Solo inicializa cuando realmente se use
        self._ensure_initialized()
        
        # LEARNING NOTE: System prompt claro y específico
        prompt = f"""
        Eres un analista de ventas experto. Analiza el siguiente cambio en la predicción:
        
        Producto: {product_id}
        Predicción original del modelo: ${original_value:,.0f}
        Predicción suavizada: ${smoothed_value:,.0f}
        Cambio porcentual: {change_percentage:.1f}%
        
        Contexto histórico (últimos 10 días): {historical_values}
        
        Proporciona una explicación breve y profesional (máximo 3 líneas) sobre:
        1. Por qué se aplicó el suavizado
        2. Si el ajuste es razonable dado el contexto
        3. Una recomendación de acción
        
        Formato: JSON con keys "explicacion", "es_razonable", "recomendacion"
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            # Intentar parsear como JSON
            try:
                result = json.loads(response.text)
                return result.get("explicacion", response.text)
            except json.JSONDecodeError:
                # Si no es JSON válido, retornar texto plano
                return response.text
                
        except Exception as e:
            logger.error(f"Error en Vertex AI: {str(e)}")
            # Fallback sin LLM
            return f"Cambio de {change_percentage:.1f}% detectado. Suavizado aplicado para mantener consistencia."
    
    async def get_market_insights(
        self,
        product_id: str,
        predictions: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Obtiene insights de mercado para las predicciones
        
        PREGUNTA: ¿Quieres que el LLM sugiera acciones específicas?
        Por ejemplo: ajustar inventario, promociones, etc.
        """
        
        # Solo inicializa cuando realmente se use
        self._ensure_initialized()
        
        prompt = f"""
        Analiza las siguientes predicciones de venta y proporciona insights:
        
        Predicciones por producto:
        {json.dumps(predictions, indent=2)}
        
        Proporciona:
        1. Tendencia general del mercado
        2. Productos con mejor desempeño esperado
        3. Alertas o riesgos identificados
        4. 2-3 acciones recomendadas
        
        Formato: JSON estructurado
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            return json.loads(response.text)
            
        except Exception as e:
            logger.error(f"Error obteniendo insights: {str(e)}")
            return {
                "tendencia": "No disponible",
                "acciones": ["Revisar predicciones manualmente"]
            }