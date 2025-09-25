"""
LEARNING NOTE: Entry point de la aplicación
Aquí se configura todo FastAPI
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from contextlib import asynccontextmanager
from backend.core.config import settings
from backend.api.v1.endpoints import predictions, products, analytics, health

# Configurar logging
logging.basicConfig(
    level=logging.INFO if not settings.debug_mode else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    LEARNING NOTE: Lifespan context manager
    Código que ejecuta al iniciar/apagar la app
    """
    # Startup
    logger.info("Starting Sales Forecaster API")
    logger.info(f"Debug mode: {settings.debug_mode}")
    logger.info(f"GCP Project: {settings.gcp_project_id}")
    
    # Aquí podrías:
    # - Precargar modelos ML
    # - Establecer conexiones a BD
    # - Inicializar caché
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sales Forecaster API")
    # Aquí podrías:
    # - Cerrar conexiones
    # - Guardar estado
    # - Limpiar recursos

# Crear aplicación
app = FastAPI(
    title="Sales Forecaster API",
    description="ML-powered sales prediction with intelligent smoothing",
    version=settings.api_version,
    lifespan=lifespan
)

# LEARNING NOTE: CORS para permitir requests desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LEARNING NOTE: Middleware custom para logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log todas las requests"""
    
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Procesar request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - Time: {process_time:.3f}s")
    
    # Agregar header con tiempo de procesamiento
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# LEARNING NOTE: Exception handler global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Maneja excepciones no capturadas"""
    
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if settings.debug_mode else "An error occurred"
        }
    )

# Incluir routers
app.include_router(
    predictions.router,
    prefix=f"/api/{settings.api_version}"
)

app.include_router(
    products.router,
    prefix=f"/api/{settings.api_version}"
)

# TODO: Implementar estos endpoints
# app.include_router(analytics.router, prefix=f"/api/{settings.api_version}")
# app.include_router(health.router, prefix=f"/api/{settings.api_version}")

# Root endpoint
@app.get("/")
async def root():
    """Health check básico"""
    return {
        "service": "Sales Forecaster API",
        "version": settings.api_version,
        "status": "operational"
    }

@app.get("/api/v1/health")
async def health_check():
    """
    Health check detallado
    
    LEARNING NOTE: Importante para Kubernetes/Cloud Run
    """
    
    health_status = {
        "status": "healthy",
        "version": settings.api_version,
        "checks": {
            "api": "operational",
            "bigquery": "not_checked",  # TODO: Verificar conexión
            "vertex_ai": "not_checked",  # TODO: Verificar API
        }
    }
    
    return health_status

if __name__ == "__main__":
    import uvicorn
    
    # LEARNING NOTE: Solo para desarrollo local
    # En producción usa: uvicorn backend.main:app
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug_mode,
        log_level="info" if not settings.debug_mode else "debug"
    )