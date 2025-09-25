"""
LEARNING NOTE: CRUD endpoints para productos
"""

from fastapi import APIRouter, HTTPException
from typing import List
from backend.infrastructure.database.bigquery import BigQueryRepository
from backend.domain.models.product import Product
from backend.core.constants import ProductID

router = APIRouter(
    prefix="/products",
    tags=["products"]
)

# Instancia del repositorio
bq_repo = BigQueryRepository()

@router.get("/", response_model=List[Product])
async def get_all_products():
    """
    Lista todos los productos disponibles
    
    LEARNING NOTE: Simple endpoint de consulta
    """
    
    try:
        # Obtener IDs de BigQuery
        product_ids = bq_repo.get_all_products()
        
        # Crear objetos Product
        # PREGUNTA: ¿De dónde sacamos los nombres y categorías?
        # Por ahora, uso datos dummy
        
        products = []
        for pid in product_ids:
            products.append(Product(
                id=pid,
                name=f"Producto {pid.split('_')[-1]}",
                category="General",  # TODO: Obtener categoría real
                is_active=True
            ))
        
        return products
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{product_id}", response_model=Product)
async def get_product(product_id: str):
    """Obtiene información de un producto específico"""
    
    # Validar que existe
    if product_id not in [p.value for p in ProductID]:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
    
    return Product(
        id=product_id,
        name=f"Producto {product_id.split('_')[-1]}",
        category="General",
        is_active=True
    )

@router.post("/{product_id}/configure")
async def configure_product_smoothing(
    product_id: str,
    max_change: float = 2.0,
    alpha: float = 0.25
):
    """
    Configura parámetros de suavizado por producto
    
    PREGUNTA: ¿Quieres configuración individual por producto?
    """
    
    # TODO: Guardar configuración en algún lado
    # Por ahora, solo validamos
    
    if max_change < 0 or max_change > 10:
        raise HTTPException(status_code=400, detail="max_change must be between 0 and 10")
    
    if alpha < 0 or alpha > 1:
        raise HTTPException(status_code=400, detail="alpha must be between 0 and 1")
    
    return {
        "product_id": product_id,
        "config": {
            "max_change_percent": max_change,
            "smoothing_alpha": alpha
        },
        "message": "Configuration saved (TODO: implement persistence)"
    }