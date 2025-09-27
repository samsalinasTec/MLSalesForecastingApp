"""
LEARNING NOTE: CRUD endpoints para productos
"""

from datetime import date, datetime
from fastapi import APIRouter, HTTPException
from typing import List, Optional

from backend.infrastructure.database.bigquery import BigQueryRepository
from backend.domain.models.product import Product
from backend.core.constants import SorteoType

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

        # Crear objetos Product (datos dummy por ahora)
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
    """
    Obtiene información de un producto específico.
    Espera formato <TIPO>_<NUMERO_EDICION>, por ejemplo: TST_210
    """
    # Validar formato <TIPO>_<EDICION>
    try:
        sorteo_raw, edicion_raw = product_id.split("_", 1)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Formato inválido. Usa <TIPO>_<NUMERO_EDICION> (ej. TST_210).",
        )

    # Validar tipo de sorteo contra Enum
    try:
        sorteo_tipo = SorteoType(sorteo_raw.upper())
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Tipo de sorteo '{sorteo_raw}' no es válido.",
        )

    # Validar edición numérica
    edicion = edicion_raw.strip()
    if not edicion.isdigit():
        raise HTTPException(
            status_code=400,
            detail="NUMERO_EDICION debe ser numérico.",
        )

    # Consultar información de sorteos
    try:
        df_info = bq_repo.get_sorteos_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error consultando sorteos: {e}")

    # Filtrar por tipo y edición
    matching_rows = df_info[
        (df_info["SORTEO_SIGLAS"].astype(str).str.upper() == sorteo_tipo.value)
        & (df_info["NUMERO_EDICION"].astype(str) == edicion)
    ]

    if matching_rows.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No se encontró sorteo para tipo {sorteo_tipo.value} "
                f"con edición {edicion}."
            ),
        )

    sorteo = matching_rows.iloc[0]

    nombre = sorteo.get("NOMBRE") or f"Sorteo {sorteo_tipo.value} {edicion}"
    categoria = sorteo.get("SORTEO_GRUPO") or sorteo_tipo.value

    # Obtener fecha de cierre y determinar si está activo
    fecha_cierre = sorteo.get("FECHA_CIERRE")
    fecha_cierre_date: Optional[date] = None
    if isinstance(fecha_cierre, datetime):
        fecha_cierre_date = fecha_cierre.date()
    elif isinstance(fecha_cierre, date):
        fecha_cierre_date = fecha_cierre
    else:
        # pandas.Timestamp u otros tipos con .date()
        try:
            fecha_cierre_date = fecha_cierre.date()
        except Exception:
            fecha_cierre_date = None

    is_active = False
    if fecha_cierre_date:
        is_active = fecha_cierre_date >= datetime.utcnow().date()

    return Product(
        id=product_id,
        name=nombre,
        category=categoria,
        is_active=is_active,
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
    # Validaciones
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
