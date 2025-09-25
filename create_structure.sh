#!/bin/bash

# Script para crear la estructura completa del proyecto Sales Forecaster
# Uso: bash create_structure.sh

echo "🚀 Creando estructura del proyecto Sales Forecaster..."

# Crear directorio principal
mkdir -p sales-forecaster
cd sales-forecaster

# Backend structure
echo "📁 Creando estructura del backend..."

# API layer
mkdir -p backend/api/v1/endpoints
mkdir -p backend/api/v1/dependencies
mkdir -p backend/api/v2  # Para futuro

# Core layer
mkdir -p backend/core

# Domain layer
mkdir -p backend/domain/models
mkdir -p backend/domain/schemas/request
mkdir -p backend/domain/schemas/response

# Infrastructure layer
mkdir -p backend/infrastructure/database
mkdir -p backend/infrastructure/ml/models
mkdir -p backend/infrastructure/external

# Application layer
mkdir -p backend/application/services
mkdir -p backend/application/workflows

# Tests (vacío por ahora)
mkdir -p backend/tests/unit
mkdir -p backend/tests/integration
mkdir -p backend/tests/e2e

# Migrations (vacío)
mkdir -p backend/migrations

# Frontend structure
echo "📁 Creando estructura del frontend..."

mkdir -p frontend/src/pages
mkdir -p frontend/src/components/charts
mkdir -p frontend/src/components/filters
mkdir -p frontend/src/components/metrics
mkdir -p frontend/src/utils
mkdir -p frontend/src/config

# Deployment
echo "📁 Creando estructura de deployment..."

mkdir -p deployment/docker
mkdir -p deployment/kubernetes
mkdir -p deployment/terraform

# Scripts
mkdir -p scripts

# Docs
mkdir -p docs

# GitHub workflows
mkdir -p .github/workflows

# Data folder for SQL queries
mkdir -p data/sql

# Crear archivos vacíos __init__.py para Python
echo "📝 Creando archivos __init__.py..."

# Función para crear __init__.py en todos los directorios Python
find backend -type d -exec touch {}/__init__.py \;
find frontend/src -type d -exec touch {}/__init__.py \;

# Crear archivos base vacíos
echo "📝 Creando archivos base..."

# Root files
touch .env.example
touch .gitignore
touch requirements.txt
touch README.md
touch Makefile

# Backend main files
touch backend/main.py

# Core files
touch backend/core/config.py
touch backend/core/constants.py
touch backend/core/exceptions.py
touch backend/core/security.py

# Domain models
touch backend/domain/models/prediction.py
touch backend/domain/models/product.py

# Domain schemas
touch backend/domain/schemas/request/prediction.py
touch backend/domain/schemas/response/prediction.py

# Infrastructure files
touch backend/infrastructure/database/bigquery.py
touch backend/infrastructure/database/redis.py
touch backend/infrastructure/ml/models/linear_regression.py
touch backend/infrastructure/ml/preprocessing.py
touch backend/infrastructure/external/vertex_ai.py

# Application services
touch backend/application/services/prediction_service.py
touch backend/application/services/smoothing_service.py
touch backend/application/services/analytics_service.py

# Workflows
touch backend/application/workflows/states.py
touch backend/application/workflows/nodes.py
touch backend/application/workflows/graphs.py

# API endpoints
touch backend/api/v1/endpoints/predictions.py
touch backend/api/v1/endpoints/products.py
touch backend/api/v1/endpoints/analytics.py
touch backend/api/v1/endpoints/health.py

# API dependencies
touch backend/api/v1/dependencies/auth.py
touch backend/api/v1/dependencies/database.py

# Frontend files
touch frontend/app.py
touch frontend/src/pages/dashboard.py
touch frontend/src/pages/predictions.py
touch frontend/src/pages/settings.py

# Components
touch frontend/src/components/charts/sales_curve.py
touch frontend/src/components/charts/comparison.py
touch frontend/src/components/filters/product_selector.py
touch frontend/src/components/metrics/kpi_cards.py

# Utils
touch frontend/src/utils/formatters.py
touch frontend/src/config/constants.py

# Deployment files
touch deployment/docker/Dockerfile
touch deployment/docker/docker-compose.yml

# Scripts
touch scripts/setup.sh
touch scripts/run_local.sh
chmod +x scripts/*.sh

# Docs
touch docs/API.md
touch docs/ARCHITECTURE.md
touch docs/DEPLOYMENT.md

# GitHub Actions
touch .github/workflows/ci.yml
touch .github/workflows/cd.yml

# Data
touch data/sql/queries.sql

# Crear .gitignore básico
cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Data
*.csv
*.xlsx
data/raw/
data/processed/

# ML Models
*.pkl
*.h5
models/

# Pytest
.pytest_cache/
.coverage
htmlcov/

# Jupyter
.ipynb_checkpoints/
*.ipynb
EOL

echo "✅ Estructura del proyecto creada exitosamente!"
echo ""
echo "📊 Resumen de la estructura:"
echo "  - Backend: API con FastAPI, servicios, ML, y workflows"
echo "  - Frontend: Preparado para Streamlit"
echo "  - Tests: Carpetas para unit, integration, y e2e tests"
echo "  - Deployment: Docker, Kubernetes, Terraform"
echo "  - Docs: Documentación del proyecto"
echo ""
echo "🎯 Próximos pasos:"
echo "  1. cd sales-forecaster"
echo "  2. Copiar el código del backend a los archivos correspondientes"
echo "  3. Configurar .env con tus credenciales"
echo "  4. pip install -r requirements.txt"
echo "  5. python -m backend.main"

# Mostrar la estructura creada
echo ""
echo "📁 Estructura creada:"
tree -L 3 -d 2>/dev/null || find . -type d -maxdepth 3 | sed 's|^\./||' | sort