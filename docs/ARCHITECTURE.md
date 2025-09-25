# Sales Forecaster - Architecture

## Overview
Sistema de predicción de ventas con ML y suavizado inteligente.

## Arquitectura de Capas

### 1. API Layer (`/api`)
- FastAPI endpoints
- Request/Response validation
- Middleware

### 2. Application Layer (`/application`)
- Business logic
- Services
- Workflows (LangGraph)

### 3. Domain Layer (`/domain`)
- Business models
- Schemas
- Constants

### 4. Infrastructure Layer (`/infrastructure`)
- External services (BigQuery, Vertex AI)
- ML models
- Data repositories

## Flujo de Datos

1. Request → API Endpoint
2. API → LangGraph Workflow
3. Workflow → Services
4. Services → Infrastructure
5. Infrastructure → External Systems
6. Response ← API

## Patrones Utilizados

- **Repository Pattern**: Acceso a datos
- **Service Layer**: Lógica de negocio
- **Dependency Injection**: Configuración
- **State Machine**: Workflows con LangGraph

## Tecnologías

- **FastAPI**: API REST
- **LangGraph**: Orquestación
- **BigQuery**: Data warehouse
- **Vertex AI**: LLM para insights
- **Scikit-learn**: Modelos ML