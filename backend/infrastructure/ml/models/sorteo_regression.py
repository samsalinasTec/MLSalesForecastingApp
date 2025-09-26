"""
LEARNING NOTE: Modelo de regresión específico para sorteos
Refactorizado de tu clase SorteosTecLRWM
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SorteoPolynomialRegressor:
    """
    Modelo de regresión polinomial para predicción de ventas de sorteos
    
    LEARNING NOTE: Esta es tu lógica de regresión extraída y mejorada
    """
    
    def __init__(self, max_degree: int = 50, max_iterations: int = 50):
        """
        Args:
            max_degree: Grado máximo del polinomio a probar
            max_iterations: Máximas iteraciones para encontrar mejor modelo
        """
        self.max_degree = max_degree
        self.max_iterations = max_iterations
        self.best_model = None
        self.best_degree = None
        self.best_random_state = None
        self.best_r2 = 0
        
    def find_best_polynomial_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        test_sizes: List[float] = None
    ) -> Tuple[Pipeline, int, int, float]:
        """
        Encuentra el mejor modelo polinomial probando diferentes configuraciones
        
        LEARNING NOTE: Esta es tu lógica de búsqueda del mejor modelo
        Prueba diferentes grados y random_states hasta encontrar R² > 0.99
        """
        
        if test_sizes is None:
            test_sizes = [0.2, 0.19, 0.18, 0.17, 0.14, 0.21, 0.22]
        
        resultados = []
        test_size_index = 0
        
        # Buscar hasta encontrar R² > 0.99 o agotar opciones
        while test_size_index < len(test_sizes) and self.best_r2 < 0.99:
            test_size = test_sizes[test_size_index]
            test_size_index += 1
            
            for random_state in range(1, self.max_iterations):
                for degree in range(1, self.max_degree):
                    try:
                        # Split datos
                        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                            X_train, y_train, test_size=test_size, random_state=random_state
                        )
                        
                        # Crear y entrenar pipeline
                        pipeline = Pipeline([
                            ('poly', PolynomialFeatures(degree=degree)),
                            ('linear', LinearRegression())
                        ])
                        
                        pipeline.fit(X_train_split, y_train_split)
                        
                        # Evaluar
                        y_pred_test = pipeline.predict(X_test_split)
                        r2 = r2_score(y_test_split, y_pred_test)
                        
                        resultados.append((random_state, degree, r2))
                        
                        # Actualizar mejor modelo
                        if r2 > self.best_r2:
                            self.best_r2 = r2
                            self.best_degree = degree
                            self.best_random_state = random_state
                            self.best_model = pipeline
                            
                    except Exception as e:
                        logger.warning(f"Error con degree={degree}, rs={random_state}: {e}")
                        continue
        
        # Si no se encontró buen modelo, usar el mejor encontrado
        if self.best_model is None and resultados:
            resultados.sort(key=lambda x: x[2], reverse=True)
            best_result = resultados[0]
            
            # Reentrenar con mejores parámetros
            pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=best_result[1])),
                ('linear', LinearRegression())
            ])
            pipeline.fit(X_train, y_train)
            
            self.best_model = pipeline
            self.best_degree = best_result[1]
            self.best_random_state = best_result[0]
            self.best_r2 = best_result[2]
        
        logger.info(
            f"Mejor modelo: degree={self.best_degree}, "
            f"random_state={self.best_random_state}, R²={self.best_r2:.4f}"
        )
        
        return self.best_model, self.best_degree, self.best_random_state, self.best_r2
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, any]:
        """
        Entrena el modelo con los datos completos
        """
        # Buscar mejor configuración
        self.find_best_polynomial_model(X, y)
        
        # Reentrenar con todos los datos
        if self.best_model:
            self.best_model.fit(X, y)
        
        return {
            "degree": self.best_degree,
            "r2_score": self.best_r2,
            "random_state": self.best_random_state
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado
        """
        if self.best_model is None:
            raise ValueError("Modelo no entrenado. Llama a train() primero")
        
        return self.best_model.predict(X)
    
    def predict_sales_curve(
        self,
        max_dnas: int,
        emision: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predice la curva completa de ventas hasta el cierre
        
        Args:
            max_dnas: Días hasta el cierre del sorteo
            emision: Total de boletos emitidos
        
        Returns:
            X_days: Array de porcentaje de DNAS
            y_sales: Array de ventas estimadas
        """
        # Crear X para predicción (0 a 1 representando el tiempo)
        X_to_predict = np.linspace(0, 1, max_dnas)
        
        # Predecir porcentajes
        y_percentages = self.predict(X_to_predict.reshape(-1, 1))
        
        # Convertir a ventas absolutas
        y_sales = y_percentages * emision
        
        return X_to_predict, y_sales