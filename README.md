# Predicción de Niveles de Obesidad

## Descripción del Proyecto
Este proyecto utiliza el dataset **Estimation of Obesity Levels Based on Eating Habits and Physical Condition** del UCI Machine Learning Repository para predecir niveles de obesidad (7 clases) basados en hábitos alimenticios, actividad física y características demográficas. Es relevante para biotech, salud pública y nutrición, con aplicaciones en programas de prevención y personalización dietética.

- **Dataset:** Obesity Levels (2,111 instancias, 17 features).
- **Fuente:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition).
- **Herramientas:** Python con Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn y SciPy.
- **Objetivos:**
  - Realizar análisis exploratorio de datos (EDA) para identificar patrones.
  - Limpiar y preparar datos (sin valores faltantes, codificación categórica).
  - Pruebas de hipótesis (e.g., asociación entre consumo de comida calórica y obesidad).
  - Modelado de clasificación multi-clase con Random Forest.
  - Evaluación con accuracy (~94%), F1-score y matriz de confusión.
  - Visualizaciones: histogramas, boxplots, mapa de calor, importancia de variables.

## Requisitos
- Python 3.8+.
- Bibliotecas: Instala con `pip install pandas numpy matplotlib seaborn scikit-learn scipy`.
- Dataset: Descarga de [aquí](https://archive.ics.uci.edu/static/public/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition.zip) y coloca `ObesityDataSet_raw_and_data_sinthetic.csv` en la carpeta del notebook.

## Metodología
1. **Carga y Limpieza:** Dataset cargado con 2,111 instancias, 0 valores faltantes.
2. **EDA:**
   - Distribución del target: 7 clases casi balanceadas (~13-16% cada una).
   - Histogramas de variables numéricas (e.g., Weight sesgada hacia valores altos).
   - Boxplots: Peso varía significativamente por nivel de obesidad.
   - Correlaciones: Bajas (<0.5) entre variables numéricas.
3. **Pruebas de Hipótesis:** Chi-cuadrado muestra asociación significativa (p-value: 1.48e-47) entre consumo de comida calórica (FAVC) y NObeyesdad.
4. **Preparación:** Codificación de categóricas, split 80/20 con estratificación (X_train: 1,688; X_test: 423), escalado de variables numéricas.
5. **Modelado:** Random Forest (100 árboles, profundidad 10). Accuracy: 94.33%.
6. **Evaluación:** F1-scores altos (>0.87), con Weight como predictor principal (importancia: 0.365).
7. **Visualizaciones:** Matriz de confusión, gráfico de importancia de features.

## Resultados Clave
- **Accuracy:** 94.33%.
- **Mejor rendimiento:** Clases como Obesity_Type_II y III (f1-score ~0.99).
- **Insights:** Peso y frecuencia de consumo vegetal (FCVC) son predictores clave, útil para estrategias de nutrición.
- **Limitaciones:** Correlaciones bajas entre variables numéricas limitan la interacción multivariada.

## Cómo Ejecutar
1. Descarga el dataset y coloca `ObesityDataSet_raw_and_data_sinthetic.csv` en la carpeta.
2. Abre `Proyecto9_NutricionObesidad.ipynb` en Jupyter Notebook.
3. Ejecuta las celdas en orden.
4. Nota: El entrenamiento toma ~1-2 minutos.

## Mejoras Futuras
- Probar balanceo de clases o XGBoost para optimizar F1-scores.
- Incluir análisis de interacción entre variables categóricas.
- Desarrollar una interfaz web para predicciones en tiempo real.

## Licencia
MIT License. Cita el dataset original si usas este proyecto.

Autor: [Adrián Galván]  
Fecha: Septiembre 2025  
