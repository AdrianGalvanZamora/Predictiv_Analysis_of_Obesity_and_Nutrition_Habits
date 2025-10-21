# 🥗 Predicción de Niveles de Obesidad  
[English version below ⬇️]  

**Sector:** Salud Pública, Biotecnología, Nutrición  
**Herramientas:** Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy)  

---

## 📋 Descripción General  
Este proyecto utiliza el dataset *Estimation of Obesity Levels Based on Eating Habits and Physical Condition* del **UCI Machine Learning Repository** para **predecir los niveles de obesidad (7 clases)** basados en hábitos alimenticios, actividad física y características demográficas.  

Su relevancia se centra en el diseño de **estrategias preventivas y de salud personalizada**, aplicables en programas de nutrición, bienestar y medicina predictiva.  

---

## 📊 Dataset  
- **Fuente:** [UCI Machine Learning Repository – Obesity Levels Dataset](https://archive.ics.uci.edu/ml/datasets/Estimation+of+Obesity+Levels+Based+on+Eating+Habits+and+Physical+Condition)  
- **Tamaño:** 2,111 instancias, 17 características  
- **Archivo:** `ObesityDataSet_raw_and_data_sinthetic.csv`  

---

## 🔍 Metodología  
1. **Carga y Limpieza de Datos**  
   - Dataset con 0 valores faltantes.  
   - Codificación de variables categóricas.  

2. **Análisis Exploratorio (EDA)**  
   - Distribución balanceada de las 7 clases (~13–16% cada una).  
   - Peso (*Weight*) sesgado hacia valores altos.  
   - Boxplots muestran diferencias marcadas por nivel de obesidad.  
   - Correlaciones numéricas bajas (< 0.5).  

3. **Pruebas de Hipótesis**  
   - *Chi-cuadrado* mostró una asociación significativa (p ≈ 1.48 × 10⁻⁴⁷) entre consumo de comida calórica (*FAVC*) y nivel de obesidad (*NObeyesdad*).  

4. **Preparación y Modelado**  
   - División 80/20 (train/test) con estratificación.  
   - Escalado de variables numéricas.  
   - Modelo: *Random Forest Classifier* (100 árboles, profundidad = 10).  

5. **Evaluación del Modelo**  
   - Accuracy: **94.33%**  
   - F1-scores > **0.87** en la mayoría de las clases.  
   - Variable más importante: **Weight (importancia = 0.365)**.  

6. **Visualizaciones Clave**  
   - Matriz de confusión.  
   - Gráfico de importancia de características.  

---

## 🌎 Principales Hallazgos  
- **El peso y la frecuencia de consumo de vegetales (FCVC)** son los predictores más relevantes.  
- El modelo presenta una **alta capacidad de clasificación multicategoría (94%)**.  
- Útil para diseñar programas de prevención y control de obesidad.  

---

## 🧠 Aplicaciones en el Mundo Real  
- Evaluación de riesgo nutricional en poblaciones jóvenes.  
- Asistente inteligente para nutricionistas y médicos.  
- Base para aplicaciones de salud digital y seguimiento alimenticio.  

---

## ⚙️ Requisitos de Ejecución  
- Python 3.8+  
- Librerías: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`  
- Archivo: `ObesityDataSet_raw_and_data_sinthetic.csv`  

Instalación rápida:
- pip install pandas numpy matplotlib seaborn scikit-learn scipy


---

## 🚀 Cómo Ejecutar  
1. Descarga el dataset y colócalo en la carpeta del notebook.  
2. Abre `Nutricion_Obesidad.ipynb` en Jupyter Notebook.  
3. Ejecuta las celdas en orden (entrenamiento: ~1–2 min).  

---

## 🔧 Mejoras Futuras  
- Probar **XGBoost** o **SMOTE** para mejorar recall.  
- Analizar interacciones entre variables categóricas.  
- Desarrollar interfaz web para predicciones en tiempo real.  

---

## 👤 Autor  
**Adrián Galván**  
**Fecha:** Septiembre 2025  

---

# 🥗 Predictive Analysis of Obesity and Nutrition Habits  

**Sector:** Public Health, Biotechnology, Nutrition  
**Tools:** Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy)  

---

## 📋 Overview  
This project analyzes the *Estimation of Obesity Levels Based on Eating Habits and Physical Condition* dataset from the **UCI Machine Learning Repository** to **predict obesity levels (7 classes)** based on eating habits, physical activity, and demographic characteristics.  

It provides insights for **preventive health and personalized nutrition programs**, supporting early detection and tailored interventions.  

---

## 📊 Dataset  
- **Source:** [UCI Machine Learning Repository – Obesity Levels Dataset](https://archive.ics.uci.edu/ml/datasets/Estimation+of+Obesity+Levels+Based+on+Eating+Habits+and+Physical+Condition)  
- **Size:** 2,111 instances, 17 features  
- **File:** `ObesityDataSet_raw_and_data_sinthetic.csv`  

---

## 🔍 Methodology  
1. **Data Cleaning**  
   - No missing values.  
   - Categorical encoding applied.  

2. **Exploratory Data Analysis (EDA)**  
   - 7 balanced classes (~13–16% each).  
   - *Weight* distribution skewed toward higher values.  
   - Boxplots show clear variation by obesity level.  
   - Low correlations (< 0.5) among numeric features.  

3. **Hypothesis Testing**  
   - *Chi-square test* revealed a significant association (p ≈ 1.48 × 10⁻⁴⁷) between calorie-rich food consumption (*FAVC*) and obesity level (*NObeyesdad*).  

4. **Modeling**  
   - 80/20 stratified split (train/test).  
   - Standardized numeric variables.  
   - Model: *Random Forest Classifier* (100 trees, depth = 10).  

5. **Model Evaluation**  
   - Accuracy: **94.33%**  
   - F1-scores > **0.87** for most classes.  
   - Top predictor: **Weight (importance = 0.365)**.  

6. **Visualizations**  
   - Confusion matrix.  
   - Feature importance chart.  

---

## 🌎 Key Findings  
- **Weight** and **vegetable intake frequency (FCVC)** are the most influential predictors.  
- The model achieved **94% classification accuracy** across 7 categories.  
- Useful for developing targeted public health interventions.  

---

## 🧠 Real-World Applications  
- Nutritional risk assessment tools for healthcare providers.  
- Early detection support for obesity-related conditions.  
- Foundation for digital health and fitness apps.  

---

## ⚙️ Execution Requirements  
- Python 3.8+  
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`  
- File: `ObesityDataSet_raw_and_data_sinthetic.csv`  

Quick install:
- pip install pandas numpy matplotlib seaborn scikit-learn scipy


---

## 🚀 How to Run  
1. Download the dataset and place it in the notebook directory.  
2. Open `Nutricion_Obesidad.ipynb` in Jupyter Notebook.  
3. Run all cells sequentially (training time: ~1–2 min).  

---

## 🔧 Future Improvements  
- Implement **XGBoost** or **SMOTE** for better recall.  
- Explore categorical feature interactions.  
- Develop a **web app for real-time predictions**.  

---

## 👤 Author  
**Adrián Galván**  
**Date:** September 2025  
