MALIS Project 1: Fast Leave-One-Out Cross-Validation for KNN Regression
Este proyecto se centra en el problema de la validación de modelos para identificar los mejores hiperparámetros, con un enfoque específico en K-Nearest Neighbors (KNN) para regresión. El objetivo principal es implementar y comparar el método de validación cruzada "Leave-One-Out" (LOOCV) estándar con un algoritmo acelerado propuesto en un paper de investigación.

📝 Resumen del Proyecto
El proyecto se divide en tres partes principales:

Investigación Teórica: Estudiar los conceptos de validación cruzada y, en particular, LOOCV.

Implementación: Codificar en Python tanto el algoritmo LOOCV estándar como una versión rápida y optimizada para regresores KNN, basándose en el paper de Motonobu Kanagawa.

Evaluación Experimental: Utilizar las implementaciones para reproducir los resultados del paper, comparando el rendimiento de ambos métodos y documentando los hallazgos en un informe y un Jupyter Notebook.

📂 Estructura del Repositorio
.
├── loocv_knn.py         # Módulo con la implementación de los algoritmos LOOCV
├── experiment.ipynb     # Notebook para ejecutar los experimentos y generar gráficos
├── report.pdf           # Informe final del proyecto
└── README.md            # Este archivo
🚀 Cómo Empezar
Prerrequisitos
Asegúrate de tener Python 3 instalado, junto con las siguientes librerías:

numpy

scikit-learn

matplotlib

jupyter

Puedes instalarlas usando pip:

Bash

pip install numpy scikit-learn matplotlib jupyter
Ejecución de los Experimentos
Abre el Jupyter Notebook experiment.ipynb.

Ejecuta todas las celdas en orden. El notebook importará las funciones do_normal_loocv y do_fast_loocv desde loocv_knn.py.

El notebook generará los gráficos (reproduciendo las Figuras 2 y 3 del paper) y mostrará los resultados comparativos de rendimiento.

📋 Tareas y Entregables
Parte I: Investigación Teórica
Tarea 1: Investigar el propósito de la validación cruzada y LOOCV, sus ventajas, desventajas y escenarios de uso. Los hallazgos se resumen en el informe final.

Parte II: Implementación
Tarea 2: Leer y comprender el paper "Fast Exact Leave-One-Out Cross-Validation for K-Nearest-Neighbor Regressor" de Motonobu Kanagawa.

Tarea 3 & 4: Implementar las funciones do_fast_loocv y do_normal_loocv en el archivo loocv_knn.py siguiendo el esqueleto proporcionado. El código está debidamente documentado para explicar la lógica interna.

Parte III: Evaluación
Tarea 5: Crear el notebook experiment.ipynb para realizar experimentos que comparen el tiempo de ejecución y los resultados de ambas implementaciones. Los resultados se visualizan y analizan para validar la eficacia del método rápido.

Entregables
El proyecto se entrega como un archivo ZIP que contiene:

Código Fuente: loocv_knn.py y experiment.ipynb.

Informe: Un reporte en PDF (report.pdf) de 500-750 palabras que incluye:

Un resumen sobre la validación cruzada y LOOCV.

Una explicación de por qué la validación cruzada puede ser computacionalmente costosa para KNN.

Un análisis comparativo de los resultados obtenidos en los experimentos.

Contribuciones: Una sección en el informe que detalla la aportación de cada miembro del equipo al proyecto.

🧑‍🤝‍🧑 Contribuciones
[Nombre del miembro 1]: [Descripción de su contribución]

[Nombre del miembro 2]: [Descripción de su contribución]

[Nombre del miembro 3]: [Descripción de su contribución]

⚠️ Nota sobre el Uso de IA
El uso de herramientas de IA como ChatGPT para asistir en este proyecto debe ser reportado. Si se utilizó, se debe incluir una explicación clara de cómo y para qué partes del proyecto se empleó. La falta de transparencia resultará en una calificación de cero.
