# Trabajo Fin de Grado – Data Augmentation con Modelos de Difusión - Alfonso Julián Zapata Velasco

Este repositorio contiene el código y los experimentos de mi Trabajo de Fin de Grado (TFG), cuyo objetivo es **evaluar si los modelos de difusión pueden mejorar la detección de objetos pequeños a través de data augmentation**.

## 📌 Objetivo del Proyecto
Este proyecto busca responder a la pregunta:  
**¿Es posible mejorar la detección de objetos pequeños utilizando modelos de difusión para generar datos sintéticos?**  

Para ello, trabajamos con **COCO**, y comparamos diferentes técnicas de data augmentation tradicionales frente a modelos generativos.

Hay que tener en cuenta que hay muchos notebooks ya que al haber mucho trabajo detrás no se ha podido incluir todo en uno. Llegado el momento, los notebooks importantes tendran IMPORTANTE<nombre> para mayor claridad ya que voy a incluir todo el trabajo que he hecho desde el inicio hasta la actualidad.

---

## 📂 Estructura del Repositorio

```
📁 TFG
│── 📁 docs          # Documentación del proyecto
│── 📁 modelos       # Pesos y configuraciones de modelos entrenados
│── 📁 nbCoco        # Notebooks de procesamiento y experimentación con COCO
│── 📁 nbInpainting  # Notebooks de experimentos con inpainting y difusión
│── 📁 nbVisDrone    # Notebooks para la exploración del dataset VisDrone
│── 📁 datasets      # Carpeta para almacenar datasets procesados
│── 📁 VisDrone      # Datos originales de VisDrone
│── .gitignore       # Archivos ignorados por Git
│── README.md        # Documentación principal del repositorio
```

### 📂 **Descripción de las carpetas**
- **`docs/`** → Documentación del TFG.
- **`modelos/`** → Modelos entrenados y configuraciones.
- **`nbCoco/`** → Notebooks relacionados con COCO.
- **`nbInpainting/`** → Notebooks de inpainting y modelos de difusión.
- **`nbVisDrone/`** → Notebooks de experimentación con VisDrone.
- **`datasets/`** → Datos procesados y preparados para entrenamiento.
- **`VisDrone/`** → Archivos originales del dataset VisDrone.

---

## 🔬 Metodología

1. **Entrenamiento de modelos de detección de objetos**  
   - YOLO, Faster R-CNN, SSD sobre COCO y VisDrone.
2. **Generación de datos sintéticos con modelos de difusión**  
   - Aplicación de inpainting y generación de imágenes sintéticas.
3. **Comparación de rendimiento**  
   - Evaluación de modelos con y sin data augmentation generativo.
   - Evaluación de las diferentes aproximaciones con data augmentation generativo
4. **Análisis de métricas**  
   - Uso de las métricas de COCO para medir impacto.

---

## ⚙️ Instalación y Dependencias

Para ejecutar los notebooks, instala las dependencias con:

```bash
pip install -r requirements.txt
```

Es recomendable utilizar un entorno con:
- Python 3.7+
- PyTorch
- TensorFlow (opcional)
- OpenCV
- COCO API
- Detectron2 (para Faster R-CNN)
- Git LFS (para manejar archivos grandes)
- Diffusers (para modelos de difusión)
- Transformers (para modelos generativos)
- Pandas y NumPy (para manipulación de datos)
- Matplotlib y Seaborn (para visualización)
- SciPy y Scikit-learn (para análisis estadístico y preprocesamiento)

---

## 📀 Resultados Esperados
Se analizará si los modelos de difusión pueden **generar datos sintéticos útiles** para mejorar el rendimiento de los detectores de objetos. También se comparará su impacto frente a otras técnicas tradicionales de data augmentation.

---

📌 **Autor**: Alfonso Julián  
👉 **Repositorio GitHub**: [TrabajoFinDeGrado](https://github.com/AlfonsoJulian/TFG)  
📅 **Fecha**: 2024/2025  
**Trabajo de investigacion previo del primer cuatri:** Se hace investigación de los modelos, de datasets, de cómo hacer inpainting y utilizar stable diffusion etc...

    [Trabajo de investigacion previo del primer cuatri](https://github.com/AlfonsoJulian/TrabajoPrimerCuatriTFG)

📚 ¡Cualquier feedback o sugerencia es bienvenido! 🚀