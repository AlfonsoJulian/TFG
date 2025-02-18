# Trabajo Fin de Grado – Data Augmentation con Modelos de Difusión - Alfonso Julián Zapata Velasco

Este repositorio contiene el código y los experimentos de mi Trabajo de Fin de Grado (TFG), cuyo objetivo es **evaluar si los modelos de difusión pueden mejorar la detección de objetos pequeños a través de data augmentation**.

## 📌 Objetivo del Proyecto
Este proyecto busca responder a la pregunta:  
**¿Es posible mejorar la detección de objetos pequeños utilizando modelos de difusión para generar datos sintéticos?**  

Para ello, trabajamos con **COCO**, y comparamos diferentes técnicas de data augmentation tradicionales frente a modelos generativos.

Hay que tener en cuenta que hay muchos notebooks ya que al haber mucho trabajo detrás no se ha podido incluir todo en uno. Llegado el momento, los notebooks importantes tendrán `IMPORTANTE_<nombre>` para mayor claridad ya que voy a incluir todo el trabajo que he hecho desde el inicio hasta la actualidad.

---

## 📂 Estructura del Repositorio

```
📁 TFG
│── 📁 docs              # Documentación y dependencias
│   ├── environment.yml  # Configuración del entorno Conda
│   ├── requirements.txt # Dependencias del proyecto
│── 📁 modelos           # Modelos entrenados y configuraciones
│   ├── SSD.ipynb        # Notebook de pruebas y entender SSD
│   ├── fasterRCNN.ipynb # Notebook de pruebas y entender Faster R-CNN
│   ├── yolo.ipynb       # Notebook de pruebas y entender YOLO
│── 📁 nbCoco            # Notebooks relacionados con COCO
│   ├── InvestigacionCoco.ipynb # Notebook trabajando, descargando y modificando coco
│   ├── download_coco.py # Archivo para descargar el dataset (más de 300 mil imagenes)
│── 📁 nbInpainting      # Notebooks de inpainting y modelos de difusión
│   ├── inpainting.ipynb 
│── 📁 nbVisDrone        # Notebooks de experimentación con VisDrone
│   ├── TomaContactoVisDrone.ipynb
│── 📁 experimentacion   # Notebooks de experimentación general y análisis y conclusiones de resultados
│   ├── experimentacion.ipynb
│── 📁 datasets          # Carpeta para almacenar datasets procesados (COCO, 20% COCO, 20% COCO + DA, 20% COCO + DA DM, 20% COCO + DA DM + DA, ...)
│── .gitignore          # Archivos ignorados por Git
│── README.md           # Documentación principal del repositorio
│── CreateHibridDatasetDMCLA.py  # Script para crear el dataset híbrido (20% COCO + DA clásico + DA DM)
│── DA_ModelosDiffusion.py       # Data Augmentation con Diffusion Models (20% COCO + DA DM)
│── DA_ModelosDiffusionSegunSize.py # DA adaptado según tamaño (20% COCO + DA DM inteligente (adapta el guidance y el strength según tamaño y dificultad de la clase))
│── entrenamiento.py              # Script general de entrenamiento
│── entrenamientoDAMD.py           # Entrenamiento con DA+DM
│── entrenamiento_DA.py            # Entrenamiento con DA clásico
│── script_entrenamiento.sh        # Script para lanzar entrenamiento
```

### 📂 **Descripción de las carpetas y archivos principales**
- **`docs/`** → Documentación del TFG y dependencias.
- **`modelos/`** → Modelos entrenados y configuraciones de SSD, Faster R-CNN y YOLO.
- **`nbCoco/`** → Notebooks relacionados con COCO.
- **`nbInpainting/`** → Notebooks de inpainting y modelos de difusión.
- **`nbVisDrone/`** → Notebooks de experimentación con VisDrone.
- **`experimentacion/`** → Notebooks generales de experimentación.
- **`datasets/`** → Datos procesados y preparados para entrenamiento.
- **Scripts principales:**
  - `CreateHibridDatasetDMCLA.py` → Generación de dataset híbrido con DA clásico y DM.
  - `DA_ModelosDiffusion.py` → Crear dataset de Data Augmentation con Modelos de Difusión.
  - `DA_ModelosDiffusionSegunSize.py` → Crear dataset de DA DM adaptado según tamaño del objeto.
  - `entrenamiento.py` → Script principal de entrenamiento.
  - `entrenamientoDAMD.py` → Entrenamiento con DA y DM.
  - `entrenamiento_DA.py` → Entrenamiento con DA clásico.
  - `script_entrenamiento.sh` → Script para ejecutar entrenamiento en servidor.

---

## 🔬 Metodología

1. **Entrenamiento de modelos de detección de objetos**  
   - YOLO, Faster R-CNN, SSD sobre COCO y VisDrone.
2. **Generación de datos sintéticos con modelos de difusión**  
   - Aplicación de inpainting y generación de imágenes sintéticas.
3. **Comparación de rendimiento**  
   - Evaluación de modelos con y sin data augmentation generativo.
   - Evaluación de las diferentes aproximaciones con data augmentation generativo.
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
## 📌 **Autor**  
**Alfonso Julián**  

## 📂 **Repositorio GitHub**  
🔗 [TrabajoFinDeGrado](https://github.com/AlfonsoJulian/TFG)  

## 📅 **Periodo académico**  
🗓️ 2024/2025  

## 📑 **Investigación previa (Primer Cuatrimestre)**  
Durante el primer cuatrimestre, se llevó a cabo una investigación sobre:  
- Modelos de detección de objetos.  
- Conjuntos de datos (datasets) relevantes.  
- Técnicas de **inpainting** y su aplicación en visión por computadora.  
- Uso de **Stable Diffusion** y otros modelos generativos.  

> ⚠️ *Debido a limitaciones de espacio, los notebooks de esta fase no están incluidos en este repositorio.*  

Puedes acceder a ellos aquí:  
🔗 [Trabajo de investigación previo del primer cuatrimestre](https://github.com/AlfonsoJulian/TrabajoPrimerCuatriTFG)  

---

📚 ¡Cualquier feedback o sugerencia es bienvenido! 🚀