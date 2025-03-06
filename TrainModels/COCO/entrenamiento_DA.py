from ultralytics import YOLO
import datetime

# 📌 **Definir rutas del dataset y archivo de resultados**
dataset_yaml = "/mnt/homeGPU/azapata/TFG/datasets/coco/coco_100_DA_classic.yaml"
results_file = "results_yolo_classic_augmented.txt"  # Archivo para guardar los resultados

# 📌 **Abrir archivo de resultados**
with open(results_file, "w") as f:
    f.write("🚀 Resultados del entrenamiento de YOLOv8 COCO100% DA classic\n")
    f.write(f"Fecha y hora de ejecución: {datetime.datetime.now()}\n")
    f.write("="*50 + "\n\n")

# 📌 **Cargar el modelo YOLOv8**
print("🚀 Cargando YOLOv8 desde cero...")
model = YOLO('yolov8n.yaml')  # Para entrenar desde cero, usa un archivo .yaml

# 📌 **Entrenar con el dataset aumentado**
print("🚀 Iniciando el entrenamiento de YOLOv8 con el 100% del dataset COCO más los datos con DA classic")
try:
    train_results = model.train(
        data=dataset_yaml,     
        epochs=50,
        batch=32,  # Ajustar según VRAM
        pretrained=False,
        device="cuda",
        # cache=True,  # Usa cache para mejorar la carga de datos
        show=True,
        augment=True,  # Habilita Data Augmentation nativo de YOLOv8  
        hsv_h=0.015,  # Pequeña variación de tono  
        hsv_s=0.7,  # Aumento de saturación  
        hsv_v=0.4,  # Aumento de brillo  
        flipud=0.2,  # Flipping vertical  
        fliplr=0.5,  # Flipping horizontal  
        mosaic=1.0,  # Habilita la técnica de Mosaic  
        mixup=0.2,  # Habilita MixUp  
        scale=0.5,  # Aumenta el escalado aleatorio  
        translate=0.1  # Pequeña traslación aleatoria  
    )
    
    # 📌 **Guardar hiperparámetros y resultados**
    with open(results_file, "a") as f:
        f.write("✅ Entrenamiento completado correctamente.\n")
        f.write("📌 Hiperparámetros utilizados:\n")
        f.write(f"- Dataset: {dataset_yaml}\n")
        f.write(f"- Épocas: 50\n")
        f.write(f"- Batch Size: 32\n")  # Antes estaba mal puesto como 16
        f.write(f"- Modelo Preentrenado: No\n")
        f.write(f"- Dispositivo: CUDA\n\n")

    print("✅ Entrenamiento completado correctamente.")
except Exception as e:
    with open(results_file, "a") as f:
        f.write(f"❌ Error en el entrenamiento: {e}\n")
    print(f"❌ Error en el entrenamiento: {e}")

# 📌 **Evaluación del modelo después del fine-tuning**
print("🚀 Evaluando el modelo tras el entrenamiento...")
try:
    metrics = model.val(data=dataset_yaml, device="cuda")

    # 📌 **Guardar métricas clave en el archivo**
    with open(results_file, "a") as f:
        f.write("="*50 + "\n")
        f.write("📊 Resultados de la Evaluación de COCO 100% con DA\n")
        f.write(f"📊 Precisión (mAP@50): {metrics['metrics/mAP_50(B)']:.4f}\n")
        f.write(f"📊 Precisión (mAP@50-95): {metrics['metrics/mAP_50-95(B)']:.4f}\n")
        f.write("="*50 + "\n")

    print("✅ Evaluación completada correctamente.")
except Exception as e:
    with open(results_file, "a") as f:
        f.write(f"❌ Error en la evaluación: {e}\n")
    print(f"❌ Error en la evaluación: {e}")

# 📌 **Confirmación final**
print(f"📄 Resultados guardados en {results_file}")
