from ultralytics import YOLO
import datetime

# 📌 **Definir rutas del dataset aumentado y de validación**
dataset_augmented_yaml = "/mnt/homeGPU/azapata/TFG/datasets/coco/coco20clasicaugmented.yaml"
dataset_val_yaml = "/mnt/homeGPU/azapata/TFG/datasets/coco/coco20.yaml"
results_file = "results_yolo_classic_augmented.txt"  # Archivo para guardar los resultados

# 📌 **Abrir archivo de resultados**
with open(results_file, "w") as f:
    f.write("🚀 Resultados del Fine-Tuning de YOLOv8\n")
    f.write(f"Fecha y hora de ejecución: {datetime.datetime.now()}\n")
    f.write("="*50 + "\n\n")

# 📌 **Cargar el modelo preentrenado**
print("🚀 Cargando YOLOv8 preentrenado...")
model = YOLO('yolov8n.pt')

# 📌 **Entrenar con el dataset aumentado**
print("🚀 Iniciando el Fine-Tuning de YOLOv8 con el dataset aumentado...")
try:
    train_results = model.train(data=dataset_augmented_yaml,
                                epochs=50,
                                batch=16,  # Tamaño óptimo para la Quadro RTX 8000
                                pretrained=True,
                                device="cuda",
                                show=True)

    with open(results_file, "a") as f:
        f.write("✅ Fine-Tuning completado correctamente.\n")
        f.write("📌 Hiperparámetros utilizados:\n")
        f.write(f"- Dataset: {dataset_augmented_yaml}\n")
        f.write(f"- Épocas: 50\n")
        f.write(f"- Batch Size: 16\n")
        f.write(f"- Modelo Preentrenado: Sí\n")
        f.write(f"- Dispositivo: CUDA\n\n")

    print("✅ Fine-tuning completado correctamente.")
except Exception as e:
    with open(results_file, "a") as f:
        f.write(f"❌ Error en el entrenamiento: {e}\n")
    print(f"❌ Error en el entrenamiento: {e}")

# 📌 **Evaluación del modelo después del fine-tuning**
print("🚀 Evaluando el modelo tras el entrenamiento...")
try:
    metrics = model.val(data=dataset_val_yaml, split="val", device="cuda")

    # 📌 **Guardar métricas clave en el archivo**
    with open(results_file, "a") as f:
        f.write("="*50 + "\n")
        f.write("📊 Resultados de la Evaluación\n")
        f.write(f"📊 Precisión (mAP@50): {metrics.box.map50:.4f}\n")
        f.write(f"📊 Precisión (mAP@50-95): {metrics.box.map:.4f}\n")
        f.write(f"📊 Precisión promedio por clase: {metrics.box.maps}\n")
        f.write("="*50 + "\n")

    print("✅ Evaluación completada correctamente.")
except Exception as e:
    with open(results_file, "a") as f:
        f.write(f"❌ Error en la evaluación: {e}\n")
    print(f"❌ Error en la evaluación: {e}")

# 📌 **Confirmación final**
print(f"📄 Resultados guardados en {results_file}")
