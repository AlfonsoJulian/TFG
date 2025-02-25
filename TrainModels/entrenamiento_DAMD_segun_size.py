from ultralytics import YOLO
import datetime

# 📌 **Definir rutas del dataset y archivo de resultados**
dataset_yaml = "/mnt/homeGPU/azapata/TFG/datasets/coco/coco20DADMsegunSize.yaml"
results_file = "results_yolo_classic_augmented.txt"  # Archivo para guardar los resultados

# 📌 **Abrir archivo de resultados**
with open(results_file, "w") as f:
    f.write("🚀 Resultados del Fine-Tuning de YOLOv8\n")
    f.write(f"Fecha y hora de ejecución: {datetime.datetime.now()}\n")
    f.write("="*50 + "\n\n")

# 📌 **Cargar el modelo YOLOv8**
print("🚀 Cargando YOLOv8 desde cero...")
model = YOLO('yolov8n.yaml')  # Para entrenar desde cero, usa un archivo .yaml

# 📌 **Entrenar con el dataset aumentado**
print("🚀 Iniciando el entrenamiento de YOLOv8 con el 20% del dataset COCO más los datos con DA")
try:
    train_results = model.train(
        data=dataset_yaml,     
        epochs=50,
        batch=32,  # Ajustar según VRAM
        pretrained=False,
        device="cuda",
        # cache=True,  # Usa cache para mejorar la carga de datos
        show=True
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
        f.write("📊 Resultados de la Evaluación\n")
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
