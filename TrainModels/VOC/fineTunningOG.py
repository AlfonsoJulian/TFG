import torch
from ultralytics import YOLO

def main():
    # Liberar memoria GPU antes de iniciar
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # Cargar el modelo preentrenado de YOLOv8
    model = YOLO("yolov8l.pt")  

    # Configuración de entrenamiento
    model.train(
        data="/mnt/homeGPU/azapata/TFG/datasets/VOC/VOC.yaml",
        epochs=50,  # Ajusta según necesidad
        batch=2,  # Reduce si hay problemas de memoria
        device="cuda",  # Entrenamiento en GPU
        amp=True,  # Habilitar precisión mixta para ahorrar memoria
        val=True  # Evaluar después de entrenar
    )

if __name__ == "__main__":
    main()
