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
        data="/mnt/homeGPU/azapata/TFG/datasets/gunsUGR/OD-WeaponDetection-master/Weapons_and_similar_handled_objects/Sohas_weapon-Detection-YOLOv5/dataset.yaml",
        epochs=50,  # Ajusta según necesidad
        batch=2,  # Reduce si hay problemas de memoria
        device="cuda",  # Entrenamiento en GPU
        amp=True,  # Habilitar precisión mixta para ahorrar memoria
        val=True,  
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

if __name__ == "__main__":
    main()
