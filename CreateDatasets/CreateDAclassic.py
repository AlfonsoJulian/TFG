import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
import shutil

# Rutas
data_dir = "/mnt/homeGPU/azapata/TFG/datasets/coco"
augmented_dir = "/mnt/homeGPU/azapata/TFG/datasets/coco/coco_100_DA_classic"

# Crear directorios si no existen
os.makedirs(os.path.join(augmented_dir, "images", "train2017"), exist_ok=True)
os.makedirs(os.path.join(augmented_dir, "labels", "train2017"), exist_ok=True)

# Definir transformaciones (reducidas)
transform = A.Compose([
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.1),
    A.RandomRotate90(p=0.2),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
    A.RandomResizedCrop(height=640, width=640, scale=(0.9, 1.0), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.1),
])

# Directorios de imágenes y etiquetas originales
image_dir = os.path.join(data_dir, "images", "train2017")
label_dir = os.path.join(data_dir, "labels", "train2017")

# Recorrer todas las imágenes
for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"❌ Error al leer {img_name}, saltando...")
        continue
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Copiar la imagen original a la carpeta de imágenes aumentadas
    dest_img_path = os.path.join(augmented_dir, "images", "train2017", img_name)
    shutil.copy(img_path, dest_img_path)

    # Aplicar transformación y guardar la imagen aumentada
    augmented = transform(image=img)["image"]
    aug_img_name = img_name.replace(".jpg", "_aug.jpg")  # Agregar "_aug" al nombre
    aug_img_path = os.path.join(augmented_dir, "images", "train2017", aug_img_name)
    
    # Guardar la imagen aumentada
    cv2.imwrite(aug_img_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

    # Copiar etiqueta original
    label_name = img_name.replace(".jpg", ".txt")
    src_label_path = os.path.join(label_dir, label_name)
    dst_label_path = os.path.join(augmented_dir, "labels", "train2017", label_name)

    if os.path.exists(src_label_path):
        shutil.copy(src_label_path, dst_label_path)

        # También duplicar la etiqueta para la imagen aumentada
        aug_label_path = os.path.join(augmented_dir, "labels", "train2017", label_name.replace(".txt", "_aug.txt"))
        shutil.copy(src_label_path, aug_label_path)

print("✅ Dataset fusionado con éxito: imágenes originales y aumentadas incluidas.")
