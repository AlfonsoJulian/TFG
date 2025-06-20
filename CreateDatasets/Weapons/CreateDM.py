import os
import cv2
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

# 📌 Directorio de imágenes y etiquetas
image_dir = "/mnt/homeGPU/azapata/TFG/datasets/gunsUGR/OD-WeaponDetection-master/Weapons_and_similar_handled_objects/Sohas_weapon-Detection-YOLOv5/obj_train_data/images/train"
label_dir = "/mnt/homeGPU/azapata/TFG/datasets/gunsUGR/OD-WeaponDetection-master/Weapons_and_similar_handled_objects/Sohas_weapon-Detection-YOLOv5/obj_train_data/labels/train"
output_dir = "/mnt/homeGPU/azapata/TFG/datasets/gunsUGR/OD-WeaponDetection-master/Weapons_and_similar_handled_objects/Sohas_weapon-Detection-YOLOv5/obj_train_data/weapons_DM"

# 📌 Crear directorios de salida si no existen
output_image_dir = os.path.join(output_dir, "images")
output_label_dir = os.path.join(output_dir, "labels")
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# 📌 Cargar el modelo de difusión
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.init()
print(torch.cuda.is_available())  # Debe devolver True 
# Verificar si CUDA está disponible
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU en uso: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Ninguna'}")

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "/mnt/homeGPU/azapata/models/Realistic_Vision",  # Ruta local del modelo
    torch_dtype=torch.float16
).to(device)

# 📌 Diccionario con las clases de weapons
weapons_CLASSES = {
    0: "pistol", 1: "smartphone", 2: "knife", 3: "monedero", 4: "billete", 5: "tarjeta"
}


def iou(bb1, bb2):
    """Calcula la Intersección sobre la Unión (IoU) entre dos bounding boxes."""
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0

def get_non_overlapping_boxes(bboxes, iou_threshold=0.05):
    """Filtra las bounding boxes que no tienen superposición con ninguna otra."""
    non_overlapping = []
    for i in range(len(bboxes)):
        class_id1, x1, y1, w1, h1 = bboxes[i]
        overlaps = False
        for j in range(i + 1, len(bboxes)):
            class_id2, x2, y2, w2, h2 = bboxes[j]
            if iou((x1, y1, w1, h1), (x2, y2, w2, h2)) > iou_threshold:
                overlaps = True
                break
        if not overlaps:
            non_overlapping.append((class_id1, x1, y1, w1, h1))
    return non_overlapping

def apply_diffusion(image, bboxes):
    """Genera nuevas versiones de los objetos en las bounding boxes usando Stable Diffusion."""
    image_pil = Image.fromarray(image)
    modified = False
    for class_id, x, y, w, h in bboxes:
        # Verificar que las dimensiones sean válidas antes de recortar
        if w <= 0 or h <= 0:
            print(f"[WARNING] Bounding box con tamaño inválido: x={x}, y={y}, w={w}, h={h}, clase={class_id}")
            continue
        
        class_name = weapons_CLASSES.get(class_id, "object")
        cropped_image = image_pil.crop((x, y, x + w, y + h)).resize((512, 512))
        prompt = f"Ultra-realistic {class_name}, detailed, photorealistic"
        generated_image = pipe(prompt=prompt, image=cropped_image, strength=0.35, guidance_scale=6.5, num_inference_steps=50).images[0]
        generated_resized = generated_image.resize((w, h))
        image_pil.paste(generated_resized, (x, y))
        modified = True
    return image_pil if modified else None

for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue
    image_file = label_file.replace(".txt", ".jpg")
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, label_file)
    if not os.path.exists(image_path):
        continue
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    with open(label_path, "r") as f:
        lines = f.readlines()
    bboxes = [(int(parts[0]), int((float(parts[1]) - float(parts[3]) / 2) * width), int((float(parts[2]) - float(parts[4]) / 2) * height), int(float(parts[3]) * width), int(float(parts[4]) * height)) for parts in (line.strip().split() for line in lines)]
    non_overlapping_bboxes = get_non_overlapping_boxes(bboxes)
    if non_overlapping_bboxes:
        augmented_image = apply_diffusion(image, non_overlapping_bboxes)
        if augmented_image:
            augmented_image.save(os.path.join(output_image_dir, f"aug_{image_file}"))
            with open(os.path.join(output_label_dir, f"aug_{label_file}"), "w") as f:
                f.writelines(lines)
print("🚀 Create DM dataset con weapons 100%.")
