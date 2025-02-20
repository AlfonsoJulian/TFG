# 1. **Strength (Fuerza de modificación)**
#    - Define **cuánto cambia la imagen original** en la generación.
#    - Valores cercanos a 0 mantienen la imagen original casi intacta.
#    - Valores cercanos a 1 generan una imagen completamente nueva basada en el prompt.
#    - En este código, el `strength` se ajusta según el tamaño del objeto:
#      - **Objetos grandes (`> 9216 píxeles²`)** → `strength = 0.75` (más cambio, más detalles nuevos).
#      - **Objetos medianos (`1024 - 9216 píxeles²`)** → `strength = 0.5` (modificación moderada).
#      - **Objetos pequeños (`< 1024 píxeles²`)** → `strength = 0.25` (poca modificación para conservar detalles clave).

# 2. **Guidance Scale (Escala de orientación)**
#    - Controla **cuánto sigue el modelo el prompt textual**.
#    - Valores altos hacen que la imagen generada siga más fielmente el prompt.
#    - Valores bajos permiten más variabilidad y detalles del ruido de difusión.
#    - En este código, el `guidance_scale` se ajusta según la dificultad de detección de la clase en los resultados previos:
#      - **Clases fáciles (e.g., persona, coche, perro, tren)** → `guidance_scale = 7.0` (generación más detallada).
#      - **Clases intermedias (e.g., semáforo, laptop, skate)** → `guidance_scale = 5.5` (balance entre precisión y variabilidad).
#      - **Clases difíciles (e.g., tijeras, libro, mando a distancia, cepillo de dientes)** → `guidance_scale = 4.5` (permitiendo más variabilidad para mejorar).


import os
import shutil
import cv2
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from diffusers import StableDiffusionImg2ImgPipeline

# 📌 Directorio de imágenes y etiquetas
image_dir = "/mnt/homeGPU/azapata/TFG/datasets/coco/coco_20/images/train2017"
label_dir = "/mnt/homeGPU/azapata/TFG/datasets/coco/coco_20/labels/train2017"
output_dir = "/mnt/homeGPU/azapata/TFG/datasets/coco/coco_20_DM_augmented_segun_size"

# 📌 Crear directorios de salida si no existen
output_image_dir = os.path.join(output_dir, "images")
output_label_dir = os.path.join(output_dir, "labels")
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# 📌 Cargar el modelo de difusión
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "/mnt/homeGPU/azapata/models/Realistic_Vision",
    torch_dtype=torch.float16
).to(device)

# 📌 Diccionario con las clases de COCO
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair dryer", 79: "toothbrush"
}


# 📌 Funciones de utilidad
def iou(bb1, bb2):
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Atendemos a la clasificación de tamaño según COCO
def classify_object_size(w, h):
    area = w * h
    if area < 1024:
        return "small"
    elif area < 9216:
        return "medium"
    return "large"

def get_dynamic_iou_threshold(bboxes):
    num_objects = len(bboxes)
    return 0.15 if num_objects > 10 else 0.1 if num_objects > 5 else 0.05

def get_non_overlapping_boxes(bboxes, iou_threshold):
    non_overlapping = []
    for i in range(len(bboxes)):
        class_id1, x1, y1, w1, h1 = bboxes[i]
        if all(iou((x1, y1, w1, h1), (x2, y2, w2, h2)) <= iou_threshold for j, (class_id2, x2, y2, w2, h2) in enumerate(bboxes) if i != j):
            non_overlapping.append((class_id1, x1, y1, w1, h1))
    return non_overlapping

def determine_strength(w, h):
    size_category = classify_object_size(w, h)
    return 0.5 if size_category == "large" else 0.35 if size_category == "medium" else 0.15

def adjust_guidance_scale(class_name):
    high_performance = {"person", "car", "dog", "bus", "train", "airplane", "elephant", "zebra", "giraffe"}
    medium_performance = {"motorcycle", "bicycle", "stop sign", "fire hydrant", "frisbee", "skateboard", "tennis racket", "pizza", "clock", "toilet", "laptop", "mouse"}
    low_performance = {"boat", "traffic light", "bench", "bird", "backpack", "handbag", "tie", "suitcase", "baseball bat", "knife", "spoon", "toothbrush", "book", "scissors", "hair drier"}
    
    if class_name in high_performance:
        return 5.5  # Mayor guidance para clases con buen desempeño
    elif class_name in medium_performance:
        return 3.5  # Ajuste intermedio
    elif class_name in low_performance:
        return 1.5  # Menor guidance para clases con bajo desempeño
    else:
        return 3.5  # Valor por defecto para clases no categorizadas

def validate_augmentation(original_image, modified_image):
    original_gray = np.array(original_image.convert("L"))
    modified_gray = np.array(modified_image.convert("L"))
    return ssim(original_gray, modified_gray) > 0.6

def apply_diffusion(image, bboxes):
    image_pil = Image.fromarray(image)
    modified = False
    for class_id, x, y, w, h in bboxes:
        class_name = COCO_CLASSES.get(class_id, "object")
        cropped_image = image_pil.crop((x, y, x + w, y + h)).resize((512, 512))
        prompt = f"Ultra-realistic {class_name}, detailed, photorealistic"
        strength = determine_strength(w, h)
        guidance_scale = adjust_guidance_scale(class_name)
        generated_image = pipe(prompt=prompt, image=cropped_image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=50).images[0]
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
    
    iou_threshold = get_dynamic_iou_threshold(bboxes)
    non_overlapping_bboxes = get_non_overlapping_boxes(bboxes, iou_threshold)
    
    # Guardar imagen original con su etiqueta
    shutil.copy(image_path, os.path.join(output_image_dir, image_file))
    shutil.copy(label_path, os.path.join(output_label_dir, label_file))
    
    # Generar y guardar imagen aumentada con prefijo
    if non_overlapping_bboxes:
        augmented_image = apply_diffusion(image, non_overlapping_bboxes)
        
        if augmented_image and validate_augmentation(Image.fromarray(image), augmented_image):
            augmented_image.save(os.path.join(output_image_dir, f"aug_{image_file}"))
            with open(os.path.join(output_label_dir, f"aug_{label_file}"), "w") as f:
                f.writelines(lines)        

print("🚀 Data augmentation completado con almacenamiento estructurado.")
