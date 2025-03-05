import os
import shutil
import cv2
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import torch

# Directorios
image_dir = "/mnt/homeGPU/azapata/TFG/datasets/coco/images/train2017"
label_dir = "/mnt/homeGPU/azapata/TFG/datasets/coco/labels/train2017"
output_dir = "/mnt/homeGPU/azapata/TFG/datasets/coco/coco_100_DM2"

# Crear directorios de salida si no existen
output_image_dir = os.path.join(output_dir, "images")
output_label_dir = os.path.join(output_dir, "labels")
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Cargar el modelo de difusión
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "/mnt/homeGPU/azapata/models/Realistic_Vision",  # Ruta local del modelo
    torch_dtype=torch.float16
).to(device)

# Copiar todas las imágenes originales al nuevo dataset
def copy_original_images():
    for image_file in os.listdir(image_dir):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(image_dir, image_file)
            output_image_path = os.path.join(output_image_dir, image_file)
            
            # Copiar imagen original
            shutil.copy(image_path, output_image_path)
            
            # Copiar la etiqueta correspondiente
            label_file = image_file.replace(".jpg", ".txt")
            label_path = os.path.join(label_dir, label_file)
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(output_label_dir, label_file))
                
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


# Función para aplicar difusión y generar imágenes aumentadas
def apply_diffusion(image, bboxes):
    image_pil = Image.fromarray(image)
    modified = False
    for class_id, x, y, w, h in bboxes:
        if w <= 0 or h <= 0:
            continue
        class_name = COCO_CLASSES.get(class_id, "object")
        cropped_image = image_pil.crop((x, y, x + w, y + h)).resize((512, 512))
        prompt = f"Ultra-realistic {class_name}, detailed, photorealistic"
        generated_image = pipe(prompt=prompt, image=cropped_image, strength=0.35, guidance_scale=6.5, num_inference_steps=50).images[0]
        generated_resized = generated_image.resize((w, h))
        image_pil.paste(generated_resized, (x, y))
        modified = True
    return image_pil if modified else None

# Generar imágenes aumentadas y etiquetarlas correctamente
def generate_augmented_images():
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue
        image_file = label_file.replace(".txt", ".jpg")
        
        # Verifica si la imagen tiene el prefijo "aug_" para asegurarse de que no se procese de nuevo
        if image_file.startswith("aug_"):
            continue  # Si la imagen es aumentada, la omitimos
        
        image_path = os.path.join(output_image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)
        if not os.path.exists(image_path):
            continue
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        
        # Leer las etiquetas de la imagen
        with open(label_path, "r") as f:
            lines = f.readlines()
        bboxes = [(int(parts[0]), int((float(parts[1]) - float(parts[3]) / 2) * width), int((float(parts[2]) - float(parts[4]) / 2) * height), int(float(parts[3]) * width), int(float(parts[4]) * height)) for parts in (line.strip().split() for line in lines)]
        
        non_overlapping_bboxes = get_non_overlapping_boxes(bboxes)
        if non_overlapping_bboxes:
            augmented_image = apply_diffusion(image, non_overlapping_bboxes)
            if augmented_image:
                augmented_image_path = os.path.join(output_image_dir, f"aug_{image_file}")
                augmented_image.save(augmented_image_path)
                
                # Guardar la nueva etiqueta
                augmented_label_path = os.path.join(output_label_dir, f"aug_{label_file}")
                with open(augmented_label_path, "w") as f:
                    f.writelines(lines)

# Ejecutar el proceso
print("🚀 Copiando imágenes originales al nuevo dataset...")
copy_original_images()

print("🚀 Generando imágenes aumentadas...")
generate_augmented_images()

print("🎉 Proceso completo: Dataset creado con imágenes originales y aumentadas.")
