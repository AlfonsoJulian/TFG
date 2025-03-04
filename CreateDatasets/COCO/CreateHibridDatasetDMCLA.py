# Creamos un dataset híbrido que mezcle las imagenes originales con las que tienen DA clásico y las que tienen DA con DM
import os
import shutil

# Rutas de los datasets

orig_images = "/mnt/homeGPU/azapata/TFG/datasets/coco/coco_20/images/train2017"
orig_labels = "/mnt/homeGPU/azapata/TFG/datasets/coco/coco_20/labels/train2017"

classic_images = "/mnt/homeGPU/azapata/TFG/datasets/coco/coco_20_classic_augmented/images/train2017"
classic_labels = "/mnt/homeGPU/azapata/TFG/datasets/coco/coco_20_classic_augmented/labels/train2017"

dm_images = "/mnt/homeGPU/azapata/TFG/datasets/coco/coco_20_DM_augmented/images"
dm_labels = "/mnt/homeGPU/azapata/TFG/datasets/coco/coco_20_DM_augmented/labels"

# Nueva carpeta destino
hib_images = "/mnt/homeGPU/azapata/TFG/datasets/coco/coco_20_DA_HIB/images/train2017"
hib_labels = "/mnt/homeGPU/azapata/TFG/datasets/coco/coco_20_DA_HIB/labels/train2017"

# Crear las carpetas destino si no existen
os.makedirs(hib_images, exist_ok=True)
os.makedirs(hib_labels, exist_ok=True)

def copy_images_and_labels(src_img, src_lbl, dst_img, dst_lbl, prefix_filter=None):
    for file in os.listdir(src_img):
        if prefix_filter and not file.startswith(prefix_filter):
            continue
        img_src_path = os.path.join(src_img, file)
        lbl_src_path = os.path.join(src_lbl, file.replace(".jpg", ".txt").replace(".png", ".txt"))
        
        img_dst_path = os.path.join(dst_img, file)
        lbl_dst_path = os.path.join(dst_lbl, file.replace(".jpg", ".txt").replace(".png", ".txt"))
        
        shutil.copy(img_src_path, img_dst_path)
        if os.path.exists(lbl_src_path):
            shutil.copy(lbl_src_path, lbl_dst_path)

# Copiar imágenes aumentadas con DM (solo las con "aug_")
copy_images_and_labels(dm_images, dm_labels, hib_images, hib_labels, prefix_filter="aug_")

# Copiar imágenes aumentadas con DA clásico (evitando duplicados)
copy_images_and_labels(classic_images, classic_labels, hib_images, hib_labels)

# Copiar imágenes originales (evitando duplicados)
copy_images_and_labels(orig_images, orig_labels, hib_images, hib_labels)

print("Dataset híbrido creado exitosamente en", hib_images)
