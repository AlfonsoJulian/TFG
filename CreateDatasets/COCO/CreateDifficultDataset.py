import os
import shutil

# Definir rutas de los datasets
IMAGE_SRC_DIR = "/mnt/homeGPU/azapata/TFG/datasets/coco/images/train2017"
LABEL_SRC_DIR = "/mnt/homeGPU/azapata/TFG/datasets/coco/labels/train2017"

IMAGE_DEST_DIR = "/mnt/homeGPU/azapata/TFG/datasets/coco/cocoDifficult/images"
LABEL_DEST_DIR = "/mnt/homeGPU/azapata/TFG/datasets/coco/cocoDifficult/labels"

# Asegurar que las carpetas de destino existen
os.makedirs(IMAGE_DEST_DIR, exist_ok=True)
os.makedirs(LABEL_DEST_DIR, exist_ok=True)

# Clases difíciles según los números de COCO
DIFFICULT_CLASSES = {26, 65, 43, 44, 73, 76, 78, 79}  # handbag, remote, knife, spoon, book, scissors, hair dryer, toothbrush

# Función para leer etiquetas y filtrar
def filter_labels():
    selected_images = set()  # Guardar las imágenes que cumplen el criterio

    for label_file in os.listdir(LABEL_SRC_DIR):
        label_path = os.path.join(LABEL_SRC_DIR, label_file)
        
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Filtrar líneas que contienen alguna de las clases difíciles
        filtered_lines = [line for line in lines if int(line.split()[0]) in DIFFICULT_CLASSES]

        if filtered_lines:  # Si hay alguna línea filtrada, mantener la imagen
            selected_images.add(label_file.replace(".txt", ".jpg"))  # Asumiendo que las imágenes son .jpg
            
            # Guardar la nueva etiqueta filtrada
            with open(os.path.join(LABEL_DEST_DIR, label_file), "w") as f_out:
                f_out.writelines(filtered_lines)

    return selected_images

# Función para copiar imágenes seleccionadas
def copy_images(selected_images):
    for image_name in selected_images:
        src_path = os.path.join(IMAGE_SRC_DIR, image_name)
        dest_path = os.path.join(IMAGE_DEST_DIR, image_name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            

# Ejecutar el filtrado
selected_images = filter_labels()
copy_images(selected_images)

print(f"✅ Filtrado completo. Se han guardado {len(selected_images)} imágenes y sus etiquetas correspondientes.")
