import os
import shutil

# Definir los directorios de origen y destino
image_source_dirs = [
    "/mnt/homeGPU/azapata/TFG/datasets/VOC/images/train2007",
    "/mnt/homeGPU/azapata/TFG/datasets/VOC/images/train2012",
    "/mnt/homeGPU/azapata/TFG/datasets/VOC/images/val2007",
    "/mnt/homeGPU/azapata/TFG/datasets/VOC/images/val2012",
]

label_source_dirs = [
    "/mnt/homeGPU/azapata/TFG/datasets/VOC/labels/train2007",
    "/mnt/homeGPU/azapata/TFG/datasets/VOC/labels/train2012",
    "/mnt/homeGPU/azapata/TFG/datasets/VOC/labels/val2007",
    "/mnt/homeGPU/azapata/TFG/datasets/VOC/labels/val2012",
]

image_dest_dir = "/mnt/homeGPU/azapata/TFG/datasets/VOC/VOC_DM/images"
label_dest_dir = "/mnt/homeGPU/azapata/TFG/datasets/VOC/VOC_DM/labels"

# Asegurar que las carpetas de destino existen
os.makedirs(image_dest_dir, exist_ok=True)
os.makedirs(label_dest_dir, exist_ok=True)

def copy_files(source_dirs, dest_dir):
    """
    Copia archivos de una lista de directorios de origen a un directorio destino,
    asegurándose de no sobrescribir archivos con el mismo nombre.
    """
    for src_dir in source_dirs:
        if not os.path.exists(src_dir):
            print(f"Advertencia: El directorio {src_dir} no existe. Se omite.")
            continue
        
        for file_name in os.listdir(src_dir):
            src_file = os.path.join(src_dir, file_name)
            dest_file = os.path.join(dest_dir, file_name)

            # Verificar si el archivo ya existe en el destino
            if os.path.exists(dest_file):
                base, ext = os.path.splitext(file_name)
                counter = 1
                new_file_name = f"{base}_{counter}{ext}"
                dest_file = os.path.join(dest_dir, new_file_name)

                # Buscar el siguiente nombre disponible
                while os.path.exists(dest_file):
                    counter += 1
                    new_file_name = f"{base}_{counter}{ext}"
                    dest_file = os.path.join(dest_dir, new_file_name)

            # Copiar archivo al destino
            shutil.copy2(src_file, dest_file)
            print(f"Copiado: {src_file} -> {dest_file}")

# Copiar imágenes y etiquetas
copy_files(image_source_dirs, image_dest_dir)
copy_files(label_source_dirs, label_dest_dir)

print("Proceso de copiado completado.")
