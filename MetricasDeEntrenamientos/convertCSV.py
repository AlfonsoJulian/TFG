import os
import pandas as pd
import re

def parse_metrics(file_path):
    """Lee un archivo de métricas y lo convierte en un DataFrame"""
    data = []
    experiment_name = os.path.basename(file_path).replace(".txt", "")
    
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        match = re.match(r"\s*(\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if match:
            class_name, images, instances, precision, recall, map50, map50_95 = match.groups()
            data.append([experiment_name, class_name, int(images), int(instances), float(precision), float(recall), float(map50), float(map50_95)])

    df = pd.DataFrame(data, columns=["Experiment", "Class", "Images", "Instances", "Precision", "Recall", "mAP50", "mAP50-95"])
    return df

# Carpeta donde están almacenados los archivos de métricas
metrics_folder = "/mnt/homeGPU/azapata/TFG/MetricasDeEntrenamientos/"

# Crear un DataFrame vacío para almacenar todos los datos
all_metrics = pd.DataFrame()

# Recorrer los archivos en la carpeta y procesar cada uno
for file in os.listdir(metrics_folder):
    if file.endswith(".txt"):
        file_path = os.path.join(metrics_folder, file)
        df_metrics = parse_metrics(file_path)
        all_metrics = pd.concat([all_metrics, df_metrics], ignore_index=True)

# Guardar en CSV y Excel
csv_output = "metricas_entrenamientos.csv"
excel_output = "metricas_entrenamientos.xlsx"

all_metrics.to_csv(csv_output, index=False)
all_metrics.to_excel(excel_output, index=False)

print(f"Archivos generados: {csv_output} y {excel_output}")