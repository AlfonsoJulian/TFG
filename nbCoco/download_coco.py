from ultralytics.utils.downloads import download
from pathlib import Path

# Definir el directorio base
dataset_path = Path("datasets/coco")

# Descargar etiquetas (segmentación o bounding boxes)
segments = False  # Cambia a True si quieres segmentación
url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/'
label_url = url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')
download([label_url], dir=dataset_path.parent)

# Descargar imágenes
image_urls = [
    'http://images.cocodataset.org/zips/train2017.zip',  # 19GB, 118k imágenes
    'http://images.cocodataset.org/zips/val2017.zip',  # 1GB, 5k imágenes
    'http://images.cocodataset.org/zips/test2017.zip'  # 7GB, 41k imágenes (opcional)
]
download(image_urls, dir=dataset_path / 'images', threads=3)
