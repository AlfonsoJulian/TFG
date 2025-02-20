#!/bin/bash

#SBATCH --job-name YOLO_COCO_DA                 # Nombre del proceso

#SBATCH --partition dios   # Cola para ejecutar

#SBATCH --gres=gpu:1                           # Numero de gpus a usar
#SBATCH -w dionisio                           

#SBATCH --mem=30GB #Memoria a reservar

export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/azapata/ENVTFG

export TFHUB_CACHE_DIR=.

python DA_ModelosDiffusionSegunSize.py

