#!/bin/bash

#SBATCH --job-name DA_COCO                 # Nombre del proceso

#SBATCH --partition dios   # Cola para ejecutar

#SBATCH -w atenea                           

#SBATCH --gres=gpu:1                           # Numero de gpus a usar

export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /home/azapata/.conda/envs/ENVTFG_AUX

export TFHUB_CACHE_DIR=.

python /mnt/homeGPU/azapata/TFG/CreateDatasets/VOC/CreateDM.py
