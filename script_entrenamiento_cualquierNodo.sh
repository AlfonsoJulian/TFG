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

# python /mnt/homeGPU/azapata/TFG/TrainModels/COCO20/fineTunningDAclassic.py
# python /mnt/homeGPU/azapata/TFG/TrainModels/COCO20/fineTunningDADM.py
# python /mnt/homeGPU/azapata/TFG/TrainModels/COCO20/fineTunningHibrid.py
# python /mnt/homeGPU/azapata/TFG/TrainModels/COCO20/fineTunningOG.py

python /mnt/homeGPU/azapata/TFG/MetricasDeEntrenamientos/convertCSV.py