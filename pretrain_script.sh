#!/bin/bash

## Resources to be used
#PBS -l select=1:ngpus=8

## Walltime for the project to be run
#PBS -l walltime=24:00:00

## Use the AI cluster
#PBS -q ai

## Use project
#PBS -P xxxxxxxx

## Job name
# PBS -N dino_512

## Merge standard output and error from PBS script
#PBS -j oe

## Code starts from login nodes directory once logged into a compute node
cd ${PBS_O_WORKDIR}

## Load minmiforge3
module load miniforge3
conda activate ssl-env
## Run python script
python /home/users/nus/idg/workspace/RS-SSL-MLR/src/ssl_models/ssl_pretrain.py \
    -ssl_model mae \
    -backbone vit \
    -epochs 300 \
    -eff_batch_size 1024 \
    -input_size 256 \
    -devices 8 \
    -nodes 1 \
    -precision 32 \
    -data_fold_sat /home/users/nus/idg/scratch/data/processed/million_aid/test \
    -save_weights_fold /home/users/nus/idg/scratch/models_weights/pretrain_weight_fold \
    -dataloader_workers 16 \
    -save_freq 20