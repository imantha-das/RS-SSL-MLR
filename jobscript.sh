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
python /home/users/nus/idg/workspace/RS-SSL-MLR/src/ssl_models/ssl_train.py \
    -ssl_model dino \
    -backbone swin-vit \
    -epochs 100 \
    -eff_batch_size 512 \
    -devices 8 \
    -nodes 2 \
    -precision 16 \
    -data_fold_drn /home/users/nus/idg/scratch/data/processed/sshsph_drn/c3_256x_pch \
    -data_fold_sat /home/users/nus/idg/scratch/data/interim/gee_sat/sen2a_c3_256x_clp0.3_uint8_ucln_pch \
    -pretrain_weights_fold /home/users/nus/idg/scratch/models_weights/pretrain_weights \
    -save_weights_fold /home/users/nus/idg/scratch/models_weights/ssl_weights \

