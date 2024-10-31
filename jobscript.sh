#!/bin/bash

## Resources to be used
#PBS -l select=1:ngpus=4

## Walltime for the project to be run
#PBS -l walltime=01:00:00

## Use the AI cluster
#PBS -q ai

## Use project
#PBS -P 11003907

## Job name
# PBS -N byol_test

## Merge standard output and error from PBS script
#PBS -j oe

## Code starts from login nodes directory once logged into a compute node
cd ${PBS_O_WORKDIR}

## Load minmiforge3
module load miniforge3
conda activate ssl-env
## Run python script
python /home/users/nus/idg/workspace/RS-SSL-MLR/src/models/simsiam_train.py -data_fold_drn  /home/users/nus/idg/scratch/data/processed/sshsph_drn/c3_256x_pch -data_fold_sat /home/users/nus/idg/scratch/data/interim/gee_sat/sen2a_c3_256x_clp0.3_uint8_ucln_pch -pretrain_weight_file /home/users/nus/idg/workspace/RS-SSL-MLR/model_weights/pretrain_weights/rsp-aid-resnet-50-e300-ckpt.pth -save_weights_fold /home/users/nus/idg/workspace/RS-SSL-MLR/model_weights/ssl_weights



