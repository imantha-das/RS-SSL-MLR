# Self Supervised finetuning Satellite Imagery for Malaria Prevelence Prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Utilising Self Supervised Learning (SSL) to identify Image Embeddings to improve Spatial Malarial Prevelence Prediction.

# Project Organization

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── raw            <- The original, immutable data dump.
│   └── processed      <- Final processed data 
│        ├── gee_sat         <- Sentinel Images (interim/raw has similar folder structure)
│        └── sshsph_drn      <- Drone Images (interim/raw has similar folder structure) 
│
│
├── models_weights     <- Folder containing pretrained & SSL & downstream task model weights. 
│
├── external           <- External models used to compare the performance 
│   ├── pretrain_weights     <- Opensource pretrained model weights
│   ├── ssl_weights          <- SSL weights
│   └── downstream_weights   <- Downstream model (Malaria classifier) weights
│
├── references         <- Explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
│
└── src
    │
    ├── data_processing          <- Scripts for clearning & processing data
    │
    ├── baseline_models          <- Scripts for SSL model performance comparison
    │   
    ├── ssl_models               <- Scripts for finetuning regional datasets
    │   └── foundation_models        <- External opensource pretrained models 
    │  
    ├── downstream_models        <- Models for training downstream task (Malaria classification)
    │
    └── visualization            <- Scripts for data visualization
```

--------

# How to use
## Task 1 : Finetuning pretrained model using SSL Algorithms
* Clone repository : `git clone https://github.com/imantha-das/RS-SSL-MLR.git`
* Clone the RSP Repository (Pretrained Model), `git clone https://github.com/ViTAE-Transformer/RSP.git` into folder `src/ssl_models/foundation_models`
* Copy the data (Satellite Image & Drone Images) into the `data` folder.
    * Model training (i.e `simsiam_train.py`) code is designed to read satellite images and drone images located in seperate folders. Necessary adjustments need to be made in order to run data located within a single folder.
    * i.e satellite images dir : `data/processed/gee_sat/sen2a_c3_256x_pch`
    * drone images dir : `data/processed/sshsph_drn/drn_c2c3_256x_pch`

* Copy pretrained weights from RSP repository (https://github.com/ViTAE-Transformer/RSP) to a desired location
    * i.e pretrained weights : `model_weights/pretrain_weights/rsp-aid-resnet-50-e300.ckpt.pth`

* Create a folder to store ssl weights : i.e `model_weights/ssl_weights`

* Create conda environment : `conda env create -f ssl-env.yml`
    * Due to version requirements required by lightning-bolts some algorithms may need alternative conda environments.
    * conda environments : ssl-env,  ssl-byol-env

* Training SSL models
    * Update `src/ssl_models/config.py` with required hyperparameter values.
    * i.e To train simsiam algorithm : `python src/ssl_models/simsiam_train.py -data_fold_drn <path to drone images> -data_fold_sat <path to satellite images> -pretrain_weights_file <path to pretrained weights file> -save_weights_fold <path to where finetuned weights are saved>`

* Conda Environment + Pretrained-weights required by algorithms
    * BYOL : ssl-byol-env (Environment) , resnet weights form RSP (Backbone)
    * SimSiam : ssl-env (Environment) , resnet weights from RSP (Backbone)
    * Dinov1 : ssl-env (Environment) , resnet/swin-vit weights from RSP (Backbone)
    * Dinov2 : (still not implemented)
    * SimMIM : (still not implemented)
    * SatMAE : (Still not implemeted)

## Task 2 : Downstream task (Malaria prediction) training

* For downstream malaria classifier training : 
    * `python src/downstream_models/malaria_train.py -ssl_weight_p <path to ssl model weights> -save_weight_p <folder path to save downstream model weights> -mlr_csv_p <path to malaria dataset>` 
    * Both conda environments are applicable.

## Data processing

* These functions are specific to the dataset used for this study, Code available at `src/data_processing`.
* For a more detailed explaination of the scripts usef, refer to "references" folder. 
    
## Running on HPC

* To run on NSCC (National Supercomputer Singapore) in *interactive mode*, example detailed below refer to training SimSiam model. 
    ```
    qsub -I -l select=1:ngpus=8 -l walltime=24:00:00 -q ai -P xxxxx

    cd RS-SSL-MLR

    conda activate ssl-env

    python src/ssl_models/simsiam_train.py -data_fold_drn ../scratch/data/processed/sshsph_drn/drn_c3_256x_pch -data_fold_sat ../scratch/data/processed/gee_sat/sen2a_c3_256x_pch -pretrain_weight_file model_weights/pretrain_weights/rsp-aid-resnet-50-e300-ckpt.pth -save_weight_fold model_weights/ssl_weights
    ```
    * `qsub` details
        * `I` : interactive mode
        * `select=1:ngpus=8` : Single node, 8 GPUS.
        * `walltime=24:00:00` : Run for 24 hours
        * `q` : queue
        * `P` : project Id

* To run using *PBS script*
    ```
    chmod a+x jobscript.sh`
    ./jobscrpt.sh
    ```