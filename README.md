# Self Supervised finetuning of Satellite Imagery for Downstream Malaria Prevelence Prediction

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
* Setting up foundation model to be used as backbone for SSL finetuning
    * `mkdir src/ssl_models/foundation_models`
    * Clone the RSP Repository (Pretrained Model), `git clone https://github.com/ViTAE-Transformer/RSP.git` into folder `src/ssl_models/foundation_models`
    * Copy pretrained weights (Both resnet & swin_vit_t) from RSP repository (https://github.com/ViTAE-Transformer/RSP) to a desired location
        * i.e pretrained weights : `model_weights/pretrain_weights/rsp-aid-resnet-50-e300.ckpt.pth`
* Copy the data (Satellite Image & Drone Images) into the `data` folder.
    * Model training (i.e `simsiam_train.py`) code is designed to read satellite images and drone images located in seperate folders. Necessary adjustments need to be made in order to run data located within a single folder.
    * i.e satellite images dir : `data/processed/gee_sat/sen2a_c3_256x_pch`
    * drone images dir : `data/processed/sshsph_drn/drn_c2c3_256x_pch`

* Create a folder to store finetuned ssl weights : i.e `model_weights/ssl_weights`

* Create conda environment : `conda env create -f ssl-env.yml`

* Finetunning SSL models
    * i.e To train simsiam algorithm : `python src/ssl_models/ssl_finetune.py -ssl_model simsiam -backbone resnet -data_fold_drn data/processed/drn_c3_256x_pch -data_fold_sat sen2a_c2_256x_clp0.3uint8_full_pch -pretrain_weights_fold model_weights/pretrain_weights -save_weights_fold model_weights/ssl_weights`
    * Arguments 
        * `-ssl_model` : ssl model (options : `simsiam`,`byol`,`dinov1`,`mae`,`satmae`)
        * `-backbone` : backbone name (options : `resnet`, `swin-vit`,`vit`); Note that some backbones are not implmented for all ssl_models.
        * `-data_fold_drn` : path to folder containing drone images
        * `-data_fold_sat` : path to folder containing satellite images
        * `-pretrain_weight_fold` : path to pretrain weights folder (weights file automatically selected based on backbone name)
        * `-save_weights_fold` : path to folder where finetuned weights will be saved
    * Optional argument
        * `-epochs` : No of epochs (default : 20)
        * `-eff_batch_size` : Effective batch size (This is total batch size run across all gpu's, dataloader batchsize = eff_batch_size / devices * nodes) (default : 512)
        * `-lr` : learning rate (By default set to None which will use a learning rate scheduler, to run a constant learning rate across all epochs, set to a desired value) (default : None)
        * `-input_size` : image input size (default : 256)
        * `-devices` : no of gpus (default : 4)
        * `-nodes` : no of computing nodes (default : 1)
        * `-precision` : torch precision (default : 32)
        * `-dataloader_workers` : number of workers in data loader (default : 16)
        * `-save_freq` : model weight save frequency (default : 1)
        * `-save_all_weights` : Optional flag to include only both optimizer and model weights require saving (This isnt useful here is useful during pretraining where to restart from model checkpoint requires optimizer weights)
    * To train on single dataset, just provide only the path to required dataset, the other will default to None.
    * SSL algorithm specific hyperparameters can found at `src/ssl_models/ssl_config.yml`

## Task 2 : Downstream task (Malaria prediction) training

* For downstream malaria classifier training : 
    * i.e To train downstream malaria classifier : `python src/downstream_models/malaria_train.py -mlr_data_file "data/processed/sshsph_mlr/mlr_nomiss_vardrop_train_v2.csv" -ssl_weight_root_fold model_weights/ssl_weights/dino-is256-effbs256-ep10-bbResnet-dsDrnSen2a-clClUcl-nmTTDrnSatNM -version 0 -weight_at_epoch 95` 
    * Arguments
        * `mlr_csv` : processed malaria dataset containing image paths
        * `ssl_weight_root` : root folder containing ssl finetuned model checkpoints, hyperparams, metrics etc.
        * `version` : version folder number inside ssl weights root folder
        * `weight_every_n_epochs` : Trains downstream model using epochs saved at spacing n (i.e if a spacing of 5 is used, ssl_weights corresponding to 5,10,15,... epochs will be used) 
        * `weight_at_epoch` : To train a downstream model using a weight saved at a specific epoch (i.e weights saved at 95th epoch)
    * Model results will be saved inside `ssl_weight_root/downstream`

## Optional Task : Pretraining backbone model

* To train i.e MAE on dataset such as Million-Aid

* Arguments are essentially the same as finetuning except for the follows
    * No argument name `-pretrain_weights_fold`
    * `-ckpt_path` : path to model weights if contuning from a checkpoint (i.e second iteration of training)

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
    qsub jobscrpt.sh
    ```