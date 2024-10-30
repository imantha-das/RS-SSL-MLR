# Self Supervised finetuning Satellite Imagery for Malaria Prevelence Prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Utilising Self Supervised Learning (SSL) to identify Image Embeddings to improve Spatial Malarial Prevelence Prediction.

## Project Organization

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
* Copy the data (Satellite Image & Drone Images) inside data folder.
    * The code is designed to read satellite images and Drone images from seperate folders. Necessary adjustments need to be made in the scripts if all the data is in single folder.
    * i.e satellite images in : `data/processed/gee_sat/sen2a_c3_256x_pch`
    * drone images in : `data/processed/sshsph_drn/drn_c2c3_256x_pch`

* Copy pretrained weights from RSP repository to desired location
    * i.e pretrained weights : `model_weights/pretrain_weights/rsp-aid-resnet-50-e300.ckpt.pth`

* Create folder to store ssl weights : i.e `model_weights/ssl_weights`

* Create conda environment : `conda env create -f ssl-env.yml`
    * Due to version requirements required by lightning-bolts some algorithms may need alternative conda environments.
    * conda environments : ssl-env, rs-ssl-mlr (will be renamed to byol-ssl-env)

* Training SSL models
    * Update `src/ssl_models/config.py` with desired hyperparameter values.
    * i.e To train simsiam algorithm : `python src/ssl_models/simsiam_train.py -data_fold_drn <path to drone images> -data_fold_sat <path to satellite images> -pretrain_weights_file <path to pretrained weights file> -save_weights_fold <path to where finetuned weights are saved>`

* Conda Envirnmonets + Weights required by algorithms
    * BYOL : rs-ssl-mlr (Environment) , resnet weights form RSP (Backbone)
    * SimSiam : ssl-env (Environment) , resnet weights from RSP (Backbone)
    * Dinov1 : ssl-env (Environment) , resnet/swin-vit weights from RSP (Backbone)
    * Dinov2 : (still not implemented)
    * SimMIM : (still not implemented)
    * SatMAE : (Still not implemeted)

## Task 2 : Downstream Malaria Prediction

* For downstream malaria classifier training : 
    * `python src/downstream_models/malaria_train.py -ssl_weight_p <path to ssl model weights> -save_weight_p <folder path to save downstream model weights> -mlr_csv_p <path to malaria dataset>` 
    * Both conda environments are applicable

## Task : Data processing

* These functions are specific to the dataset used for this study. Code is available in `src/data_processing`.
* For a more detailed explaination of the scripts refer to "references" folder and scripts. 
    
    