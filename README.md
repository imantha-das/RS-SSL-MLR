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
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
│
├── models             <- Saved model weights from SSL training and open source pretrained model weights
│
├── external           <- External models used to compare the performance 
│
│
├── references         <- Explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
│
└── src
    │
    ├── data                            <- Scripts to download or generate data
    │   └──add_img_with_pts2df.py       <- add image paths to dataframe if image present for lat/lon coordinate
    │   └──clean_noisy_images.py        <- Remove any noisy images (images with no information) from dataset
    │   └──extract_img_window.py        <- Extract section of image (window) centering a given lat/lon coordinate 
    │   └──helper_qgis_fns.py           <- Functions to make loading images easier in QGIS.
    │   └──organise_data2folders.py     <- Unzip raw images and seperate them to differnt folder based on number of channels.
    │   └──patch_images.py              <- Patch large image tiles to smaller image chips to make training possible.   
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── config.py               <- Hyperparameter configurations for training models
    │   └── simsiam_train.py        <- Train SimiSiam algorithm on Regional RS Image Set
    │   └── byol_train.py           <- Train BYOL algorithm on Regional RS Image set
    │   ├── malaria_train.py        <- Train final classifier on Malaria Dataset
    │   └── utils.py                <- Helper functions for model training
    │
    └── 
```

--------

# How to use

* Clone repository : `git clone https://github.com/imantha-das/RS-SSL-MLR.git`
* Add RSP Repository (Pretrained Model) : `git clone https://github.com/ViTAE-Transformer/RSP.git`
* Example, To finetune SSL algorithm on specific data : `python src/models/simsiam_train -dfold <path to data>`