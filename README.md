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
    ├── data           <- Scripts to download or generate data
    │   └──make_dataset.py
    │
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── config.py               <- Hyperparameter configurations for training models
    │   └── simsiam_train.py        <- Train SimiSiam algorithm on Regional RS Image Set
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