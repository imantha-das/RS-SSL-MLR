# ==============================================================================
# Desc : Baseline Model without/with (some) Geospatial Feature to check the 
# performance for malaria prediction
# We will test with available features if Random Fores / Logistic Regression 
# models performance.
# Abe to get an accuracy of ~85% with RF/LR
# ==============================================================================

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from typing import List
from termcolor import colored
import models.malaria_config as malaria_config

def remove_missing(df:pd.DataFrame, feat:list, target:list):
    Xy = df.filter(feat + target)
    Xy.dropna(inplace = True) #drop missing values
    X = Xy.filter(feat)
    y = Xy.filter(target)
    return X, y

def train_test_splits(X, y, test_size = 0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y , shuffle = True, test_size=test_size)
    return X_train, X_test, y_train, y_test

def one_hot_encode(df:pd.DataFrame,cat_names:List[str]):
    column_trans = make_column_transformer(
        (OneHotEncoder(), cat_names), 
        remainder = "passthrough"
    )
    X_train_enc = column_trans.fit_transform(df)

    return X_train_enc, column_trans


if __name__ == "__main__":
    df = pd.read_csv("data/processed/xss_luminex_sero_dem_preds_pts_flt.csv")
    
    numeric_feat = malaria_config.numeric_feat
    cat_feat = malaria_config.cat_feat
    geo_feat = malaria_config.geo_feat #! we wont be using this !
    target = malaria_config.target

    # Preprocess : Remove missing values
    all_feat = numeric_feat + cat_feat # concatenate lists
    X, y = remove_missing(df, all_feat, target) # Using all features reduces values from 10118 -> 5218
    # Train / Test Splits
    X_train, X_test, y_train, y_test = train_test_splits(X, y)
    #print(colored(f"X_train : {X_train.shape}", "green"))
    # Preprocess : OHE
    X_train_enc, transformer = one_hot_encode(X_train, cat_feat)
    X_test_enc = transformer.transform(X_test)

    # Logistic Regression
    print(colored(f"Logistic Regression results ...", "green"))
    lr = LogisticRegression()
    lr.fit(X_train_enc, y_train)
    print(f"Train Score : {lr.score(X_train_enc, y_train)} , Test score : {lr.score(X_test_enc, y_test)}")

    print(colored(f"Random Forest results ...", "green"))
    rf_hyper = {"max_depth" : 5}
    rf = RandomForestClassifier(**rf_hyper)
    rf.fit(X_train_enc, y_train)
    print(f"Train Score : {rf.score(X_train_enc, y_train)} , Test score : {rf.score(X_test_enc, y_test)}")

