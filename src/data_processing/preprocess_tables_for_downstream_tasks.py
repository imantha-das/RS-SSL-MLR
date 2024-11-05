# ==============================================================================
# This Script is used to preprocess Malaria Data for Downstream Prediction Tasks
# This involves joining tables, removing unnecssary columns and removing
# missing data.
# ==============================================================================
import numpy as np
import pandas as pd
import os
import ast

from typing import List

def join_dfs_filter_cols(df_feats:pd.DataFrame, df_targets:pd.DataFrame, feat_ignore:List[str],target_ignore:List[str]):
    """Features and Target comes in two dataframes. This function join these dataframes and select
    necessary columns"""
    # Select required Columns for targets
    target_cols_selected = df_targets.columns.values[[nm not in target_ignore for nm in df_targets.columns.values]]
    df_targets = df_targets.filter(target_cols_selected)
    # Select required columns for features
    feat_cols_selected = df_feats.columns.values[[nm not in feat_ignore for nm in df_feats.columns.values]]
    df_feats = df_feats.filter(feat_cols_selected)
    # Join both DataFrames
    df = df_feats.merge(df_targets, on = "Sample")
    
    return df

def join_dfs_imgp(df:pd.DataFrame, df_imgp:pd.DataFrame, imgp_ignore:List[str]):
    # Select columsn in df_imgp that are useful.
    imgp_selected = df_imgp.columns.values[[nm not in imgp_ignore for nm in df_imgp.columns.values]]
    df_imgp = df_imgp.filter(imgp_selected)
    # Merge dataframes on "sample"
    df = df.merge(df_imgp, on = "Sample")
    return df

def prioratise_and_select_imgp(df:pd.DataFrame, drn_repo_p:str, sat_repo_p:str):
    """This function looks at image paths in drn and sentinel datasets, and prioratises on using drone
    images if images are not present, it will select the first sentinel image if available"""
    
    selected_imgp = []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        drn_img_list = ast.literal_eval(row["drn_c3c4_256x_ext"])
        sat_img_list = ast.literal_eval(row["sen2a_c3_256x_clp0.3uint8_full_ext"])
        if len(drn_img_list) > 0:
            # Select the first image from drone images - This is sorted based on date but not on image suffix if two images are there for the same date
            selected_imgp.append(os.path.join(drn_repo_p,drn_img_list[0]))
        elif len(sat_img_list) > 0:
            # SImilarly select firsr image from sen2a images - This is not sorted, refer to "get_imgp_for_geopts.py"
            selected_imgp.append(os.path.join(sat_repo_p, sat_img_list[0]))
        else:
            selected_imgp.append(np.nan)
    
    # Drop these columns as they are string values containg a list as such they have non nan values when there are some empty lists
    df.drop(["drn_c3c4_256x_ext", "sen2a_c3_256x_clp0.3uint8_full_ext"], axis = 1, inplace = True)

    df["selected_imgp"] = selected_imgp
    # There are a number of columns with missing values which can be imputed but since we dont know
    # much about the variables just drop them
    df.dropna(inplace = True)

    return df


if __name__ == "__main__":

    # ------------ Join Feature + Target dataframes and filter columsn ----------- #
    df_feat = pd.read_csv("data/raw/sshsph_mlr/XSS_Luminex_Sero_Dem.csv")
    df_targets = pd.read_csv("data/raw/sshsph_mlr/pred_xss_pk_class.csv", index_col = "Unnamed: 0")
    feat_cols_ignore = [
        # Variables I think we should be dropping
        "mergeID", "hadMalaria","houseID","individualID","GPS_X", "GPS_Y",
        # Variables droped to maximize non-null values
        "PvRII","PfSEA", "S..mansoni.GST.control"
    ]
    target_cols_ignore = ["PkSSP2","PkSera3.Ag2","pred_pk"]
    df = join_dfs_filter_cols(df_feat, df_targets, feat_cols_ignore, target_cols_ignore)

    # ---------- Join filtered df with img paths and remove missing data --------- #
    df_imgp = pd.read_csv("data/interim/sshsph_mlr/mlr_geopts_imgp_v2.csv")
    imgp_cols_ignore = ["GPS_X","GPS_Y"]
    df = join_dfs_imgp(df, df_imgp, imgp_cols_ignore)
    
    # ----------------------- Prioratise and Select Images ----------------------- #
    df = prioratise_and_select_imgp(
        df, 
        drn_repo_p="data/processed/sshsph_drn/drn_c3c4_256x_ext",
        sat_repo_p = "data/processed/gee_sat/sen2a_c3_256x_clp0.3uint8_full_ext"
    )
    print(df.info())
    
    #! There are only 700ish entries. Check which cols to drop to maximize dataset
    df.to_csv("data/processed/sshsph_mlr/mlr_data_nomiss_some_var_dropped_v2.csv", index= False)
