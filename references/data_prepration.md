# Data Prepration and Cleaning

Note the data processing guideline listed below is specific to the dataset we use for this study.

**Data Processing High Level Overview for SSL training**

* We use two datasets containing Sentinel2A data from the period 2015-10-14 to 2016-10-24 as well as drone images that were taken in2014.
* Images are next patched to a size of 256x256 pixel chips as larger RS image tiles are too computational expensive to run SSL algorithms.
* Images containing no pixel information (as result of masking) are removed from the dataset.
    * (Optional) Remove images containing redudant features such as the ocean as Sentinel2a dataset contains large portions of the ocean. 
    * This requires manual maksing and extracting using QGIS. (not completed)
* Resulting Images are moved to the "processed" folder for SSL.
* As Sentinel Images are Surface Relectance values, they are **scaled by 10000** as mentioned in GEE documentaion. The resulting values are **clipped to a range of 0-0.3** since reflectance values for majority of earth object such as vegeataion lie with this range. Next these values are multiplied by 255 and saved as uint8 format. The reason for this is two fold, 1. drone images are in a same scale, 2. Lightly SSL package requires images to be in unint8 format. (Note this step is only applicable for Sentinel images)

**Data processing High level Overview for Downstream training**

* Instead of pactching each RS image tile, we require extracting and image chip (i.e 256x256) centering around lat/lon values where individual live (house) or visited from the malaria dataset.
* Clipping an scaling as mentioned above are applicable for processing Sentinel images.

All scripts required for data prepration is available in `src/data_processing`.

Procedure of using scripts for data cleaning.
- To train SSL algorithms
    - `organise_data2folders.py`: To organise raw data for furthur processing
    - `patch_images.py` : Patch larger image tiles to smaller chips
    - `clean_noisy_images.py` : Remove any noisy image chips such as ones where there is a large portion of masked pixel no information (little if there are portion of te image).
    - `convert2uint8.py` : For scaling & clipping Sentinel Images (This script is only relevant to sentinel images)
- To train malaria classifier
    - `ext_or_msk_ROI` : Extract image chips centering around lat/lon points where individual lives or visited.
    - Use `convert2uint8.py` as mentioned above for scaling ans clipping.
    - `get_imgp_for_geopts.py` : To construct a dataframe containing lat/lon values and respective images chips extracted using `ext_or_msk_ROI.py`
    - `preprocess_tables_for_downstrean_tasks.py` : Creates a dataframe containing malaria specific features as well as respective paths to images for downstream prediction tasks.

Todo : Create a script called `make_datasets.py` to sequentially carry out all these steps