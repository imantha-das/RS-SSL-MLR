# Information on Source Code folder (src)

- data : Contains helper functions clean and organize data
    - organise_data2folders : Organises the raw data from zip file to folders. This is since the original data contains images with different channels (1,3,4). For easy preprocessing the data is organised into different folders based on the number of channels. i.e channel4 contains images with 4 channels. Note images with 4 chanels in the data are RGB images with a masking channel.
        - Usage : `python src/data/organise_data2folders -data_path <path to zip file>`
        - todo : this script needs to be tested to check if it works properly


    - clean_noisy_images : Cleans noisy images in a folder. There are large portion of noisy images containing no information in this dataset (black images). These are removed. Here we remove them by computing the channel mean and if less than a threshold they will be removed. A channel mean of 75 was picked as a threshold, hence any image below this value was archived and can be found in the data/interim folder.
        - Usage : `python src/data/clean_noisy_images -src_fold <source folder containing images> -dst_fold <destination folder where archive images (noisy) will be stored>`

    - helper_qgis_fns : helper functions to be used with qgis for easier loading of tif files from folders.

    - extract_img_window : To attain images for lat/lon points in the malaria dataset we geoprocess lat/lon values and extract corresponding image chips from larger image tiles. This script centers around a specified lat/lon value and extracts an nxn window image from the raw images (i.e found in data/interim/channel3). Note that all lat/lon values (in csv) will contain corresponding images and hence only a subset of these values will result in image chips being formed.
        - Usage : `python src/data/extract)img_window.py -geopts_p <path to csv with lat/lon values> -rsimg_p <path to image folder> -savefld_p <path to where extracted images are saved>`

    - add_img_with_pts2df : adds paths for extracted images with lat/lon points to dataframe. NOTE this script must be used after extracting relevent images with lat/lon points using "extract_img_window.py". Furthur its ideal to remove any noisy images using "clean_noisy_images.py". Once a soruce folder is specified all paths of images present in that folder will appended to the dataframe. So ensure all cleaning is carried out before running this script.
        - NOTE : In this dataset there were instances where multiple images corresponding to a single lat/lon value. This is due to different images being taken at different times in the same location. As we only need one of these images a simple strategy of selecting the most recent image (date) was employed.
        - Usage : `python src/data/add_img_with_pts2df.py -pts_p <lat/lon points csv path> -img_p <path to extracted images> -pts_dst_p <dataframe save path>`

- models : Contains code for SSL training and final downstral malaria classifier.

    - Pretrained Backbone Models are loaded from RSP repo - So please clone it inside your repository (repo root dir)
        - `git clone https://github.com/ViTAE-Transformer/RSP.git`
    
    - SSL model finetuning : Training ssl algorithm to extract image representaions (scripts for simsiam/byol/dino)
        - usage : `python src/models/byol_train.py -data_fold <path to data> -pretrain_weights_file <path to pretrained weights> -save_weights_fold <path to folder where model is saved>`
        - `dino_train.py` has an additional flag `-backbone_name` which takes the arguments resenet or swin-vit
        - ammend the config file (`src/models/config.py`) to change hyperparamers such as number of epochs.
    
    - Training downstream tasks : Training final malaria classifier 
        - usage : `python malaria_train.py -ssl_weights_p <path to trained ssl weights (i.e simsiam)> -save_weights_p <path to where classifiers weights will be saved> -mlr_csv_p <path to malaria dataset>`
        - ammend the malaria config file (`src/models/malaria_config.py`) to change hyperparameters such as number of epochs as well as features and target of the data.

- baseline_models : Contains baseline comparison models used to compare accuracy of malaria calssifier against.

- visualization : Scripts for plotting accuracy and losses for models
