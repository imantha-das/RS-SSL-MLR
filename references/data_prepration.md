# Data Prepration and Cleaning

All scripts required for data prepration is available in `src/data`.

Procedure of using scripts are as follows
- To train SSL algorithms
    - Use "organise_data2folders.py" first on raw data to seperate them based on number of channels
    - Use "patch_images.py" to patch larger image tiles to smaller chips
    - Use "clean_noisy_images.py" to remove any noisy image chips
- To train malaria classifier
    - Use "extract_img_window.py" on the original images (resulting images from organise_data2folders.py) to extract portion of the images corresponding to the lat/lon values.
    - Use "clean_noisy_images.py" to remove any noisy images in these folders
    - Use "add_img_with_pts2df.py" to add the image paths to 

For a more detail explation what each script does and any assumptions taken refer to utils section in `references/src_info`.

Todo : Create a script called `make_datasets.py` to sequentially carry out all these steps