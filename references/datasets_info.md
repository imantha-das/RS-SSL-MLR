# Information on Datasets

The following file contains information raw / interim and processed datasets. 

# Raw Folder 
Contains the  raw unprocessed datasets

# Interim Folder
Contains the intermediate processed data. This includes any removed noisy images, unreadable files and irrelevent files
- channel1, channel3, channel4 : Unprocessed data comes in various sizes as well as different number of channels. They 
were distributed to different folders to allow easier preprocessing.
    - channel1 : Grayscale images not used 
    - channel3 : These images were Patched (256x256 etc.) and then used for finetuning SSL algorithms. Patched images can be found in "processed" folder.
    - channel4 : These images were Geoprocessed to extract patches relative to lat/lon vales from malaria data. Patched images from be found in the "processed" folder
        - Note channel4 images are just RGB images with a masking channel.
- channel3_256x256p_arch : Noisy and defected image files removed from channel3_256x256p (found in processed folder) 
- channel3_256x256pts_arch : Remaining noisy images from the extracted images based on lat/lon values from malaria dataset. These images were orignall extracted from channel3. Refer to extract_image_window.py or its reference documentation for more verbose explanation. 
- channel4_256x256pts_arch : Same as above except channel4 images were used as the base images for extraction.
- tiles : Larger tiles that were found in Raw data. These were never used and can be used as a final testing set or for further training if required.

# Processed Folder 
Contains final processed datasets used for model training
- channel3_96x96p : Contains images chips of dimension 96x96 patched from Channel3 images (in interim folder). 
- channel3_256x256p : Contains images chips of dimension 256x256 patched from Channel3 images (in interim folder). Used for finetuning SSL algorithms.
- channel3_512x512p : Contains images chips of dimension 512x512 patched from Channel3 images (in interim folder)
- channel3_256x256pts : Contains images patches with respect to lat/lon values from malaria dataset. These were used for final training
of malaria dataset.
