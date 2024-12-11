

from torch.utils.data import Dataset, DataLoader
from Lightly.data import LightlyDataset

from typing import Union

def clean_image_dataset():
    """
    Construct an image dataset by moving all .tif files
    - Constructs data folder to store all mosiacs data
    - Within the data folder a folder named "train" to store all .tif images and "archive" to store all other data.
    """
    # Construct data folder
    if not os.path.exists("data"):
        os.mkdir("data")
    # Construct train folder
    if not os.path.exists("data/train"):
        os.mkdir("data/train")
    # construct archive folder
    if not os.path.exists("data/archive"):
        os.mkdir("data/archive")

    for f in glob("*"):
        if f.endswith(".tif"):
            shutil.move(src = f, dst = "data/train")
        if f.endswith(".enp") or f.endswith(".ovr") or f.endswith("pdf") or f.endswith(".prj") or f.endswith(".tfw") or f.endswith("tiles") or f.endswith(".xml") or f.endswith(".zip"):
            shutil.move(src = f, dst = "data/archive")

def remove_images(paths:list):
    """
    Removes Images that are ...
        - Not of type ndarray (incomplete info, in this case grayscale)
        - Images that donot have 3 channels
    Inputs
        - list of paths to images
    """
    nd_array_cnt = 0 # 
    other_cnt = 0
    img3_cnt = 0 # 3D image count
    img3not_cnt = 0 # anything that isnt 3D image count
    for p in paths:
        img = cv2.imread(p)
        # Check whether data is of type ndarray (as there are nontype data in the dataset)
        if isinstance(img, np.ndarray):
            nd_array_cnt += 1
            # Check if the data is 3 Dimensional
            if img.shape[2] == 3:
                img3_cnt += 1
            # Move any non 3D images to archive
            else:
                img3not_cnt += 1
                print(f"Not an 3D img : {p}")
                shutil.move(src = p , dst = "data/archive")
        # Check if the data is of type 'NoneType' 
        else:
            print(f"None type images : {p}")
            print("Moving to archive ...")
            shutil.move(src = p , dst = "data/archive")
            other_cnt += 1

    print(f"Clean image count : {nd_array_cnt}")
    print(f"Damaged images : {other_cnt}")
    
# --------------------- Computes Mean & Std for Datasets --------------------- #

def get_img_stats(data:Union[Dataset, LightlyDataset, DataLoader]):
    """
    Computes Image statistics. Able to handle multiple dataset/dataloader functions.
    Note the dataloader must return a torch tensor and not an image
    Inputs
    ------
        data : PyTorch DataLoader or Dataset Class
    Notes
    -----
    The Dataset class can return tensor of different shapes (image_sizes different in million-aid)
    """
    c_sum, c_sq_sum, total_pixels = 0,0,0
    # Function to compute total pixels and sum & sum square
    def compute_sq_sum_pixels(X):
        """Function to compute total pixels, sum of all pixels and sum square"""
        # To keep track of the global varaibles
        nonlocal c_sum, c_sq_sum, total_pixels
        # handles dataset classes where there is not batch dimensions
        if X.dim() == 3:
            c,w,h = X.shape # (c,w,h)
            num_pixels = w * h
            c_sum += torch.sum(X, dim = [1,2]) # ignore channel dim when summing
            c_sq_sum += torch.sum(X ** 2, dim = [1,2])
        # handles dataloader which have batch dimension
        elif X.dim() == 4:
            b,c,w,h = X.shape # (b,c,w,h)
            num_pixels =  b * w * h 
            c_sum += torch.sum(X, dim = [0,2,3]) # Sum across channels, data.shape = (b,c,w,h)
            c_sq_sum += torch.sum(X ** 2, dim = [0,2,3])
        else:
            raise(TypeError("Incorrect number of dimensions in data"))
        
        total_pixels += num_pixels
        return total_pixels, c_sum, c_sq_sum

    # The number of variables spit out from dataset may change based on what function we use
    # i.e LightlyDataset spits out 3 varaibles (X,y,paths) while torchvision image folder only spits out 2 
    # variables (X,y)
    data_ = next(iter(data))
    # Handle 2 outputs from dataset/dataloader (X,y)
    if len(data_) == 2:
        for X,_ in tqdm(data):
            assert isinstance(X, torch.Tensor), "X not a torch tensor"
            total_pixels, c_sum, c_sq_sum = compute_sq_sum_pixels(X)
        mean = c_sum / total_pixels
        std = ((c_sq_sum/total_pixels) - (mean**2))**0.5
    # Handle 3 outputs from dataset/dataloader (X,y,p)
    elif len(data_) == 3:
        for X,_,_ in tqdm(data):
            assert isinstance(X, torch.Tensor), "X not a torch tensor"
            total_pixels, c_sum, c_sq_sum = compute_sq_sum_pixels(X)
        mean = c_sum / total_pixels
        std = ((c_sq_sum/total_pixels) - (mean**2))**0.5

    return mean,std

# ---------------------------- Get Max's and Mins ---------------------------- #

def get_maxmin_stats(dataset:Union[Dataset,LightlyDataset], dataset_args:dict, bs:int = 512, save_prefix:str = "",save_suffix:str = "")->None:
    """
    Computes Max Min in Dataset : A lot of the data falls above or normalised data
    Inputs
        - dataset : Dataset you wish to find Max and Min for 
        - dataset_args : arguments for dataset
        - no_augmented_views : pretraining datasets all return 2 values while finetuning datasets return 4
    """
    
    if isinstance(dataset, LightlyDataset):
        dataloader = DataLoader(dataset, batch_size = bs)
        mins_all_imgs_x1 = [] ; mins_all_imgs_x2 = []
        maxs_all_imgs_x1 = [] ; maxs_all_imgs_x2 = []
        for X, _, _ in tqdm(dataloader):
            X1,X2 = X
            x1min = X1.amin(dim = (1,2,3)) ; x2min = X2.amin(dim = (1,2,3))
            x1max = X1.amax(dim = (1,2,3)) ; x2max = X2.amax(dim = (1,2,3))
            mins_all_imgs_x1.extend(x1min) ; mins_all_imgs_x2.extend(x2min)
            maxs_all_imgs_x1.extend(x1max) ; maxs_all_imgs_x2.extend(x2max)

        mins_all_imgs_x1 = list(map(lambda x : x.item(), mins_all_imgs_x1)) ; mins_all_imgs_x2 = list(map(lambda x : x.item(), mins_all_imgs_x2))
        maxs_all_imgs_x1 = list(map(lambda x : x.item(), maxs_all_imgs_x1)) ; maxs_all_imgs_x2 = list(map(lambda x : x.item(), maxs_all_imgs_x2))
        pmin_x1 = px.histogram(mins_all_imgs_x1, title = "Distribution of Mins per Image", marginal = "box") ; pmin_x2 = px.histogram(mins_all_imgs_x2, title = "Distribution of Mins per Image", marginal = "box")
        pmax_x1 = px.histogram(maxs_all_imgs_x1, title = "Distribution of Maxs per Image", marginal = "box") ; pmax_x2 = px.histogram(maxs_all_imgs_x2, title = "Distribution of Maxs per Image", marginal = "box")
        pmin_x1.update_layout(showlegend = False) ; pmin_x2.update_layout(showlegend = False)
        pmax_x1.update_layout(showlegend = False) ; pmax_x2.update_layout(showlegend = False)
        pmin_x1.write_image(f"tmp/mins_dist_{save_suffix}_x1.png") ; pmin_x2.write_image(f"tmp/mins_dist_{save_suffix}_x2.png")
        pmax_x1.write_image(f"tmp/maxs_dist_{save_suffix}_x1.png") ; pmax_x2.write_image(f"tmp/maxs_dist_{save_suffix}_x2.png")

    else:
        dataloader = DataLoader(dataset(**dataset_args), batch_size=bs)
        mins_all_imgs = []
        maxs_all_imgs = []
        for X, _ in tqdm(dataloader):
            xmin = X.amin(dim = (1,2,3)) # (b,)
            xmax = X.amax(dim = (1,2,3)) #(b,)
            mins_all_imgs.extend(xmin)
            maxs_all_imgs.extend(xmax)
    
        mins_all_imgs = list(map(lambda x : x.item(), mins_all_imgs))
        maxs_all_imgs = list(map(lambda x : x.item(), maxs_all_imgs))
        pmin = px.histogram(mins_all_imgs, title = "Distribution of Mins per Image", marginal = "box")
        pmax = px.histogram(maxs_all_imgs, title = "Distribution of Maxs per Image", marginal = "box")
        pmin.update_layout(showlegend = False)
        pmax.update_layout(showlegend = False)
        pmin.write_image(f"tmp/{save_prefix}_mins_dist_{save_suffix}.png")
        pmax.write_image(f"tmp/{save_prefix}_maxs_dist_{save_suffix}.png")