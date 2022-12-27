from typing import List, Set, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
import os
from utils.utils import cprint
import torch

def read_metadata(path: str) -> pd.DataFrame:
    result = pd.read_csv(path)

    if not isinstance(result, pd.DataFrame):
        raise TypeError("error")

    return result

def save_metadata(metadata: pd.DataFrame, path: str):
    metadata.to_csv(path, index=False)

def save_image(image: np.ndarray, path: str):
    np.save(path, image)

def filter_metadata_by_multi_cell_image_names(metadata: pd.DataFrame, multi_cell_image_names: List[str]) -> pd.DataFrame:
    """ Also corresponds to the subfolder name. """
    result = metadata[np.isin(metadata["Multi_Cell_Image_Name"], multi_cell_image_names)]

    if not isinstance(result, pd.DataFrame):
        raise TypeError("error")

    return result

def get_relative_image_path(metadata_row: pd.Series) -> str:
    """ returns 'singh_cp_pipeline_singlecell_images'/subfolder/image_name """
    multi_cell_image_name = metadata_row["Multi_Cell_Image_Name"]
    single_cell_image_id = metadata_row["Single_Cell_Image_Id"]
    result = "singh_cp_pipeline_singlecell_images/" + multi_cell_image_name + "/" + multi_cell_image_name + "_" + str(single_cell_image_id) + ".npy"
    return result

def get_relative_image_paths(metadata: pd.DataFrame) -> List[str]:
    """ 
    returns [..., 'singh_cp_pipeline_singlecell_images'/subfolder/image_name] 
    """
    result = []

    for _, row in metadata.iterrows():
        path = get_relative_image_path(row)
        result.append(path)
        
    return result

def get_relative_image_folders(metadata: pd.DataFrame) -> List[str]:
    """ returns 'singh_cp_pipeline_singlecell_images'/subfolder/"""
    result = []

    multi_cell_image_name = metadata["Multi_Cell_Image_Name"]
    single_cell_image_id = metadata["Single_Cell_Image_Id"]

    if not isinstance(multi_cell_image_name, pd.Series):
        raise TypeError("error")

    if not isinstance(single_cell_image_id, pd.Series):
        raise TypeError("error")
    
    for multi_cell_name, image_id in zip(multi_cell_image_name, single_cell_image_id):
        folder = "singh_cp_pipeline_singlecell_images/" + multi_cell_name
        result.append(folder)
        
    return result

def load_images(paths: List[str], verbose: bool = False, log_every: int = 10_000, logfile=None):
    image_0 = load_image(paths[0])
    
    dims = [len(paths)] + list(image_0.shape)
    result = torch.zeros(dims)
    
    for i, path in enumerate(paths):
        image = load_image(path)
        result[i] = image
    
        if verbose:
            if i % log_every == 0:
                cprint("loaded {}/{} images ({:.2f}%).".format(i, len(paths), i  / len(paths) * 100), logfile)

    if verbose:
        cprint("loaded {}/{} images ({:.2f}%).".format(len(paths), len(paths), 100), logfile)
        
    return result.permute(0, 3, 1, 2)
    
def load_image(path: str) -> torch.Tensor:
    return torch.from_numpy(np.array(np.load(path), dtype=np.float32))

def drop_redundant_metadata_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    " Drops unused and redundant columns "

    to_drop = ["Unnamed: 0", "Single_Cell_Image_Name", "Image_FileName_DAPI", "Image_PathName_DAPI", "Image_FileName_Tubulin", "Image_PathName_Tubulin", "Image_FileName_Actin", "Image_PathName_Actin"]
    result = metadata.drop(columns=to_drop)

    if not isinstance(result, pd.DataFrame):
        raise TypeError("error")

    return result
 
def create_directory(dir_path: str):
    """ subdirectories are also created, e.g. folderA/folderB """
    if not os.path.exists(dir_path):
       os.makedirs(dir_path)

def get_server_directory_path() -> str:
    return "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/"


def shuffle_metadata(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    result = df.sample(frac=1, random_state = seed)
    result = result.reset_index(drop=True)
    return result
    
def split_metadata(df: pd.DataFrame, split_fraction: float) -> pd.DataFrame:
    N_rows = df.shape[0]
    split_index = N_rows * split_fraction
    mask = np.arange(N_rows) <= split_index
    return df[mask], df[~mask]
    
def get_label_mappings(labels: np.ndarray) -> Dict[str, int]:
    label_to_id = {}
    for label in labels:
        label_to_id[label] = len(label_to_id)
    return label_to_id
    
def get_MOA_mappings(metadata: pd.DataFrame) -> Dict[str, int]:
    labels = metadata["moa"].unique()
    labels.sort()
    return get_label_mappings(labels)
    

def get_MOA_to_id() -> Dict[str, int]:
    return {'Actin disruptors': 0,
            'Aurora kinase inhibitors': 1,
            'Cholesterol-lowering': 2,
            'DMSO': 3,
            'DNA damage': 4,
            'DNA replication': 5,
            'Eg5 inhibitors': 6,
            'Epithelial': 7,
            'Kinase inhibitors': 8,
            'Microtubule destabilizers': 9,
            'Microtubule stabilizers': 10,
            'Protein degradation': 11,
            'Protein synthesis': 12}
