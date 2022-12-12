import torch
import numpy as np
import pandas as pd
from typing import List, Set, Dict, Tuple, Optional, Any
from collections import defaultdict

from gmfpp.utils.data_preparation import *

def extract_batch_from_indices(indices: np.ndarray, images: torch.Tensor, metadata: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_images = extract_images_from_metadata_indices(indices, images)
    batch_labels = extract_MOA_ids_from_indices(indices, metadata)
    return batch_images, batch_labels

def get_treatment_indices(metadata: pd.DataFrame) -> Dict[str, list]:
    treatments = np.array(metadata[['Image_Metadata_Compound','Image_Metadata_Concentration']])
    result = defaultdict(list)
    
    for i in range(treatments.shape[0]):
        compound, concetration = treatments[i]
        result[(compound, concetration)] += [i]
    
    return result
    
def extract_images_from_metadata_indices(indices: np.ndarray, images: torch.Tensor) -> torch.Tensor:
    return images[indices]

def extract_MOA_ids_from_indices(indices: np.ndarray, metadata: pd.DataFrame) -> torch.Tensor:
    moa_to_id = get_MOA_to_id()
    rows = metadata["moa"].loc[indices]
    return torch.tensor([moa_to_id[row] for row in rows])
    
class TreatmentBalancedBatchGenerator:
    
    def __init__(self, images: torch.Tensor, metadata: pd.DataFrame):
        self.treatment_indices = get_treatment_indices(metadata)
        self.images = images
        self.metadata = metadata
        
        self.treatment_index_at = defaultdict(int)
        
        for treatment in self.treatment_indices:
            assert(len(self.treatment_indices[treatment]) > 0)
            
            self.treatment_index_at[treatment] = 0
    
    def next_indices(self) -> np.ndarray:
        result = np.empty(len(self.treatment_indices), dtype=np.int64)
        
        for i, treatment in enumerate(self.treatment_indices):
            indices = self.treatment_indices[treatment]
            index = self.treatment_index_at[treatment]
            self.treatment_index_at[treatment] = (index + 1) % len(indices)
            result[i:i+1] = indices[index]
        
        return result
    
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self.next_indices()
        X, y = extract_batch_from_indices(indices, self.images, self.metadata)
        return X, y

