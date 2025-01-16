from torchy import Tensor 
from abc import ABC, abstractmethod

import numpy as np

class Dataset(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def __len__(self)-> int:
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        pass
    
class DataLoader():
    def __init__(self, data: Dataset, batch_size: int, shuffle: bool):
        self.data = data 
        self.batch_size = batch_size
        self.shuffle = shuffle 
    
    def __iter__(self):
        inds = np.arange(0, len(self.data))
        if self.shuffle:
            np.random.shuffle(inds)
        
        for batch_ids in np.array_split(inds, np.ceil(len(inds) / self.batch_size)):
            yield self.data[batch_ids]