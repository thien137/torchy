class Dataset():
    """An abstract class representing a Dataset.""" 

    def __len__(self)-> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__")
    
    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__")
