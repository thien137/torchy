class Optimizer(ABC):
    def __init__(self):
        self.lr = None 
        self.lr_scheduler = None 

    @abstractmethod
    def step(self):
        pass