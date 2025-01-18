from torchy._tensor import Tensor

_default_device = "cpu"

def set_default_device(device: str):
    global _default_device 
    if device not in ["cpu", "cuda"]:
        raise ValueError("Invalid device. Supported devices are 'cpu and 'cuda'")
    _default_device = device

def get_default_device() -> str:
    return _default_device