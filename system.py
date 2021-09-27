import torch

def PyTorchInfo():
    if torch.cuda.is_available():
        print("Cuda is available")
    else:
        print("Cuda is unavailable")

    print("Torch version:"+torch.__version__)