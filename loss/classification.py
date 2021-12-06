
import torch

def Probability2MostProbableIndex(Probability):
    # Probability: [BatchSize, ClassNum]
    #Max, MaxIndices = torch.max(Probability, dim=1)
    #return utils_torch.TorchTensor2NpArray(MaxIndices) # [BatchSize]
    return torch.argmax(Probability, axis=1)