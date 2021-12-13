
import torch
import utils_torch
def Probability2MostProbableIndex(Probability):
    # Probability: [BatchSize, ClassNum]
    #Max, MaxIndices = torch.max(Probability, dim=1)
    #return utils_torch.TorchTensor2NpArray(MaxIndices) # [BatchSize]
    return torch.argmax(Probability, axis=1)

def LogAccuracyForSingleClassPrediction(ClassIndexPredicted, ClassIndexTruth, log):
    #log = utils_torch.ParseLog(log)
    ClassIndexPredicted = utils_torch.ToNpArray(ClassIndexPredicted)
    ClassIndexTruth = utils_torch.ToNpArray(ClassIndexTruth)
    NumCorrect, NumTotal = utils_torch.evaluate.CalculateAccuracyForSingelClassPrediction(ClassIndexPredicted, ClassIndexTruth)
    log.AddLogDict(
        "Accuracy",
        {
            "SampleNumTotal": NumTotal,
            "SampleNumCorrect": NumCorrect,
        }
    )

