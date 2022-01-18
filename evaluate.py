import numpy as np

import utils_torch

def CalculateAccuracyForSingelClassPrediction(ClassIndexPredicted, ClassIndexTruth):
    NumTotal = ClassIndexPredicted.shape[0]
    NumCorrect = np.sum(ClassIndexPredicted==ClassIndexTruth)
    return NumCorrect, NumTotal

def InitAccuracy():
    Accuracy = utils_torch.EmptyPyObj()
    Accuracy.NumTotal = 0
    Accuracy.NumCorrect = 0
    return Accuracy

def ResetAccuracy(Accuracy):
    Accuracy.NumTotal = 0
    Accuracy.NumCorrect = 0
    return Accuracy

def CalculateAccuracy(Accuracy):
    Accuracy.RatioCorrect = 1.0 * Accuracy.NumCorrect / Accuracy.Num
    return Accuracy

def LogAccuracyForSingleClassPrediction(Accuracy, Output, OutputTarget):
    # Output: np.ndarray. Predicted class indices in shape of [BatchNum]
    # OutputTarget: np.ndarray. Ground Truth class indices in shape of [BatchNum]
    NumCorrect, NumTotal = CalculateAccuracyForSingelClassPrediction(Output, OutputTarget)
    Accuracy.NumTotal += NumTotal
    Accuracy.NumCorrect += NumCorrect