import torch
import utils_torch
from utils_torch.attrs import SetAttrs, HasAttrs, EnsureAttrs

def InitFromParams(param):
    # to be implemented
    return
def load_model(param):
    return

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, param):
        super(MultiLayerPerceptron, self).__init__()
        if param is not None:
            self.InitFromParams(param)
    def InitFromParams(self, param):
        self.param = param
        self.Layers = []
        EnsureAttrs(param, "Initialize.Method", default="FromNeuronNum")
        EnsureAttrs(param, "NonLinear", default="ReLU")
        EnsureAttrs(param.Layers, "Bias", default="True")
        EnsureAttrs(param.Layers, "Type", default="f(Wx+b)")
        self.Layers = []
        if param.Initialize.Method in ["FromNeuronNum"]:
            EnsureAttrs(param.Layers, "Num", default=len(param.Neurons.Num) - 1)
            for layerIndex in range(param.Layers.Num):
                layerParam = utils_torch.EmptyPyObj()
                SetAttrs(layerParam, "Type", "SingleLayer")
                SetAttrs(layerParam, "Subtype", param.Layers.Type)
                SetAttrs(layerParam, "Bias", param.Layers.Bias)
                SetAttrs(layerParam, "Input.Num", value=param.Neurons.Num[layerIndex])
                SetAttrs(layerParam, "Output.Num", value=param.Neurons.Num[layerIndex + 1])
                SetAttrs(layerParam, "NonLinear", value=param.NonLinear)
                layer = utils_torch.model.BuildModule(layerParam)
                #setattr(self, "layer%d"%layerIndex, layer)
                self.add_module("layer%d"%layerIndex, layer)
                SetAttrs(param, "Modules", value=layerParam)
                self.Layers.append(layer)
        else:
            raise Exception()

    def forward(self, input):
        activity = utils_torch.EmptyPyObj()
        for layerIndex, layer in enumerate(self.Layers):
            output = layer.forward(input)
            SetAttrs(activity, "layer%d"%layerIndex, value=output)
            input = output
        return output

    
