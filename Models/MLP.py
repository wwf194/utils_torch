import torch
import utils_torch
from utils_torch.attrs import SetAttrs, HasAttrs, EnsureAttrs

def InitFromParam(param):
    # to be implemented
    return
def load_model(param):
    return

class MLP(torch.nn.Module):
    def __init__(self, param):
        super(MLP, self).__init__()
        if param is not None:
            self.param = param
    def InitFromParam(self, param):
        self.param = param
        self.Layers = []
        EnsureAttrs(param, "Initialize.Method", default="FromNeuronNum")
        EnsureAttrs(param, "NonLinear", default="ReLU")
        EnsureAttrs(param.Layers, "Bias", default="True")
        EnsureAttrs(param.Layers, "Type", default="f(Wx+b)")
        self.Layers = []
        if param.Initialize.Method in ["FromNeuronNum"]:
            EnsureAttrs(param.Layers, "Num", default=len(param.Neurons.Num) - 1)
            for LayerIndex in range(param.Layers.Num):
                LayerParam = utils_torch.EmptyPyObj()
                SetAttrs(LayerParam, "Type", "SingleLayer")
                SetAttrs(LayerParam, "Subtype", param.Layers.Type)
                SetAttrs(LayerParam, "Bias", param.Layers.Bias)
                SetAttrs(LayerParam, "Input.Num", value=param.Neurons.Num[LayerIndex])
                SetAttrs(LayerParam, "Output.Num", value=param.Neurons.Num[LayerIndex + 1])
                SetAttrs(LayerParam, "NonLinear", value=param.NonLinear)
                layer = utils_torch.model.BuildModule(LayerParam)
                #setattr(self, "layer%d"%LayerIndex, layer)
                self.add_module("layer%d"%LayerIndex, layer)
                SetAttrs(param, "Modules", value=LayerParam)
                self.Layers.append(layer)
        else:
            raise Exception()
    def forward(self, Input):
        States = utils_torch.EmptyPyObj()
        for LayerIndex, layer in enumerate(self.Layers):
            Output = layer.forward(getattr(States, "%d"%(LayerIndex)))
            setattr(States, "%d"%(LayerIndex + 1), value=Output)
        return {
            "Output": getattr(States, len(self.Layers))
        }