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
            self.data = utils_torch.json.EmptyPyObj()
            self.cache = utils_torch.json.EmptyPyObj()
    def InitFromParam(self, param=None):
        if param is None:
            param = self.param
            data = self.data
            cache = self.cache
        else:
            self.param = param
        cache.Layers = []
        EnsureAttrs(param, "Init.Method", default="FromNeuronNum")
        EnsureAttrs(param, "NonLinear", default="ReLU")
        EnsureAttrs(param.Layers, "Bias", default="True")
        EnsureAttrs(param.Layers, "Type", default="f(Wx+b)")
        cache.Layers = []
        if param.Init.Method in ["FromNeuronNum"]:
            EnsureAttrs(param.Layers, "Num", default=len(param.Neurons.Num) - 1)
            for LayerIndex in range(param.Layers.Num):
                LayerParam = utils_torch.json.EmptyPyObj()
                SetAttrs(LayerParam, "Type", "NonLinearLayer")
                SetAttrs(LayerParam, "Subtype", param.Layers.Type)
                SetAttrs(LayerParam, "Bias", param.Layers.Bias)
                SetAttrs(LayerParam, "Input.Num", value=param.Neurons.Num[LayerIndex])
                SetAttrs(LayerParam, "Output.Num", value=param.Neurons.Num[LayerIndex + 1])
                SetAttrs(LayerParam, "NonLinear", value=param.NonLinear)
                Layer = utils_torch.model.BuildModule(LayerParam)
                self.add_module("Layer%d"%LayerIndex, Layer)
                SetAttrs(param, "Modules", value=LayerParam)
                cache.Layers.append(Layer)
        else:
            raise Exception()
        
        for Layer in cache.Layers:
            Layer.InitFromParam()

    def forward(self, Input):
        cache = self.cache
        States = {}
        States["0"] = Input
        for LayerIndex, Layer in enumerate(cache.Layers):
            Output = Layer.forward(States[str(LayerIndex)])
            States[str(LayerIndex + 1)] =Output
        return [
            States[str(len(cache.Layers))]
        ]
    def SetTensorLocation(self, Location):
        cache = self.cache
        for Layer in cache.Layers:
            Layer.SetTensorLocation(Location)