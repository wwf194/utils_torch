import torch
import utils_torch
from utils_torch.attrs import set_attrs, has_attrs, ensure_attrs

def init_from_param(param):
    # to be implemented
    return
def load_model(param):
    return

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, param):
        super(MultiLayerPerceptron, self).__init__()
        if param is not None:
            self.init_from_param(param)
    def init_from_param(self, param):
        self.param = param
        self.layers = []
        ensure_attrs(param, "initialize.method", default="FromNeuronNum")
        ensure_attrs(param, "nonlinear", default="ReLU")
        ensure_attrs(param.layers, "bias", default="True")
        ensure_attrs(param.layers, "type", default="f(Wx+b)")
        self.layers = []
        if param.initialize.method in ["FromNeuronNum"]:
            ensure_attrs(param.layers, "num", default=len(param.neurons.num) - 1)
            for layerIndex in range(param.layers.num):
                layerParam = utils_torch.EmptyPyObj()
                set_attrs(layerParam, "type", "SingleLayer")
                set_attrs(layerParam, "subtype", param.layers.type)
                set_attrs(layerParam, "bias", param.layers.bias)
                set_attrs(layerParam, "input.num", value=param.neurons.num[layerIndex])
                set_attrs(layerParam, "output.num", value=param.neurons.num[layerIndex + 1])
                set_attrs(layerParam, "nonlinear", value=param.nonlinear)
                layer = utils_torch.model.build_module(layerParam)
                #setattr(self, "layer%d"%layerIndex, layer)
                self.add_module("layer%d"%layerIndex, layer)
                set_attrs(param, "modules", value=layerParam)
                self.layers.append(layer)
        else:
            raise Exception()

    def forward(self, input):
        activity = utils_torch.EmptyPyObj()
        for layerIndex, layer in enumerate(self.layers):
            output = layer.forward(input)
            set_attrs(activity, "layer%d"%layerIndex, value=output)
            input = output
        return output

    
