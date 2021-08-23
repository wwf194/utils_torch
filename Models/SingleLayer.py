import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.model import *
from utils_torch.utils import *
from utils_torch.utils import has_attrs, ensure_attrs, match_attrs, compose_function, set_attrs
from utils_torch.model import get_non_linear_function, get_constraint_function, create_self_connection_mask, create_excitatory_inhibitory_mask, create_2D_weight

def init_from_param(param):
    model = SingleLayer()
    model.init_from_param(param)
    return model

def load_model(args):
    return 

class SingleLayer(nn.Module):
    def __init__(self, param=None):
        super(SingleLayer, self).__init__()
        if param is not None:
            self.init_from_param(param)
    def init_from_param(self, param):
        super(SingleLayer, self).__init__()
        set_attrs(param, "type", value="SingleLayer")
        ensure_attrs(param, "subtype", default="f(Wx+b)")
        self.param = param
        ensure_attrs(param, "subtype", default="f(Wx+b)")

        ensure_attrs(param, "weight", default=utils_torch.PyObjFromJson(
            {"initialize":{"method":"kaiming", "coefficient":1.0}}))

        if not has_attrs(param.weight, "size"):
            set_attrs(param.weight, "size", value=[param.input.num, param.output.num])
        if param.subtype in ["f(Wx+b)"]:
            self.create_weight()
            self.create_bias()
            self.NonLinear = get_non_linear_function(param.nonlinear)
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.get_weight()) + self.bias)
        elif param.subtype in ["f(Wx)+b"]:
            self.create_weight()
            self.create_bias()
            self.NonLinear = get_non_linear_function(param.nonlinear)
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.get_weight())) + self.bias
        elif param.subtype in ["Wx"]:
            self.create_weight()
            self.forward = lambda x:torch.mm(x, self.get_weight())
        elif param.subtype in ["Wx+b"]:
            self.create_weight()
            self.create_bias()
            self.forward = lambda x:torch.mm(x, self.get_weight()) + self.bias         
        else:
            raise Exception("SingleLayer: Invalid subtype: %s"%param.subtype)
    def create_bias(self, size=None):
        param = self.param
        ensure_attrs(param, "bias", default=False)
        if size is None:
            size = param.weight.size[1]
        if match_attrs(param.bias, value=False):
            self.bias = 0.0
        elif match_attrs(param.bias, value=True):
            self.bias = torch.nn.Parameter(torch.zeros(size))
        else:
            # to be implemented 
            raise Exception()
    def create_weight(self):
        param = self.param
        sig = has_attrs(param.weight, "size")
        if not has_attrs(param.weight, "size"):
            set_attrs(param.weight, "size", value=[param.input.num, param.output.num])
        self.weight = torch.nn.Parameter(create_2D_weight(param.weight))
        get_weight_function = [lambda :self.weight]
        if match_attrs(param.weight, "isExciInhi", value=True):
            self.ExciInhiMask = create_excitatory_inhibitory_mask(*param.weight.size, param.weight.excitatory.num, param.weight.inhibitory.num)
            get_weight_function.append(lambda weight:weight * self.ExciInhiMask)
            ensure_attrs(param.weight, "ConstraintMethod", value="AbsoluteValue")
            self.WeightConstraintMethod = get_constraint_function(param.weight.ConstraintMethod)
            get_weight_function.append(self.WeightConstraintMethod)
        if match_attrs(param.weight, "NoSelfConnection", value=True):
            if param.weight.size[0] != param.weight.size[1]:
                raise Exception("NoSelfConnection requires weight to be square matrix.")
            self.SelfConnectionMask = create_self_connection_mask(param.weight.size[0])            
            get_weight_function.append(lambda weight:weight * self.SelfConnectionMask)
        self.get_weight = compose_function(get_weight_function)