from utils_model import *

class SingleLayer(nn.Module):
    def __init__(self, param):
        super(SingleLayer, self).__init__()
        ensure_attrs(param, "subtype", default="f(Wx+b)")
        self.param = param


        if match_attrs(param.bias, value=False):
            self.bias = 0.0
        elif match_attrs(param.bias, value=True):
            self.bias = torch.nn.Parameter(torch.zeros(param.weight.size[1]))
        else:
            # to be implemented
            raise Exception()
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

        else:
            raise Exception("SingleLayer: Invalid subtype: %s"%param.subtype)
    
    def create_weight(self):
        param = self.param
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