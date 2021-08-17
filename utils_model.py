from utils import PyJSON, ensure_attrs, has_attrs
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils_pytorch.utils import ensure_path, get_args, get_name, get_name_args, get_from_dict, search_dict, contain, contain_all, get_row_col, prep_title
from utils_pytorch.utils import get_attrs, compose_func, match_attrs, compose_function
from utils_pytorch.LRSchedulers import LinearLR

def build_module(module):
    if module.type in ["SingleLayer"]:
        return SingleLayer(module)

def create_self_connection_mask(size):
    return torch.from_numpy(np.ones(size, size) - np.eye(size))

def create_excitatory_inhibitory_mask(input_num, output_num, excitatory_num, inhibitory_num=None):
    if inhibitory_num is None:
        inhibitory_num = input_num - excitatory_num
    else:
        if input_num != excitatory_num + inhibitory_num:
            raise Exception("get_excitatory_inhibitory_mask: input_num==excitatory_num + inhibitory_num must be satisfied.")

    excitatory_mask = np.ones(excitatory_num, output_num)
    inhibitory_mask = np.ones(inhibitory_num, output_num)
    excitatory_inhibitory_mask = np.concatenate([excitatory_mask, inhibitory_mask], axis=0)
    return excitatory_inhibitory_mask

def create_mask(N_num, output_num, device=None):
    if device is None:
        device = torch.device('cpu')
    mask = torch.ones((N_num, output_num), device=device, requires_grad=False)
    return mask

def get_constraint_function(method):
    if method in ["AbsoluteValue", "abs"]:
        return lambda x:torch.abs(x)
    elif method in ["Square", "square"]:
        return lambda x:x ** 2
    elif method in ["CheckAfterUpdate", "force"]:
        return lambda x:x
    else:
        raise Exception("get_constraint_function: Invalid consraint method: %s"%method)

def get_non_linear_function(description):
    param = parse_non_linear_function_description(description)
    if param.type in ["relu"]:
        if param.coefficient==1.0:
            return F.relu
        else:
            return lambda x:param.coefficient * F.relu(x)
    elif param.type in ["tanh"]:
        if param.coefficient==1.0:
            return F.tanh
        else:
            return lambda x:param.coefficient * F.tanh(x)       
    elif param.type in ["sigmoid"]:
        if param.coefficient==1.0:
            return F.tanh
        else:
            return lambda x:param.coefficient * F.tanh(x)         
    else:
        raise Exception("get_non_linear_function: Invalid nonlinear function description: %s"%description)

get_activation_function = get_non_linear_function

def parse_non_linear_function_description(description):
    if isinstance(description, str):
        return PyJSON({
            "type":description,
            "coefficient": 1.0
        })
    elif isinstance(description, list):
        if len(description)==2:
            return PyJSON({
                "type": description[0],
                "coefficient": description[1]
            })
        else:
            # to be implemented
            pass
    elif isinstance(description, object):
        return description
    else:
        raise Exception("parse_non_linear_function_description: invalid description type: %s"%type(description))

class SingleLayer(nn.Module):
    def __init__(self, param):
        super(SingleLayer, self).__init__()
        ensure_attrs(param, "subtype", default="f(Wx+b)")
        self.param = param
        param.weight = param.weight
    
        self.weight = create_2D_weight(param.weight)
        self.register_parameter("weight", self.weight)
        
        get_weight_function = [lambda :self.weight]
        if match_attrs(param.weight, "isExcitatoryInhibitory", value=True):
            self.ExcitatoryInhibitoryMask = create_excitatory_inhibitory_mask(*param.weight.size, param.weight.excitatory.num, param.weight.inhibitory.num)
            get_weight_function.append(lambda weight:weight * self.ExcitatoryInhibitoryMask)
            ensure_attrs(param.weight, "ConstraintMethod", value="AbsoluteValue")
            self.WeightConstraintMethod = get_constraint_function(param.weight.ConstraintMethod)
            get_weight_function.append(self.WeightConstraintMethod)
        if match_attrs(param.weight, "NoSelfConnection", value=True):
            if param.weight.size[0] != param.weight.size[1]:
                raise Exception("NoSelfConnection requires weight to be square matrix.")
            self.SelfConnectionMask = create_self_connection_mask(param.weight.size[0])            
            get_weight_function.append(lambda weight:weight * self.SelfConnectionMask)
        self.get_weight = compose_function(get_weight_function)

        if match_attrs(param.bias, value=False):
            self.bias = 0.0
        elif match_attrs(param.bias, value=True):
            self.bias = torch.zeros(param.weight.size[1])
            self.register_parameter("bias", self.bias)
        else:
            # to be implemented
            pass

        self.NonLinear = get_non_linear_function()

        if param.subtype in ["f(Wx+b)"]:
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.weight) + self.bias)
        elif param.subtype in ["f(Wx)+b"]:
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.weight)) + self.bias
        else:
            raise Exception("SingleLayer: Invalid subtype: %s"%param.subtype)

def create_2D_weight(param):
    if param.method in ["kaiming", "he"]:
        param.method = "kaiming"
        ensure_attrs(param, "mode", default="in")
        ensure_attrs(param, "distribution", default="uniform")
        ensure_attrs(param, "coefficient", default=1.0)
        if param.mode in ["in"]:
            if param.distribution in ["uniform"]:
                range = [
                    - param.coefficient * 6 ** 0.5 / param.size[0] ** 0.5,
                    param.coefficient * 6 ** 0.5 / param.size[0] ** 0.5
                ]
                weight = np.random.uniform(*range, tuple(param.size))
            elif param.distribution in ["uniform+"]:
                range = [
                    0.0,
                    2.0 * param.coefficient * 6 ** 0.5 / param.size[0] ** 0.5
                ]
                weight = np.random.uniform(*range, tuple(param.size))
            else:
                # to be implemented
                raise Exception()
        else:
            raise Exception()
            # to be implemented
    
    elif param.method in ["xaiver", "glorot"]:
        param.method = "xaiver"

    else:
        raise Exception()
        # to be implemented

    return torch.from_numpy(weight)

def init_weight(weight, init_info, cons_method=None): # weight: [input_num, output_num]
    name, args = get_name_args(init_info)
    coeff = 1.0
    init_method = None
    if isinstance(args, float):
        coeff = args
    if isinstance(name, str):
        name = [name]

    if contain(name, ['ortho', 'orthogonal']):
        init_method = 'ortho'
    elif contain(name, ['glorot', 'glorot_uniform', 'xavier', 'xavier_uniform']):
        init_method = 'glorot'
    elif contain(name, 'uniform'):
        init_method = 'uniform'
    elif contain(name, ['gaussian', 'normal']):
        init_method = 'normal'
    else:
        init_method = 'uniform' # default distribution
    if init_method in ['normal', 'uniform']: # init method that has input/output mode.
        if contain(name, ['input_output', 'output_input', 'in_out', 'out_in']):
            # to be implemented
            pass
        elif contain(name, 'input'):
            if cons_method is None or cons_method in ['abs']:
                lim_l = - coeff / weight.size(0)
                lim_r = - lim_l
            elif cons_method in ['force']:
                lim_l = 0.0
                lim_r = 2 * coeff / weight.size(0)
            else:
                raise Exception('Invalid')            
        elif contain(name, 'output'):
            if cons_method is None or cons_method in ['abs']:
                lim_l = - coeff / weight.size(1)
                lim_r = - lim_l
            elif cons_method in ['force']:
                lim_l = 0.0
                lim_r = 2 * coeff / weight.size(1)
            else:
                raise Exception('Invalid cons_method: %s'%cons_method)
        else:
            raise Exception('Invalid init weight mode: %s'%str(name))
    if init_method is not None:
        if init_method in ['uniform']:
            torch.nn.init.uniform_(weight, lim_l, lim_r)  
        elif init_method in ['normal']:
            torch.nn.init.normal_(weight, mean=(lim_l+lim_r)/2, std=(lim_r+lim_l)/2)  
        elif init_method in ['ortho']:
            weight_ = weight.detach().clone()
            torch.nn.init.orthogonal_(weight_, gain=1.0) # init input weight to be orthogonal.
            with torch.no_grad():  # avoid gradient calculation error during in-place operation.
                weight.copy_( weight_ * coeff )        
        elif init_method in ['glorot']:
            weight_ = weight.detach().clone()
            torch.nn.init.xavier_uniform_(weight_, gain=1.0)
            with torch.no_grad():
                weight.copy_( weight_ * coeff )
        else:
            raise Exception('Invalid init method: %s'%init_method)

def get_ei_mask(E_num, N_num, device=None):
    if device is None:
        device = torch.device('cpu')
    ei_mask = torch.zeros((N_num, N_num), device=device, requires_grad=False)
    for i in range(E_num):
        ei_mask[i][i] = 1.0
    for i in range(E_num, N_num):
        ei_mask[i][i] = -1.0
    return ei_mask



def get_loss_func(loss_func_str, truth_is_label=False, num_class=None):
    if loss_func_str in ['MSE', 'mse']:
        if truth_is_label:
            #print('num_class: %d'%num_class)
            #return lambda x, y:torch.nn.MSELoss(x, scatter_label(y, num_class=num_class))
            return lambda x, y:F.mse_loss(x, scatter_label(y, num_class=num_class))
        else:
            return torch.nn.MSELoss()
    elif loss_func_str in ['CEL', 'cel']:
        return torch.nn.CrossEntropyLoss()
    else:
        raise Exception('Invalid main loss: %s'%loss_func_str)


def get_act_func_module(act_func_str):
    name = act_func_str
    if act_func_str in ['relu']:
        return nn.ReLU()
    elif act_func_str in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif act_func_str in ['softplus', 'SoftPlus']:
        return nn.Softplus()
    elif act_func_str in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    else:
        raise Exception('Invalid act func str: %s'%act_func_str)
        
def get_act_func_from_str(name='relu', param=None):
    if param is None:
        param = 'default'
    if name in ['none']:
        return lambda x:x
    elif name in ['relu']:
        #print(param)
        if param in ['default']:
            return lambda x: F.relu(x)
        else:
            return lambda x: param * F.relu(x)
    elif name in ['tanh']:
        if param in ['default']:
            return lambda x:torch.tanh(x)
        else:
            return lambda x:param * F.tanh(x)
    elif name in ['relu_tanh']:
        if param in ['default']:
            return lambda x:F.relu(torch.tanh(x))
        else:
            return lambda x:param * F.relu(torch.tanh(x))
    else:
        raise Exception('Invalid act func name: %s'%name)

def build_optimizer(dict_, params=None, model=None, load=False):
    type_ = get_from_dict(dict_, 'type', default='sgd', write_default=True)
    #func = dict_['func'] #forward ; rec, output, input
    #lr = get_from_dict(dict_, 'lr', default=1.0e-3, write_default=True)
    lr = dict_['lr']
    weight_decay = get_from_dict(dict_, 'weight_decay', default=0.0, write_default=True)
    if params is not None:
        pass
    elif model is not None:
        if hasattr(model, 'get_param_to_train'):
            params = model.get_param_to_train()
        else:
            params = model.parameters()
    else:
        raise Exception('build_optimizer: Both params and model are None.')
    
    if type_ in ['adam', 'ADAM']:
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif type_ in ['rmsprop', 'RMSProp']:
        optimizer = optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    elif type_ in ['sgd', 'SGD']:
        momentum = dict_.setdefault('momentum', 0.9)
        optimizer = optim.SGD(params, momentum=momentum, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception('build_optimizer: Invalid optimizer type: %s'%type_)

    if load:
        optimizer.load_state_dict(dict_['state_dict'])
    else:
        dict_['state_dict'] = optimizer.state_dict()
    return optimizer

def build_scheduler(dict_, optimizer, load=False, verbose=True):
    #lr_decay = dict['lr_decay']
    scheduler_dict = dict_['scheduler']
    scheduler_type = search_dict(scheduler_dict, ['type', 'method'], default='None', write_default=True)
    if verbose:
        print('build_scheduler: scheduler_type: %s'%scheduler_type)
    if scheduler_type is None or scheduler_type in ['None', 'none']:
        scheduler = None
        #update_lr = update_lr_none
    elif scheduler_type in ['exp']:
        decay = search_dict(scheduler_dict, ['decay', 'coeff'], default=0.98, write_default=True, write_default_dict='decay')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)
        #update_lr = update_lr_
    elif scheduler_type in ['stepLR', 'exp_interval']:
        decay = search_dict(scheduler_dict, ['decay', 'coeff'], default=0.98, write_default=True, write_default_key='decay')
        step_size = search_dict(scheduler_dict, ['interval', 'step_size'], default=0.98, write_default=True, write_default_key='decay')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, step_size=step_size, gamma=decay)
        #update_lr = update_lr_
    elif scheduler_type in ['Linear', 'linear']:
        milestones = search_dict(scheduler_dict, ['milestones'], throw_none_error=True)
        scheduler = LinearLR(optimizer, milestones=milestones, epoch_num=dict_['epoch_num'])
        #update_lr = update_lr_
    else:
        raise Exception('build_scheduler: Invalid lr decay method: '+str(scheduler_type))
    return scheduler

# search for directory or file of most recently saved models(model with biggest epoch index)
def get_last_model(model_prefix, base_dir=None, is_dir=True):
    if is_dir:
        max_epoch = None
        pattern = model_prefix+'(\d+)'
        dirs = os.listdir(base_dir)
        for dir_name in dirs:
            result = re.search(r''+pattern, dir_name)
            if result is not None:
                try:
                    epoch_num = int(result.group(1))
                except Exception:
                    print('error in matching model name.')
                    continue
                if max_epoch is None:
                    max_epoch = epoch_num
                else:
                    if max_epoch < epoch_num:
                        max_epoch = epoch_num
    if max_epoch is not None:
        return base_dir + model_prefix + str(max_epoch) + '/'
    else:
        return 'error'

def cal_acc_from_label(output, label):
    # output: [batch_size, num_class]; label: [batch_size], label[i] is the index of correct category of i_th batch.
    correct_num = (torch.max(output, dim=1)[1]==label).sum().item()
    sample_num = label.size(0)
    #return {'correct_num':correct_num, 'data_num':label_num} 
    return correct_num, sample_num

def scatter_label(label, num_class=None, device=None): # label: must be torch.LongTensor, shape: [batch_size], label[i] is the index of correct category of i_th batch.
    #print('aaa')
    if num_class is None:
        #print(torch.max(label).__class__)
        num_class = torch.max(label).item() + 1
    scattered_label = torch.zeros((label.size(0), num_class), device=device).to(label.device).scatter_(1, torch.unsqueeze(label, 1), 1)
    #return scattered_label.long() # [batch_size, num_class]
    #print(label.type())
    #print(scattered_label.type())
    return scattered_label # [batch_size, num_class]

def build_mlp(dict_, load=False, device=None):
    print(dict_)
    return MLP(dict_, load=load, device=device)

class MLP(nn.Module):
    def __init__(self, dict_, load=False, device=None):#input_num is neuron_num.
        super(MLP, self).__init__()
        self.dict = dict_
        if device is None:
            self.device = self.dict.setdefault('device', 'cpu')
        else:
            self.device = device
            self.dict['device'] = device
        self.act_func_on_last_layer = self.dict.setdefault('act_func_on_last_layer', False)
        self.use_bias = self.dict.setdefault('bias', False)
        self.bias_on_last_layer = self.dict.setdefault('bias_on_last_layer', True)
        self.N_nums = self.dict['N_nums']
        self.layer_num = self.dict['layer_num'] = len(self.N_nums) - 1
        #print('use_bias: %s'%self.use_bias)
        self.params = {}
        if load:
            self.weights = self.dict['weights']
            self.biases = self.dict['biases']
        else:
            self.weights = self.dict['weight'] = []
            self.biases = self.dict['biases'] = []
            if self.dict.get('init_weight') is None:
                self.dict['init_weight'] = [['input', 'uniform'], 1.0]
            self.init_weight = self.dict['init_weight']
            #print(self.layer_num)
            #print(self.N_nums)
            
            for layer_index in range(self.layer_num):
                weight = nn.Parameter(torch.zeros(self.N_nums[layer_index], self.N_nums[layer_index + 1], device=self.device))
                #for parameter in self.parameters():
                #    print(parameter.size())
                init_weight(weight, self.init_weight)
                #self.register_parameter(name='w%d'%layer_index, param=weight)
                self.weights.append(weight)
                
            for layer_index in range(self.layer_num - 1):
                if self.use_bias:
                    bias = nn.Parameter(torch.zeros(self.N_nums[layer_index], device=self.device))  
                else:
                    bias = 0.0
                self.biases.append(bias)
            
            if self.bias_on_last_layer:
                bias = nn.Parameter(torch.zeros(self.N_nums[-1], device=self.device))
                self.biases.append(bias)
                #self.register_parameter('b%d'%self.layer_num, bias)
            else:
                self.biases.append(0.0)
            
        for index, weight in enumerate(self.weights):
            name = 'w%d'%index
            self.register_parameter(name=name, param=weight)
            self.params[name] = weight

        if self.use_bias:
            for index, bias in enumerate(self.biases):
                name = 'b%d'%index
                if isinstance(bias, torch.Tensor):
                    self.register_parameter(name, bias)
                self.params[name] = bias
        
        if not (self.layer_num == 1 and not self.dict['act_func_on_last_layer']):
            self.act_func = get_act_func(self.dict['act_func'])

        self.alt_weight_scale = self.dict.setdefault('alt_weight_scale', False)
        
        if self.alt_weight_scale:
            self.alt_weight_scale_assess_num = self.dict.setdefault('alt_weight_scale_assess_num', 10)
            self.forward = self.forward_alt_weight_scale
            self.alt_weight_scale_assess_count = 0
        else:
            self.forward = self.forward_

        #self.use_batch_norm = self.dict.setdefault('batch_norm', False)
        self.use_batch_norm = search_dict(self.dict, ['batch_norm', 'use_batch_norm'], default=False, write_default=True)
        if self.use_batch_norm:
            self.bns = []
            for layer_index in range(1, len(self.N_nums)):
                bn = nn.BatchNorm1d(num_features=self.N_nums[layer_index], eps=1.0e-10)
                self.add_module('bn%d'%layer_index, bn)
            self.bns.append(bn)
        self.cache = {}
        #self.mlp = build_mlp_sequential(dict_=self.dict, load=load)
        #print(self.parameters())

        #print_model_param(self)
        #input()
    def forward_alt_weight_scale(self, x, **kw):
        out = self.forward(x)
        out_target = search_dict(kw, ['out_target', 'out_truth'])

        # to be implemented: calculate weight scale

        self.alt_weight_scale_assess_count += 1
        if self.alt_weight_scale_assess_count >= self.alt_weight_scale_assess_num:
            self.forward = self.forward_
            # to be implemented: alter weight scale
        return out
    def forward_(self, x, **kw):
        x = x.to(self.device)
        for layer_index in range(self.layer_num - 1):
            x = self.act_func(torch.mm(x, self.weights[layer_index]) + self.biases[layer_index])
            if self.use_batch_norm:
                x = self.bns[layer_index](x)
        # last layer
        #print(self.weights)
        layer_index = self.layer_num - 1
        x = torch.mm(x, self.weights[layer_index])
        if self.bias_on_last_layer:
            x += self.biases[layer_index]
        if self.act_func_on_last_layer:
            x = self.act_func(x)
        if self.use_batch_norm:
            x = self.bns[-1](x)
        return x
    def anal_weight_change(self, title=None, verbose=True):
        title = prep_title(title)
        result = title
        # analyze weight change
        for index, weight in enumerate(self.weights):
            name = 'w%d'%index
            #print(name)
            w_1 = weight.detach().cpu().numpy()
            if self.cache.get(name) is not None:
                w_0 = self.cache[name]
                w_change_rate = np.sum(abs(w_1 - w_0)) / np.sum(np.abs(w_0))
                result += '%s_change_rate: %.3f '%(name, w_change_rate)
            self.cache[name] = w_1
        # analyze bias change
        for index, bias in enumerate(self.biases):
            name = 'b%d'%index
            #print('%s=%s'%(name, bias))
            if isinstance(bias, torch.Tensor):
                b_1 = bias.detach().cpu().numpy()
                if self.cache.get(name) is not None:
                    b_0 = self.cache[name]
                    b_0_abs = np.sum(np.abs(b_0))
                    if b_0_abs > 0.0:
                        b_change_rate = np.sum(abs(b_1 - b_0)) / b_0_abs
                        result += '%s_change_rate: %.3f '%(name, b_change_rate)
                    else:
                        b_1_abs = np.sum(np.abs(b_1))
                        if b_1_abs==0.0:
                            result += '%s_change_rate: %.3f '%(name, 0.0)
                        else:
                            result += '%s_change_rate: +inf'%(name)
                self.cache[name] = b_1
            elif isinstance(bias, float):
                result += '%s===%.3e'%(name, bias)
            else:
                raise Exception('Invalid bias type: %s'%bias.__class__)
        if verbose:
            print(result)
        return result
    def plot_weight(self, axes=None, save=False, save_path='./', save_name='mlp_weight_plot.png'):
        row_num, col_num = get_row_col()
        if axes is None:
            fig, axes = plt.subplots(nrows=row_num, ncols=col_num, figsize=(5*col_num, 5*row_num))
        if save:
            ensure_path(save_path)
            plt.savefig(save_path + save_name)
        return
    def print_info(self, name='', verbose=True):
        report = '%s. MLP Instance. N_nums: %s. bias: %s'%(name, self.N_nums)
        if verbose:
            print(report)
        return report
    def print_grad(self, verbose=True):
        result = ''
        for index, weight in enumerate(self.weights):
            name = 'w%d'%index
            result += str(weight.grad)
            result += 'This is %s.grad'%name
            if not weight.requires_grad:
                result += '%s does not require grad.'%name
            result += '\n'
        for index, bias in enumerate(self.biases):
            name = 'b%d'%index
            if isinstance(bias, torch.Tensor):
                result += str(bias.grad)
                result += 'This is %s.grad.'%name
                if not bias.requires_grad:
                    result += '%s does not require grad.'%name
                result += '\n'
            else:
                result += '%s is not a tensor, and has not grad.'%name
        if verbose:
            print(result)
        return result

'''
def build_mlp(dict_):
    act_func = get_act_func_module(dict_['act_func'])
    layers = []
    layer_dicts = []
    N_nums = dict_['N_nums'] #input_num, hidden_layer1_unit_num, hidden_layer2_unit_numm ... output_num
    layer_num = len(N_nums) - 1
    for layer_index in range(layer_num):
        current_layer = nn.Linear(N_nums[layer_index], N_nums[layer_index+1], bias=dict['bias'])
        layers.append(current_layer)
        layer_dicts.append(current_layer.state_dict())
        if not (dict_['act_func_on_last_layer'] and layer_index==layer_num-1):
            layers.append(act_func)
    dict['layer_dicts'] = layer_dicts
    return torch.nn.Sequential(*layers)
'''

def build_mlp_sequential(dict_, load=False):
    act_func = get_act_func_module(dict_['act_func'])
    use_bias = dict_['bias']
    layers = []
    layer_dicts = []
    N_nums = dict_['N_nums'] #input_num, hidden_layer1_unit_num, hidden_layer2_unit_numm ... output_num
    layer_num = len(N_nums) - 1
    act_func_on_last_layer = get_from_dict(dict_, 'act_func_on_last_layer', default=True, write_default=True)
    for layer_index in range(layer_num):
        layer_current = nn.Linear(N_nums[layer_index], N_nums[layer_index+1], bias=use_bias)
        if load:
            layer_current.load_state_dict(dict_['layer_dicts'][layer_index])
        layers.append(layer_current)
        layer_dicts.append(layer_current.state_dict())
        if not (act_func_on_last_layer and layer_index==layer_num-1):
            layers.append(act_func)
    return torch.nn.Sequential(*layers)

def print_model_param(model):
    for name, param in model.named_parameters():
        print(param)
        print('This is my %s. size:%s is_leaf:%s device:%s requires_grad:%s'%
            (name, list(param.size()), param.is_leaf, param.device, param.requires_grad))

def get_tensor_info(tensor, name='', verbose=True, complete=True):
    print(tensor.device)
    report = '%s...\033[0;31;40mVALUE\033[0m\n'%str(tensor)
    if complete:
        report += '%s...\033[0;31;40mGRADIENT\033[0m\n'%str(tensor.grad)
    report += 'Tensor \033[0;32;40m%s\033[0m: size:%s is_leaf:%s device:%s type:%s requires_grad:%s'%\
        (name, list(tensor.size()), tensor.is_leaf, tensor.device, tensor.type(), tensor.requires_grad)
    if verbose:
        print(report)
    return report

def get_tensor_stat(tensor, verbose=False):
    return {
        "min": torch.min(tensor),
        "max": torch.max(tensor),
        "mean": torch.mean(tensor),
        "std": torch.std(tensor),
        "var": torch.var(tensor)
    }

def print_optimizer_params(optimizer):
    dict_ = optimizer.state_dict()
    for key, value in dict_.items():
        print('%s: %s'%(key, value))

