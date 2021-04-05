import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils_ import ensure_path, get_args, get_name, get_name_args, get_from_dict, search_dict, contain, contain_all, get_row_col, prep_title

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

def get_mask(N_num, output_num, device=None):
    if device is None:
        device = torch.device('cpu')
    mask = torch.ones((N_num, output_num), device=device, requires_grad=False)
    return mask

def get_cons_func(method):
    if method in ['abs']:
        return lambda x:torch.abs(x)
    elif method in ['square']:
        return lambda x:x ** 2
    elif method in ['force']:
        return lambda x:x

get_constraint_func = get_cons_func

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

def get_act_func(act_func_info):
    if isinstance(act_func_info, list):
        act_func_name = act_func_info[0]
        act_func_param = act_func_info[1]
    elif isinstance(act_func_info, str):
        act_func_name = act_func_info
        act_func_param = None
    elif isinstance(act_func_info, dict):
        act_func_name = act_func_info['name']
        act_func_param = act_func_info['param']
    else:
        raise Exception('Invalid act_func_info type: %s'%type(act_func_info))
    return get_act_func_from_str(act_func_name, act_func_param)

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
    if name in ['none']:
        return lambda x:x
    elif name in ['relu']:
        if param in ['default']:
            return lambda x:F.relu(x)
        else:
            return lambda x:param * F.relu(x)
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

def build_optimizer(dict_, model, load=False):
    type_ = get_from_dict(dict_, 'type', default='sgd', write_default=True)
    #func = dict_['func'] #forward ; rec, output, input
    #lr = get_from_dict(dict_, 'lr', default=1.0e-3, write_default=True)
    lr = dict_['lr']
    weight_decay = get_from_dict(dict_, 'weight_decay', default=0.0, write_default=True)
    params = model.parameters()
    if type_ in ['adam', 'ADAM']:
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif type_ in ['rmsprop']:
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
    #print('bbb')
    return scattered_label # [batch_size, num_class]

def build_mlp(dict_, load=False, device=None):
    return MLP(dict_, load=load, device=device)

class MLP(nn.Module):
    def __init__(self, dict_, load=False, device=None):#input_num is neuron_num.
        super(MLP, self).__init__()
        self.dict = dict_
        if device is None:
            self.device = self.dict.setdefault('device', 'cpu')
        else:
            self.device = self.dict['device'] = device
        self.act_func_on_last_layer = self.dict.setdefault('act_func_on_last_layer', False)
        self.use_bias = self.dict.setdefault('bias', False)
        self.bias_on_last_layer = self.dict.setdefault('bias_on_last_layer', True)
        self.N_nums = self.dict['N_nums']
        self.layer_num = self.dict['layer_num'] = len(self.N_nums) - 1
        print('use_bias: %s'%self.use_bias)
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
            self.register_parameter(name='w%d'%index, param=weight)

        if self.use_bias:
            for index, bias in enumerate(self.biases):
                if isinstance(bias, torch.Tensor):
                    self.register_parameter('b%d'%index, bias)

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

        #print_model_params(self)
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

def print_model_params(model):
    for name, param in model.named_parameters():
        print(param)
        print('This is my %s. size:%s is_leaf:%s device:%s requires_grad:%s'%
            (name, list(param.size()), param.is_leaf, param.device, param.requires_grad))

def get_tensor_info(tensor, name='', verbose=True, complete=True):
    report = '%s...\033[0;31;40mVALUE\033[0m\n'%str(tensor)
    if complete:
        report += '%s...\033[0;31;40mGRADIENT\033[0m\n'%str(tensor.grad)
    report += 'Tensor \033[0;32;40m%s\033[0m: size:%s is_leaf:%s device:%s type:%s requires_grad:%s'%\
        (name, list(tensor.size()), tensor.is_leaf, tensor.device, tensor.type(), tensor.requires_grad)
    if verbose:
        print(report)
    return report

def print_optimizer_params(optimizer):
    dict_ = optimizer.state_dict()
    for key, value in dict_.items():
        print('%s: %s'%(key, value))

