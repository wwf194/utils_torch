
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import re

import utils_torch
#from utils_torch.utils import EnsurePath, Getargs, Getname, Getname_args, GetFromDict, search_dict, contain, contain_all, Getrow_col, prep_title
from utils_torch.json import *
from utils_torch.attrs import *
from utils_torch.LRSchedulers import LinearLR
from utils_torch.Router import *

def BuildModule(ModuleParam):
    if ModuleParam.Type in ["SingleLayer"]:
        return utils_torch.Models.SingleLayer(ModuleParam)
    elif ModuleParam.Type in ["MultiLayerPerceptron", "MLP", "mlp"]:
        return utils_torch.Models.MultiLayerPerceptron(ModuleParam)
    elif ModuleParam.Type in ["SerialReceiver"]:
        pass
    elif ModuleParam.Type in ["SerialSender"]:
        pass
    else:
        raise Exception("build_module: No such module: %s"%ModuleParam.Type)
build_module = BuildModule

def CreateSelfConnectionMask(Size):
    return torch.from_numpy(np.ones(Size, Size) - np.eye(Size))

def CreateExcitatoryInhibitoryMask(InputNum, OutputNum, ExcitatoryNum, InhibitoryNum=None):
    # Assumed weight matrix shape: [InputNum, OutputNum]
    if InhibitoryNum is None:
        InhibitoryNum = InputNum - ExcitatoryNum
    else:
        if InputNum != ExcitatoryNum + InhibitoryNum:
            raise Exception("GetExcitatoryInhibitoryMask: InputNum==ExcitatoryNum + InhibitoryNum must be satisfied.")

    ExcitatoryMask = np.ones(ExcitatoryNum, OutputNum)
    InhibitoryMask = np.ones(InhibitoryNum, OutputNum)
    ExcitatoryInhibitoryMask = np.concatenate([ExcitatoryMask, InhibitoryMask], axis=0)
    return ExcitatoryInhibitoryMask

def CreateMask(N_num, OutputNum, device=None):
    if device is None:
        device = torch.device('cpu')
    mask = torch.ones((N_num, OutputNum), device=device, requires_grad=False)
    return mask

create_mask = CreateMask

def GetConstraintFunction(Method):
    if Method in ["AbsoluteValue", "abs"]:
        return lambda x:torch.abs(x)
    elif Method in ["Square", "square"]:
        return lambda x:x ** 2
    elif Method in ["CheckAfterUpdate", "force"]:
        return lambda x:x
    else:
        raise Exception("GetConstraintFunction: Invalid consraint Method: %s"%Method)

def GetNonLinearFunction(description):
    Param = parse_non_linear_function_description(description)
    if Param.Type in ["relu", "ReLU"]:
        if Param.Coefficient==1.0:
            return F.relu
        else:
            return lambda x:Param.Coefficient * F.relu(x)
    elif Param.Type in ["tanh"]:
        if Param.Coefficient==1.0:
            return F.tanh
        else:
            return lambda x:Param.Coefficient * F.tanh(x)       
    elif Param.Type in ["sigmoid"]:
        if Param.Coefficient==1.0:
            return F.tanh
        else:
            return lambda x:Param.Coefficient * F.tanh(x)         
    else:
        raise Exception("GetNonLinearFunction: Invalid nonlinear function description: %s"%description)

Getactivation_function = GetNonLinearFunction

def parse_non_linear_function_description(description):
    if isinstance(description, str):
        return JsonObj2PyObj({
            "Type":description,
            "Coefficient": 1.0
        })
    elif isinstance(description, list):
        if len(description)==2:
            return JsonObj2PyObj({
                "Type": description[0],
                "Coefficient": description[1]
            })
        else:
            # to be implemented
            pass
    elif isinstance(description, object):
        return description
    else:
        raise Exception("parse_non_linear_function_description: invalid description Type: %s"%Type(description))
def Create2DWeight(Param):
    Initialize = Param.Initialize
    if Initialize.Method in ["kaiming", "he"]:
        Initialize.Method = "kaiming"
        EnsureAttrs(Initialize, "mode", default="in")
        EnsureAttrs(Initialize, "Distribution", default="uniform")
        EnsureAttrs(Initialize, "Coefficient", default=1.0)
        if Initialize.mode in ["in"]:
            if Initialize.Distribution in ["uniform"]:
                range = [
                    - Initialize.Coefficient * 6 ** 0.5 / Param.Size[0] ** 0.5,
                    Initialize.Coefficient * 6 ** 0.5 / Param.Size[0] ** 0.5
                ]
                weight = np.random.uniform(*range, tuple(Param.Size))
            elif Initialize.Distribution in ["uniform+"]:
                range = [
                    0.0,
                    2.0 * Initialize.Coefficient * 6 ** 0.5 / Param.Size[0] ** 0.5
                ]
                weight = np.random.uniform(*range, tuple(Param.Size))
            else:
                # to be implemented
                raise Exception()
        else:
            raise Exception()
            # to be implemented
    elif Initialize.Method in ["xaiver", "glorot"]:
        Initialize.Method = "xaiver"
        raise Exception()
        # to be implemented
    else:
        raise Exception()
        # to be implemented
    return torch.from_numpy(weight)

def GetLossFunction(LossFunctionDescription, truth_is_label=False, num_class=None):
    if LossFunctionDescription in ['MSE', 'mse']:
        if truth_is_label:
            #print('num_class: %d'%num_class)
            #return lambda x, y:torch.nn.MSELoss(x, scatter_label(y, num_class=num_class))
            return lambda x, y:F.mse_loss(x, scatter_label(y, num_class=num_class))
        else:
            return torch.nn.MSELoss()
    elif LossFunctionDescription in ['CEL', 'cel']:
        return torch.nn.CrossEntropyLoss()
    else:
        raise Exception('Invalid loss function description: %s'%LossFunctionDescription)

Getloss_function = GetLossFunction

def Getact_func_module(act_func_str):
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
        
def Getact_func_from_str(name='relu', Param=None):
    if Param is None:
        Param = 'default'
    if name in ['none']:
        return lambda x:x
    elif name in ['relu']:
        #print(Param)
        if Param in ['default']:
            return lambda x: F.relu(x)
        else:
            return lambda x: Param * F.relu(x)
    elif name in ['tanh']:
        if Param in ['default']:
            return lambda x:torch.tanh(x)
        else:
            return lambda x:Param * F.tanh(x)
    elif name in ['relu_tanh']:
        if Param in ['default']:
            return lambda x:F.relu(torch.tanh(x))
        else:
            return lambda x:Param * F.relu(torch.tanh(x))
    else:
        raise Exception('Invalid act func name: %s'%name)

def build_optimizer(dict_, Params=None, model=None, load=False):
    Type_ = GetFromDict(dict_, 'Type', default='sgd', write_default=True)
    #func = dict_['func'] #forward ; rec, output, input
    #lr = GetFromDict(dict_, 'lr', default=1.0e-3, write_default=True)
    lr = dict_['lr']
    weight_decay = GetFromDict(dict_, 'weight_decay', default=0.0, write_default=True)
    if Params is not None:
        pass
    elif model is not None:
        if hasattr(model, 'GetParam_to_train'):
            Params = model.GetParam_to_train()
        else:
            Params = model.Parameters()
    else:
        raise Exception('build_optimizer: Both Params and model are None.')
    
    if Type_ in ['adam', 'ADAM']:
        optimizer = optim.Adam(Params, lr=lr, weight_decay=weight_decay)
    elif Type_ in ['rmsprop', 'RMSProp']:
        optimizer = optim.RMSprop(Params, lr=lr, weight_decay=weight_decay)
    elif Type_ in ['sgd', 'SGD']:
        momentum = dict_.setdefault('momentum', 0.9)
        optimizer = optim.SGD(Params, momentum=momentum, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception('build_optimizer: Invalid optimizer Type: %s'%Type_)

    if load:
        optimizer.load_state_dict(dict_['state_dict'])
    else:
        dict_['state_dict'] = optimizer.state_dict()
    return optimizer

def build_scheduler(dict_, optimizer, load=False, verbose=True):
    #lr_decay = dict['lr_decay']
    scheduler_dict = dict_['scheduler']
    scheduler_Type = search_dict(scheduler_dict, ['Type', 'Method'], default='None', write_default=True)
    if verbose:
        print('build_scheduler: scheduler_Type: %s'%scheduler_Type)
    if scheduler_Type is None or scheduler_Type in ['None', 'none']:
        scheduler = None
        #update_lr = update_lr_none
    elif scheduler_Type in ['exp']:
        decay = search_dict(scheduler_dict, ['decay', 'coeff'], default=0.98, write_default=True, write_default_dict='decay')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)
        #update_lr = update_lr_
    elif scheduler_Type in ['stepLR', 'exp_interval']:
        decay = search_dict(scheduler_dict, ['decay', 'coeff'], default=0.98, write_default=True, write_default_key='decay')
        step_Size = search_dict(scheduler_dict, ['interval', 'step_Size'], default=0.98, write_default=True, write_default_key='decay')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, step_Size=step_Size, gamma=decay)
        #update_lr = update_lr_
    elif scheduler_Type in ['Linear', 'linear']:
        milestones = search_dict(scheduler_dict, ['milestones'], throw_none_error=True)
        scheduler = LinearLR(optimizer, milestones=milestones, epoch_num=dict_['epoch_num'])
        #update_lr = update_lr_
    else:
        raise Exception('build_scheduler: Invalid lr decay Method: '+str(scheduler_Type))
    return scheduler

# search for directory or file of most recently saved models(model with biggest epoch index)
def Getlast_model(model_prefix, base_dir=None, is_dir=True):
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
    # output: [batch_Size, num_class]; label: [batch_Size], label[i] is the index of correct category of i_th batch.
    correct_num = (torch.max(output, dim=1)[1]==label).sum().item()
    sample_num = label.Size(0)
    #return {'correct_num':correct_num, 'data_num':label_num} 
    return correct_num, sample_num

def scatter_label(label, num_class=None, device=None): # label: must be torch.LongTensor, shape: [batch_Size], label[i] is the index of correct category of i_th batch.
    #print('aaa')
    if num_class is None:
        #print(torch.max(label).__class__)
        num_class = torch.max(label).item() + 1
    scattered_label = torch.zeros((label.Size(0), num_class), device=device).to(label.device).scatter_(1, torch.unsqueeze(label, 1), 1)
    #return scattered_label.long() # [batch_Size, num_class]
    #print(label.Type())
    #print(scattered_label.Type())
    return scattered_label # [batch_Size, num_class]

def build_mlp(dict_, load=False, device=None):
    print(dict_)
    return MLP(dict_, load=load, device=device)

class MLP(nn.Module):
    def __init__(self, dict_, load=False, device=None):#InputNum is neuron_num.
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
        self.Params = {}
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
                #for Parameter in self.Parameters():
                #    print(Parameter.Size())
                init_weight(weight, self.init_weight)
                #self.register_Parameter(name='w%d'%layer_index, Param=weight)
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
                #self.register_Parameter('b%d'%self.layer_num, bias)
            else:
                self.biases.append(0.0)
            
        for index, weight in enumerate(self.weights):
            name = 'w%d'%index
            self.register_Parameter(name=name, Param=weight)
            self.Params[name] = weight

        if self.use_bias:
            for index, bias in enumerate(self.biases):
                name = 'b%d'%index
                if isinstance(bias, torch.Tensor):
                    self.register_Parameter(name, bias)
                self.Params[name] = bias
        
        if not (self.layer_num == 1 and not self.dict['act_func_on_last_layer']):
            self.act_func = Getact_func(self.dict['act_func'])

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
        #print(self.Parameters())

        #print_model_Param(self)
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
                raise Exception('Invalid bias Type: %s'%bias.__class__)
        if verbose:
            print(result)
        return result
    def plot_weight(self, axes=None, save=False, save_path='./', save_name='mlp_weight_plot.png'):
        row_num, col_num = Getrow_col()
        if axes is None:
            fig, axes = plt.subplots(nrows=row_num, ncols=col_num, figSize=(5*col_num, 5*row_num))
        if save:
            EnsurePath(save_path)
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
    act_func = Getact_func_module(dict_['act_func'])
    Layers = []
    layer_dicts = []
    N_nums = dict_['N_nums'] #InputNum, hidden_layer1_unit_num, hidden_layer2_unit_numm ... OutputNum
    layer_num = len(N_nums) - 1
    for layer_index in range(layer_num):
        current_layer = nn.Linear(N_nums[layer_index], N_nums[layer_index+1], bias=dict['bias'])
        Layers.append(current_layer)
        layer_dicts.append(current_layer.state_dict())
        if not (dict_['act_func_on_last_layer'] and layer_index==layer_num-1):
            Layers.append(act_func)
    dict['layer_dicts'] = layer_dicts
    return torch.nn.Sequential(*Layers)
'''

def build_mlp_sequential(dict_, load=False):
    act_func = Getact_func_module(dict_['act_func'])
    use_bias = dict_['bias']
    Layers = []
    layer_dicts = []
    N_nums = dict_['N_nums'] #InputNum, hidden_layer1_unit_num, hidden_layer2_unit_numm ... OutputNum
    layer_num = len(N_nums) - 1
    act_func_on_last_layer = GetFromDict(dict_, 'act_func_on_last_layer', default=True, write_default=True)
    for layer_index in range(layer_num):
        layer_current = nn.Linear(N_nums[layer_index], N_nums[layer_index+1], bias=use_bias)
        if load:
            layer_current.load_state_dict(dict_['layer_dicts'][layer_index])
        Layers.append(layer_current)
        layer_dicts.append(layer_current.state_dict())
        if not (act_func_on_last_layer and layer_index==layer_num-1):
            Layers.append(act_func)
    return torch.nn.Sequential(*Layers)

def print_model_Param(model):
    for name, Param in model.named_Parameters():
        print(Param)
        print('This is my %s. Size:%s is_leaf:%s device:%s requires_grad:%s'%
            (name, list(Param.Size()), Param.is_leaf, Param.device, Param.requires_grad))

def Gettensor_info(tensor, name='', verbose=True, complete=True):
    print(tensor.device)
    report = '%s...\033[0;31;40mVALUE\033[0m\n'%str(tensor)
    if complete:
        report += '%s...\033[0;31;40mGRADIENT\033[0m\n'%str(tensor.grad)
    report += 'Tensor \033[0;32;40m%s\033[0m: Size:%s is_leaf:%s device:%s Type:%s requires_grad:%s'%\
        (name, list(tensor.Size()), tensor.is_leaf, tensor.device, tensor.Type(), tensor.requires_grad)
    if verbose:
        print(report)
    return report

def Gettensor_stat(tensor, verbose=False):
    return {
        "min": torch.min(tensor),
        "max": torch.max(tensor),
        "mean": torch.mean(tensor),
        "std": torch.std(tensor),
        "var": torch.var(tensor)
    }

def print_optimizer_Params(optimizer):
    dict_ = optimizer.state_dict()
    for key, value in dict_.items():
        print('%s: %s'%(key, value))

