import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils import get_args, get_name, get_name_args, get_from_dict, search_dict, contain

def init_weight(weight, init_info, cons_method=None):
    num_dim = len(list(weight.size()))
    name, args = get_name_args(args)
    #print('weight:%s init_method:%s'%(weight_name, str(name)))
    coeff = args
    if cons_method is None:
        if isinstance(args, dict):
            cons_method = args.setdefault('cons_method', 'abs')
        else:
            cons_method = 'abs'
    sig = False
    if num_dim==1: # 1d tensor
        divider = weight.size(0)
        sig = True
    elif num_dim==2: # 2d tensor
        init_weight_2d(weight, init_info, cons_method)
    else:
        raise Exception('init_weight: %dd tensor is not supported.'%num_dim)
    
    if sig:
        lim = coeff / divider
        if cons_method in ['force']:
            torch.nn.init.uniform_(weight, 0.0, lim)
        else:
            torch.nn.init.uniform_(weight, -lim, lim)
def init_weight_1d():
    return
def init_weight_2d(weight, init_info, cons_method=None): # weight: [input_num, output_num]
    name, args = get_name_args(init_info)
    coeff = 1.0
    init_method = None
    if isinstance(args, float):
        coeff = args
    if isinstance(name, list):
        if contain(name, 'input') and contain(name, 'output'):
            # to be implemented
            pass
        elif contain(name, 'input'):
            if cons_method is None:
                lim_l = - coeff * 1.0 / weight.size(0)
                lim_r = - lim_l
            elif cons_method in ['abs']:
                lim_l = 0.0
                lim_r = 2 * coeff / weight.size(0)
            else:
                raise Exception('Invalid')            
            if contain(name, 'uniform'):
                init_method = 'uniform'
            elif contain(name, 'gaussian', 'normal'):
                init_method = 'normal'
            else:
                init_method = 'uniform' # default distribution

        elif contain(name, 'output'):
        else:
            
    if init_method is not None:
        if init_method in ['']:
        elif init_method in ['']:

        else:


    if name in ['output']:
        divider = weight.size(1)
        sig = True
    elif name in ['input']:
        1 / weight.size(0)
        sig = True
    elif name in ['ortho', 'orthogonal']:
        weight_ = weight.detach().clone()
        torch.nn.init.orthogonal_(weight_, gain=1.0) # init input weight to be orthogonal.
        with torch.no_grad():  # avoid gradient calculation error during in-place operation.
            weight.copy_( weight_ * coeff )
        return
    elif name in ['glorot', 'glorot_uniform', 'xavier', 'xavier_uniform']:
        weight_ = weight.detach().clone()
        torch.nn.init.xavier_uniform_(weight_, gain=1.0)
        with torch.no_grad():
            weight.copy_( weight_ * coeff )
        return
    else:
        raise Exception('Unsupported init method: %s'%name)

    if sig:
        lim = coeff / divider
        if cons_method in ['force']:
            torch.nn.init.uniform_(weight, 0.0, lim)
        else:
            torch.nn.init.uniform_(weight, -lim, lim)   

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

def get_loss_func(loss_func_str):
    if loss_func_str in ['MSE', 'mse']:
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
        if param=='default':
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
        optimizer = optim.SGD(params, momentum=0, lr=lr, weight_decay=weight_decay)

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
    if num_class is None:
        #print(torch.max(label).__class__)
        num_class = torch.max(label).item() + 1
    scattered_label = torch.zeros((label.size(0), num_class), device=device).to(label.device).scatter_(1, torch.unsqueeze(label, 1), 1)
    #return scattered_label.long() # [batch_size, num_class]
    return scattered_label # [batch_size, num_class]

class MLP(nn.Module):
    def __init__(self, dict_, load=False):#input_num is neuron_num.
        super(MLP, self).__init__()
        self.dict = dict_
        self.act_func_on_last_layer = self.dict.setdefault('act_func_on_last_layer', False)
        self.bias = self.dict.setdefault('bias', False)
        self.bias_on_last_layer = self.dict.setdefault('bias_on_last_layer', False)
        self.N_nums = self.dict['N_nums']
        self.layer_num = self.dict['layer_num'] = len(self.N_nums) - 1

        if load:
            self.weights = self.dict['weights']
            self.bias = self
        else:
            self.weights.append
            for layer_index in range(self.layer_num):
                weight = torch.

        self.act_func = get_act_func

        self.mlp = build_mlp_sequential(dict_=self.dict, load=load)
        print(self.mlp.parameters)
        print(self.mlp.parameters())
        

    def forward(self, x):
        for layer_index in self.layer_num:
            x = torch.mm(x, self.weights[layer_index]) + self.bias[layer_index] 
            
        if self.act_func_on_last_layer:
            x = self.act_func(x)

        return x
    def anal_weight_change():
        
def build_mlp(dict_, load=False):
    return MLP(dict_, load=load)

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