import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils import get_args, get_name, get_name_args, get_from_dict, search_dict

def init_weight(weight, args, weight_name='unnamed', cons_method=None):
    dim_num = len(list(weight.size()))
    name, args = get_name_args(args)
    #print('weight:%s init_method:%s'%(weight_name, str(name)))
    coeff = args
    if cons_method is None:
        if isinstance(args, dict):
            cons_method = args.setdefault('cons_method', 'abs')
        else:
            cons_method = 'abs'

    sig = False
    if dim_num==1:
        divider = weight.size(0)
        sig = True
    elif dim_num==2:
        if name in ['output']:
            divider = weight.size(1)
            sig = True
        elif name in ['input']:
            divider = weight.size(0)
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

def get_act_func(act_func_des):
    if isinstance(act_func_des, list):
        act_func_name = act_func_des[0]
        act_func_param = act_func_des[1]
    elif isinstance(act_func_des, str):
        act_func_name = act_func_des
        act_func_param = 'default'
    elif isinstance(act_func_des, dict):
        act_func_name = act_func_des['name']
        act_func_param = act_func_des['param']
    return get_act_func_from_name(act_func_name, act_func_param)

def get_act_func_module(act_func_des):
    name = act_func_des
    if name in ['relu']:
        return nn.ReLU()
    elif name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['softplus', 'SoftPlus']:
        return nn.Softplus()
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()

def get_act_func_from_name(name='relu', param='default'):
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

def build_mlp(dict_, load=False):
    act_func = get_act_func_module(dict_['act_func'])
    use_bias = dict_['bias']
    layers = []
    layer_dicts = []
    unit_nums = dict_['unit_nums'] #input_num, hidden_layer1_unit_num, hidden_layer2_unit_numm ... output_num
    layer_num = len(unit_nums) - 1
    act_func_on_last_layer = get_from_dict(dict_, 'act_func_on_last_layer', default=True, write_default=True)
    for layer_index in range(layer_num):
        layer_current = nn.Linear(unit_nums[layer_index], unit_nums[layer_index+1], bias=use_bias)
        if load:
            layer_current.load_state_dict(dict_['layer_dicts'][layer_index])
        layers.append(layer_current)
        layer_dicts.append(layer_current.state_dict())
        if not (act_func_on_last_layer and layer_index==layer_num-1):
            layers.append(act_func)
    return torch.nn.Sequential(*layers)


'''
def build_mlp(dict_):
    act_func = get_act_func_module(dict_['act_func'])
    layers = []
    layer_dicts = []
    unit_nums = dict_['unit_nums'] #input_num, hidden_layer1_unit_num, hidden_layer2_unit_numm ... output_num
    layer_num = len(unit_nums) - 1
    for layer_index in range(layer_num):
        current_layer = nn.Linear(unit_nums[layer_index], unit_nums[layer_index+1], bias=dict['bias'])
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