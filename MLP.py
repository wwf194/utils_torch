import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_ import set_instance_attr
from utils_model_ import init_weight, get_ei_mask, get_mask, get_cons_func, get_act_func, build_mlp

#training parameters.
class MLP(nn.Module):
    def __init__(self, dict_, load=False):#input_num is neuron_num.
        super(MLP, self).__init__()
        self.dict = dict_
    
        self.mlp = build_mlp(dict=self.dict, load=load)



    def anal_weight_change()