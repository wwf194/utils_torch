import torch
import torch.nn as nn
import numpy as np
#from utils_model_ import get_tensor_info
import time
cache = {}
cache['time'] = time.time()
cache['count'] = 0
def print_time():
    time_1 = time.time()
    print('%d: %s'%(cache['count'], time_1 - cache['time']))
    cache['time'] = time_1
    cache['count'] += 1

print_time()
o = torch.zeros((100, 100), device=torch.device('cpu'), requires_grad=True)
print_time()

o = o.to(torch.device('cuda:2'))
print_time()

o = torch.zeros((100, 100), device=torch.device('cpu'), requires_grad=True)
print_time()

o = o.to(torch.device('cuda:8'))
print_time()

