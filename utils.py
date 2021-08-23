import os
import re
import sys
import functools
import threading
import time
import math
import cmath
import warnings
import json
import jsons
import json5
import jsonpickle
import pickle
import importlib
from typing import Iterable, List
#import pynvml
#from pynvml.nvml import nvmlDeviceOnSameBoard
from types import SimpleNamespace
from numpy import linalg
import timeout_decorator
import numpy as np
import torch
import torch.nn as nn

import matplotlib as mpl
from matplotlib import pyplot as plt

from utils_torch.json import *
from utils_torch.attrs import *

def add_log(log, time_stamp=True):
    logger = utils_torch.args_global.logger
    if logger is not None:
        if time_stamp:
            logger.debug("[%s]%s"%(get_time(), log))
        else:
            logger.debug("%s"%log)

def set_logger(logger):
    utils_torch.args_global.logger = logger

def get_time(format="%Y-%m-%d %H:%M:%S", verbose=False):
    time_str = time.strftime(format, time.localtime()) # Time display style: 2016-03-20 11:45:39
    if verbose:
        print(time_str)
    return time_str

def compose_function(*function):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), function, lambda x: x)

def contain(list, items):
    if isinstance(items, Iterable) and not isinstance(items, str): # items is a list
        sig = False
        for item in items:
            if item in list:
                sig = True
                break
        return sig
    else: # items is uniterable object
        if items in list:
            return True
        else:
            return False

def contain_all(list_, items):
    if isinstance(items, Iterable): # items is a list
        sig = True
        for item in items:
            if item in list_:
                continue
            else:
                sig = False
                break
        return sig
    else: # items is uniterable object
        if items in list_:
            return True
        else:
            return False    

def get_row_col(tot_num, row_num=None, col_num=None):
    if row_num is None and col_num is not None:
        row_num = tot_num // col_num
        if tot_num % col_num > 0:
            row_num += 1
        return row_num, col_num
    elif row_num is not None and col_num is None:
        col_num = tot_num // row_num
        if tot_num % row_num > 0:
            col_num += 1
        return row_num, col_num
    else:
        if tot_num != row_num * col_num:
            raise Exception('get_row_col: cannot simultaneouly fit row_num %d and col_num %d'%(row_num, col_num))
        else:
            return row_num, col_num

def get_ax(axes, row_index, col_index, row_num, col_num):
    if row_num==1: # deal with degraded cases where col_num or row_num is 1.
        if col_num>1:
            ax = axes[col_index]
        else:
            ax = axes
    else:
        ax = axes[row_index, col_index]
    return ax

#@timeout_decorator.timeout(15)
def timeout_(timeout=5, daemon_threads_target=[], daemon_threads_args=[], daemon_threads_kwargs=[]):
    for target, args, kw in zip(daemon_threads_target, daemon_threads_args, daemon_threads_kwargs):
        thread = threading.Thread(target=target, args=args, kwargs=kw)
        thread.setDaemon(True)
        thread.start()
    time.sleep(timeout)

def get_best_gpu(timeout=15, default_device='cuda:0'):
    dict_ = {'device':None}
    timeout_thread = threading.Thread(target=timeout_, args=(timeout, [get_best_gpu_], [()], [{'dict_':{}}]))
    timeout_thread.start()
    timeout_thread.join()

    if dict_['device'] is None:
        if default_device is None:
            raise Exception('get_best_gpu: Time Out.')
        else:
            warnings.warn('get_best_gpu: Time out. Using default device.')
            return default_device
    else:
        return dict_['device']

def get_best_gpu_(dict_={}): # return torch.device with largest available gpu memory.
    '''
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    deviceMemory = []
    #print('aaa')
    for i in range(deviceCount):
        #print(i)
        handle = pynvml.nvmlDeviceGetHandleByIndex(i) # sometimes stuck here.
        # print('GPU', i, ':', pynvml.nvmlDeviceGetName(handle))
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        deviceMemory.append(mem_info.free)
        #print(i)
    deviceMemory = np.array(deviceMemory, dtype=np.int64)
    best_device_index = np.argmax(deviceMemory)    
    dict_['device'] = 'cuda:%d'%(best_device_index)
    '''
    return dict_
    
def split_batch(data, batch_size): #data:(batch_size, image_size)
    sample_num = data.size(0)
    batch_sizes = [batch_size for _ in range(sample_num // batch_size)]
    if not sample_num % batch_size==0:
        batch_sizes.apend(sample_num % batch_size)
    return torch.split(data, section=batch_sizes, dim=0)

def cat_batch(dataloader): #data:(batch_num, batch_size, image_size)
    if not isinstance(dataloader, list):
        dataloader = list(dataloader)
    '''
    print(len(dataloader))
    print(len(dataloader[0]))
    print(dataloader[0].size())
    print(dataloader[0].__class__.__name__)
    print(dataloader[0][0].size())
    print(dataloader[0][0].__class__.__name__)
    '''
    return torch.cat(dataloader, dim=0)

def read_data(read_dir): #read data from file.
    if not os.path.exists(read_dir):
        return None
    f = open(read_dir, 'rb')
    data = torch.load(f)
    f.close()
    return data

def save_data(data, save_path): # save data into given file path. existing file will be overwritten.
    f = open(save_path, 'wb')
    torch.save(data, f)
    f.close()

def ensure_path(path, is_folder=False): # check if given path exists. if not, create it.
    if is_folder: # caller of this function makes sure that path is a directory/folder.
        if not path.endswith('/'): # folder
            warnings.warn('%s is a folder, and should ends with /.'%path)
            path += '/'
            #print(path)
            #input()
        if not os.path.exists(path):
            os.makedirs(path)
    else: # path can either be a directory or a folder. If path exists, then it is what it is (file or folder). If not, depend on whether it ends with '/'.
        if os.path.exists(path): # path exists
            if os.path.isdir(path):
                if not path.endswith('/'): # folder
                    path += '/'     
            elif os.path.isfile(path):
                raise Exception('file already exists: %s'%str(path))
            else:
                raise Exception('special file already exists: %s'%str(path))
        else: # path does not exists
            if path.endswith('/'): # path is a folder
                path_strip = path.rstrip('/')
            else:
                path_strip = path
            if os.path.exists(path_strip): # folder with same name exists
                raise Exception('ensure_path: homonymous file exists.')
            else:
                if not os.path.exists(path_strip):
                    os.makedirs(path_strip)
                    #os.mkdir(path) # os.mkdir does not support creating multi-level folders.
                #filepath, filename = os.path.split(path)
    return path

ensure_dir = ensure_path # sometimes we don't distinguish between dir(ectory) and path.

def cal_path_from_main(path_rel=None, path_start=None, path_main=None):
    # path_rel: file path relevant to path_start
    if path_main is None:
        path_main = sys.path[0]
    if path_start is None:
        path_start = path_main
        warnings.warn('cal_path_from_main: path_start is None. using default: %s'%path_main)
    path_start = os.path.abspath(path_start)
    path_main = os.path.abspath(path_main)
    if os.path.isfile(path_main):
        path_main = os.path.dirname(path_main)
    if not path_main.endswith('/'):
        path_main += '/' # necessary for os.path.relpath to calculate correctly
    if os.path.isfile(path_start):
        path_start = os.path.dirname(path_start)
    #path_start_rel = os.path.relpath(path_start, start=path_main)

    if path_rel.startswith('./'):
        path_rel.lstrip('./')
    elif path_rel.startswith('/'):
        raise Exception('path_rel: %s is a absolute path.'%path_rel)
    
    path_abs = os.path.abspath(os.path.join(path_start, path_rel))
    #file_path_from_path_start = os.path.relpath(path_rel, start=path_start)
    
    path_from_main = os.path.relpath(path_abs, start=path_main)

    #print('path_abs: %s path_main: %s path_from_main: %s'%(path_abs, path_main, path_from_main))
    '''
    print(main_path)
    print(path_start)
    print('path_start_rel: %s'%path_start_rel)
    print(file_name)
    print('file_path: %s'%file_path)
    #print('file_path_from_path_start: %s'%file_path_from_path_start)
    print('file_path_from_main_path: %s'%file_path_from_main_path)
    print(target_path_module(file_path_from_main_path))
    '''
    #print('path_rel: %s path_start: %s path_main: %s'%(path_rel, path_start, path_main))
    return path_from_main

def import_module(module_path):
    return importlib.import_module(module_path)

def import_file(file_from_sys_path):
    if not os.path.isfile(file_from_sys_path):
        raise Exception("%s is not a file."%file_from_sys_path)
    if file_from_sys_path.startswith("/"):
        raise Exception("import_file: file_from_sys_path must not be absolute path.")
    if file_from_sys_path.startswith("./"):
        module_path = file_from_sys_path.lstrip("./")
    module_path = module_path.replace("/", ".")
    return importlib.import_module(module_path)

def get_sys_type():
    if re.match(r'win',sys.platform) is not None:
        sys_type = 'windows'
    elif re.match(r'linux',sys.platform) is not None:
        sys_type = 'linux'
    else:
        sys_type = 'unknown'
    return sys_type

def get_name(param): # a mechanism supporting name and args given in different types. a parameter consist of a name of type str and optional args.
    if isinstance(param, dict):
        name = search_dict(param, ['name', 'method', 'type'])
        return name
    elif isinstance(param, list):
        return param[0]
    elif isinstance(param, str):
        return param
    else:
        raise Exception('Invalid param type:' +str(type(param)))

def get_args(param): # a mechanism supporting name and args given in different types. a parameter consist of a name of type str and optional args.
    #print(type(param))
    #print(param)
    if isinstance(param, dict):
        args = param.get('args')
        if args is not None:
            return args
        else:
            return param
    elif isinstance(param, list):
        return param[1]
    elif isinstance(param, str):
        #return 1.0 #using default coefficient: 1.0
        return None
    else:
        raise Exception('Invalid param type:' +str(type(param)))

def get_name_args(param):
    return get_name(param), get_args(param)

get_arg = get_args

def get_items_from_dict(dict_, keys):
    items = []
    for name in keys:
        items.append(dict_[name])
    if len(items) == 1:
        return items[0]
    else:
        return tuple(items)   

def get_from_dict(dict_, key, default=None, write_default=False, throw_keyerror=True):
    # try getting dict[key]. 
    # if key exists, return dict[key].
    # if no such key, return default, and set dict[key]=default if write_default==True.
    '''
    if isinstance(key, list):
        items = []
        for name in key:
            items.append(dict_[name])
        return tuple(items)
    else:
    '''
    if dict_.get(key) is None:
        if write_default:
            dict_[key] = default
        else:
            if default is not None:
                return default
            else:
                if throw_keyerror:
                    raise Exception('KeyError: dict has not such key: %s'%str(key))
                else:
                    return None
    return dict_[key]

def search_dict(dict_, keys, default=None, write_default=False, write_default_key=None, throw_multi_error=False, throw_none_error=True):
    values = []
    for key in keys:
        value = dict_.get(key)
        if value is not None:
            values.append(value)
    if len(values)>1:
        if throw_multi_error:
            raise Exception('search_dict: found multiple keys.')
        else:
            return values
    elif len(values)==0:
        if default is not None:
            if write_default:
                if write_default_key is None: # write value to all keys
                    for key in keys:
                        dict_[key] = default
                else:
                    dict_[write_default_key] = default
            return default
        if throw_none_error:
            raise Exception('search_dict: no keys matched.')
        else:
            return None
    else:
        return values[0]

def write_dict_info(dict_, save_path='./', save_name='dict info.txt'): # write readable dict info into file.
    values_remained = []
    with open(save_path+save_name, 'w') as f:
        for key in dict_.keys():
            value = dict_[value]
            if isinstance(value, str) or isinstance(value, int):
                f.write('%s: %s'%(str(key), str(value)))
            else:
                values_remained.append([key, value])

def print_torch_info(): # print info about training environment, global variables, etc.
    torch.pytorch_info()
    #print('device='+str(device))

def get_act_func_module(act_func_info):
    name = get_name(act_func_info)
    if name in ['relu']:
        return nn.ReLU()
    elif name in ['tanh']:
        return nn.Tanh()
    elif name in ['softplus']:
        return nn.Softplus()
    elif name in ['sigmoid']:
        return nn.Sigmoid()

def trunc_prefix(string, prefix):
    if(string[0:len(prefix)]==prefix):
        return string[len(prefix):len(string)]
    else:
        return string

def update_key(dict_0, dict_1, prefix='', strip=False, strip_only=True, exempt=[]):
    if not strip:
        for key in dict_1.keys():
            dict_0[prefix+key]=dict_1[key]
    else:
        for key in dict_1.keys():
            trunc_key=trunc_prefix(key, prefix)
            if strip_only:
                if(trunc_key!=key or key in exempt):
                    dict_0[trunc_key]=dict_1[key]
            else:
                dict_0[trunc_key]=dict_1[key]

def get_np_stat(data, verbose=False):
    return {
        "min": np.min(data),
        "max": np.max(data),
        "mean": np.mean(data),
        "std": np.std(data),
        "var": np.var(data)
    }


def plot_train_curve(stat_dir, loss_only=False):
    try:
        f=open(stat_dir, 'rb')
        train_loss_list=pickle.load(f)
        train_acc_list=pickle.load(f)
        val_loss_list=pickle.load(f)
        val_acc_list=pickle.load(f)
        f.close()
        if(loss_only==False):
            plot_training_curve_0(train_loss_list, train_acc_list, val_loss_list, val_acc_list)
        else:
            plot_training_curve_1(train_loss_list, val_loss_list)
    except Exception:
        print('exception when printing training curve.')

def plot_train_curve_0(train_loss_list, train_acc_list, val_loss_list, val_acc_list, fontsize=40):
    print('plotting training curve.')
    x = range(epoch_start, epoch_num)
    plt.subplot(2, 1, 1)
    plt.plot(x, train_loss_list, '-', label='train', color='r')
    plt.plot(x, val_loss_list, '-', label='test', color='b')
    plt.title('Loss', fontsize=fontsize)
    plt.ylabel('loss', fontsize=fontsize)
    plt.xlabel('epoch', fontsize=fontsize)
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(x, train_acc_list, '-', label='train', color='r')
    plt.plot(x, val_acc_list, '-', label='test', color='b')
    plt.title('Accuracy', fontsize=fontsize)
    plt.ylabel('acc', fontsize=fontsize)
    plt.xlabel('epoch', fontsize=fontsize)
    plt.legend(loc='best')
    plt.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
    plt.savefig(save_path_anal+'training_curve.jpg')
    plt.close()

def plot_train_curve_1(train_loss_list, val_loss_list, fontsize=40):
    print('plotting training curve.')
    fig = plt.figure()
    x = range(len(train_loss_list))
    plt.plot(x, train_loss_list, '-', label='train', color='r')
    x = range(len(val_loss_list))
    plt.plot(x, val_loss_list, '-', label='test', color='b')
    plt.title('Loss - epoch', fontsize=fontsize)
    plt.ylabel('loss', fontsize=fontsize)
    plt.xlabel('epoch', fontsize=fontsize)
    plt.legend(loc='best')
    plt.close()


def set_instance_attr(self, dict_, keys=None, exception=[]):
    if keys is None: # set all keys as instance variables.
        for key, value in dict_.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key not in exception:
                setattr(self, key, value)
    else: # set values of designated keys as instance variables.
        for key, value in dict_.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key in keys:
                if key not in exception:
                    setattr(self, key, value)

set_instance_variable = set_instance_attr

def set_dict_variable(dict_1, dict_0, keys=None, exception=['self']): # dict_1: target. dict_0: source.
    if keys is None: # set all keys as instance variables.
        for key, value in dict_0.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key not in exception:
                dict_1[key] = value
    else: # set values of designated keys as instance variables.
        for key, value in dict_0.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key in keys:
                if key not in exception:
                    dict_1[key] = value
        
def set_instance_variable_and_dict(self, dict_1, dict_0, keys=None, exception=['self']): # dict_0: source. dict_1: target dict. self: target class object.
    if keys is None: # set all keys as instance variables.
        for key, value in dict_0.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key not in exception:
                dict_1[key] = value
                setattr(self, key, value)
    else: # set values of designated keys as instance variables.
        for key, value in dict_0.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key in keys:
                if key not in exception:
                    dict_1[key] = value
                    setattr(self, key, value)
                
def set_default_attr(self, key, value):
    if self.__dict__.get(key) is None:
        setattr(self, key, value)

set_dict_and_instance_variable = set_class_variable_and_dict = set_instance_variable_and_dict

class Param:
    pass
def load_param(dict_, exception=[], default_exception=['kw', 'param', 'key', 'item'], use_default_exception=True):
    param = Param()
    for key, item in dict_.items():
        if key not in exception:
            if use_default_exception:
                if key not in default_exception:
                    setattr(param, key, item)
            else:
                setattr(param, key, item)
    return param

def print_dict(dict_):
    for key, items in dict_.items():
        print('%s=%s'%(str(key), str(items)), end=' ')
    print('\n')


def get_last_model(model_prefix, base_dir='./', is_dir=True):
    # search for directory or file of most recently saved models(model with biggest epoch index)
    if is_dir:
        max_epoch = None
        pattern = model_prefix+'(\d*)'
        dirs = os.listdir(base_dir)
        for dir_name in dirs:
            result = re.search(r''+pattern, dir_name)
            if result is not None:
                try:
                    epoch_num = int(result.group(1))
                except Exception:
                    print('error in matching model name.')
                    continue
                if(max_epoch is None):
                    max_epoch = epoch_num
                else:
                    if(max_epoch < epoch_num):
                        max_epoch = epoch_num
    if max_epoch is not None:
        return base_dir + model_prefix + str(max_epoch) + '/'
    else:
        return "error"

def standardize_suffix(suffix):
    pattern = re.compile(r'\.?(\w+)')
    result = pattern.match(suffix)
    if result is None:
        raise Exception('check_suffix: %s is illegal suffix.'%suffix)
    else:
        suffix = result.group(1)
    return suffix

def ensure_suffix(name, suffix):
    if not suffix.startswith("."):
        suffix = "." + suffix
    if name.endswith(suffix):
        return suffix
    else:
        return name + suffix

def check_suffix(name, suffix=None, is_path=True):
    # check whether given file name has suffix. If true, check whether it's legal. If false, add given suffix to it.
    if suffix is not None:
        if isinstance(suffix, str):
            suffix = standardize_suffix(suffix)
        elif isinstance(suffix, list):
            for i, suf_ in enumerate(suffix):
                suffix[i] = standardize_suffix(suf_)
            if len(suffix)==0:
                suffix = None
        else:
            raise Exception('check_suffix: invalid suffix: %s'%(str(suffix)))      

    pattern = re.compile(r'(.*)\.(\w+)')
    result = pattern.match(name)
    if result is not None: # match succeeded
        name = result.group(1)
        suf = result.group(2)
        if suffix is None:
            return name + '.' + suf
        elif isinstance(suffix, str):
            if name==suffix:
                return name
            else:
                warnings.warn('check_suffix: %s is illegal suffix. replacing it with %s.'%(suf, suffix))
                return name + '.' + suffix
        elif isinstance(suffix, list):
            sig = False
            for suf_ in suffix:
                if suf==suf_:
                    sig = True
                    return name
            if not sig:
                warnings.warn('check_suffix: %s is illegal suffix. replacing it with %s.'%(suf, suffix[0]))
                return name + '.' + suffix[0]                
        else:
            raise Exception('check_suffix: invalid suffix: %s'%(str(suffix)))
    else: # fail to match
        if suffix is None:
            raise Exception('check_suffix: %s does not have suffix.'%name)
        else:
            if isinstance(suffix, str):
                suf_ = suffix
            elif isinstance(suffix, str):
                suf_ = suffix[0]
            else:
                raise Exception('check_suffix: invalid suffix: %s'%(str(suffix)))
            warnings.warn('check_suffix: no suffix found in %s. adding suffix %s.'%(name, suffix))            
            return name + '.' + suf_

def remove_suffix(name, suffix='.py', must_match=False):
    pattern = re.compile(r'(.*)%s'%suffix)
    result = pattern.match(name)
    if result is None:
        if must_match:
            raise Exception('%s does not have suffix %s'%(name, suffix))
        else:
            return name
    else:
        return result.group(1)

def scan_files(path, pattern, ignore_folder=True, raise_not_found_error=False):
    if not path.endswith('/'):
        path.append('/')
    files_path = os.listdir(path)
    matched_files = []
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    for file_name in files_path:
        #print(file_name)
        if pattern.match(file_name) is not None:
            if os.path.isdir(path + file_name):
                if ignore_folder:
                    matched_files.append(file_name)
                else:
                    warnings.warn('%s is a folder, and will be ignored.'%(path + file))
            else:
                matched_files.append(file_name)
    
    if raise_not_found_error:
        if len(matched_files)==0:
            raise Exception('scan_files: cannot find any files that match pattern %s'%pattern)

    return matched_files

def copy_files(file_list, source_path='./', target_path=None, sys_type='linux'):
    if not source_path.endswith('/'):
        source_path += '/'

    if not target_path.endswith('/'):
        target_path += '/'

    ensure_path(target_path)

    '''
    if subpath is not None:
        if not subpath.endswith('/'):
             subpath += '/'
        path += subpath
    ensure_path(path)
    '''
    #print(target_path)
    if sys_type in ['linux']:
        for file in file_list:
            file = file.lstrip('./')
            file = file.lstrip('/')
            #print(path)
            #print(file)
            #shutil.copy2(file, dest + file)
            #print(source_path + file)
            #print(target_path + file)
            ensure_path(os.path.dirname(target_path + file))
            if os.path.exists(target_path + file):
                os.system('rm -r %s'%(target_path + file))
            #print('cp -r %s %s'%(file_path + file, path + file))
            os.system('cp -r %s %s'%(source_path + file, target_path + file))
    elif sys_type in ['windows']:
        # to be implemented 
        pass
    else:
        raise Exception('copy_files: Invalid sys_type: '%str(sys_type))

def join_path(path_0, path_1):
    if not path_0.endswith('/'):
        path_0 += '/'
    if path_1.startswith('./'):
        path_1 = path_1.lstrip('./')
    if path_1.startswith('/'):
        raise Exception('join_path: path_1 is a absolute path: %s'%path_1)
    return path_0 + path_1
def target_path_module(path):
    path = path.lstrip('./')
    path = path.lstrip('/')
    if not path.endswith('/'):
        path += '/'
    path =  path.replace('/','.')
    return path

def select_file(name, candidate_files, default_file=None, match_prefix='', match_suffix='.py', file_type='', raise_no_match_error=True):
    use_default_file = False
    perfect_match = False
    if name is None:
        use_default_file = True
    else:
        matched_count = 0
        matched_files = []
        perfect_match_name = None
        if match_prefix + name + match_suffix in candidate_files: # perfect match. return this file directly
            perfect_match_name = match_prefix + name + match_suffix
            perfect_match = True
            matched_files.append(perfect_match_name)
            matched_count += 1
        for file_name in candidate_files:
            if name in file_name:
                if file_name!=perfect_match_name:
                    matched_files.append(file_name)
                    matched_count += 1
        #print(matched_files)
        if matched_count==1: # only one matched file
            return matched_files[0]
        elif matched_count>1: # multiple files matched
            warning = 'multiple %s files matched: '%file_type
            for file_name in matched_files:
                warning += file_name
                warning += ' '
            warning += '\n'
            if perfect_match:
                warning += 'Using perfectly matched file: %s'%matched_files[0]
            else:
                warning += 'Using first matched file: %s'%matched_files[0]
            warnings.warn(warning)
            return matched_files[0]
        else:
            warnings.warn('No file matched name: %s. Trying using default %s file.'%(str(name), file_type))
            use_default_file = True
    if use_default_file:
        if default_file is not None:
            if default_file in candidate_files:
                print('Using default %s file: %s'%(str(file_type), default_file))
                return default_file
            else:
                sig = True
                for candidate_file in candidate_files:
                    if default_file in candidate_file:
                        print('Using default %s file: %s'%(str(file_type), candidate_file))
                        sig = False
                        return candidate_file
                if not sig:
                    if raise_no_match_error:
                        raise Exception('Did not find default %s file: %s'%(file_type, str(default_file)))
                    else:
                        return None
        else:
            if raise_no_match_error:
                raise Exception('Plan to use default %s file. But default %s file is not given.'%(file_type, file_type))
            else:
                return None
    else:
        return None

import hashlib
def get_md5(path, md5=None):
    if md5 is None:
        md5 = hashlib.md5()
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            bytes = f.read()
        md5.update(bytes)
        md5_str = md5.hexdigest()
        #print(md5_str.__class__)
        return md5_str
    else:
        warnings.warn('%s is not a file. '%path)
        return None

def visit_path(args=None, func=None, recur=False, path=None):
    if func is None:
        func = args.func   
    if path is None:
        path = args.path
    else:
        func = None
        warnings.warn('visit_dir: func is None.')
    filepaths=[]
    abspath = os.path.abspath(path) # relative path also works well
    for name in os.listdir(abspath):
        file_path = os.path.join(abspath, name)
        if os.path.isdir(file_path):
            if recur:
                visit_path(args=args, func=func, recur=True)
        else:
            func(args=args, file_path=file_path)
    return filepaths

visit_dir = visit_path


def copy_folder(source_path, target_path, exceptions=[], verbose=True):
    '''
    if args.path is not None:
        path = args.path
    else:
        path = '/data4/wangweifan/backup/'
    '''
    #ensure_path(source_path)
    ensure_path(target_path)
    
    for i in range(len(exceptions)):
        exceptions[i] = os.path.abspath(exceptions[i])
        if os.path.isdir(exceptions[i]):
            exceptions[i] += '/'

    source_path = os.path.abspath(source_path)
    target_path = os.path.abspath(target_path)

    if not source_path.endswith('/'):
        source_path += '/'
    if not target_path.endswith('/'):
        target_path += '/'

    if verbose:
        print('Copying folder from %s to %s. Exceptions: %s'%(source_path, target_path, exceptions))

    if source_path + '/' in exceptions:
        warnings.warn('copy_folder: neglected the entire root path. nothing will be copied')
        if verbose:
            print('neglected')
    else:
        copy_folder_recur(source_path, target_path, subpath='', exceptions=exceptions)

def copy_folder_recur(source_path, target_path, subpath='', exceptions=[], verbose=True):
    #ensure_path(source_path + subpath)
    ensure_path(target_path + subpath)
    items = os.listdir(source_path + subpath)
    for item in items:
        #print(target_path + subpath + item)
        path = source_path + subpath + item
        if os.path.isfile(path): # is a file
            if path + '/' in exceptions:
                if verbose:
                    print('neglected file: %s'%path)
            else:
                if os.path.exists(target_path + subpath + item):
                    md5_source = get_md5(source_path + subpath + item)
                    md5_target = get_md5(target_path + subpath + item)
                    if md5_target==md5_source: # same file
                        #print('same file')
                        continue
                    else:
                        #print('different file')
                        os.system('rm -r "%s"'%(target_path + subpath + item))
                        os.system('cp -r "%s" "%s"'%(source_path + subpath + item, target_path + subpath + item))     
                else:
                    os.system('cp -r "%s" "%s"'%(source_path + subpath + item, target_path + subpath + item))
        elif os.path.isdir(path): # is a folder.
            if path + '/' in exceptions:
                if verbose:
                    print('neglected folder: %s'%(path + '/'))
            else:
                copy_folder_recur(source_path, target_path, subpath + item + '/', verbose=verbose)
        else:
            warnings.warn('%s is neither a file nor a path.')

def prep_title(title):
    if title is None:
        title = ''
    else:
        if title.endswith(':'):
            title += ' '
        elif title.endswith(': '):
            pass
        else:
            title += ': '
    return title

def np_cosine_similarity(vecA, vecB):
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    #normA_ = np.sum(vecA ** 2) ** 0.5
    #normB_ = np.sum(vecB ** 2) ** 0.5
    consine_similarity = np.dot(vecA.T, vecB) / (normA * normB)
    return consine_similarity

def EmptyPyObj():
    return type('test', (), {})()

new_empty_object = EmptyPyObj

def RemoveAllFiles(path, verbose=True):
    if not os.path.exists(path):
        raise Exception()
    if not os.path.isdir(path):
        raise Exception()
    for file in ListAllFiles(path):
        file_path = os.path.join(path, file)
        os.remove(file_path)
        utils_torch.add_log("utils_pytorch: removed file: %s"%file_path)

def ListAllFiles(path):
    if not os.path.exists(path):
        raise Exception()
    if not os.path.isdir(path):
        raise Exception()
    items = os.listdir(path)
    files = []
    for item in items:
        if os.path.isfile(os.path.join(path, item)):
            files.append(item)
    return files

args_global = utils_torch.json.JsonObj2PyObj({
    "logger": None
})