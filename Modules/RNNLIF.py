import random

import numpy as np
from numpy import select, unravel_index

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib as mpl
from matplotlib import pyplot as plt

from utils_torch.attrs import *
import utils_torch

def InitFromParam(param):
    model = RNNLIF()
    model.InitFromParam(param)
    return model

def load_model(args):
    return 

class RNNLIF(nn.Module):
    # Singel-Layer Recurrent Neural Network with Leaky Integrate-and-Fire Dynamics
    def __init__(self, param=None, data=None, **kw):
        super(RNNLIF, self).__init__()
        utils_torch.model.InitForModel(self, param, data, ClassPath="utils_torch.Modules.RNNLIF", **kw)
    def InitFromParam(self, IsLoad=False):
        utils_torch.model.InitFromParamForModel(self, IsLoad)
        param = self.param
        data = self.data
        cache = self.cache
        
        if cache.IsInit:
            utils_torch.AddLog("RNNLIF: Initializing...")
        else:
            utils_torch.AddLog("RNNLIF: Loading...")
        CheckAttrs(param, "Type", value="RNNLIF")

        Neurons = param.Neurons
        EnsureAttrs(Neurons.Recurrent, "IsExciInhi", value=True)
        if GetAttrs(Neurons.Recurrent.IsExciInhi):
            EnsureAttrs(Neurons, "Recurrent.Excitatory.Ratio", default=0.8)

        cache.NeuronNum = Neurons.Recurrent.Num
        self.BuildModules()
        self.InitModules()

        EnsureAttrs(param, "InitTasks", default=[])
        for Task in self.param.InitTasks:
            utils_torch.DoTask(Task, ObjCurrent=self.cache, ObjRoot=utils_torch.GetGlobalParam())
        
        self.ParseRouters()

        if cache.IsInit:
            utils_torch.AddLog("RNNLIF: Initialized.")
        else:
            utils_torch.AddLog("RNNLIF: Loaded.")
    def GenerateZeroInitState(self, RefInput):
        data = self.data
        cache = self.cache
        BatchSize = RefInput.size(0)
        InitState = torch.zeros((BatchSize, cache.NeuronNum * 2), device=self.GetTensorLocation(), requires_grad=False)
        return InitState
    def anal_weight_change_(self):
        for name, value in self.named_parameters():
            #if name in ['encoder.0.weight','encoder.2.weight']:
            if True:
                #print('name: {0},\t grad: {1}'.format(name, value.requires_grad))
                #print(value)
                #print(value.grad)
                value_np = value.detach().cpu().Numpy()
                if self.cache.get(name) is not None:  
                    #print('  change in %s: '%name, end='')
                    #print(value_np - self.cache[name])
                    print('  ratio change in %s: '%name, end='')
                    print( np.sum(np.abs(value_np-self.cache[name])) / np.sum(np.abs(self.cache[name])) )
                self.cache[name] = value_np
    def anal_weight_change(self, verbose=True):
        result = ''
        r_1 = self.Getr().detach().cpu().Numpy()
        if self.cache.get('r') is not None:
            r_0 = self.cache['r']
            r_change_rate = np.sum(abs(r_1 - r_0)) / np.sum(np.abs(r_0))
            result += 'r_change_rate: %.3e '%r_change_rate
        self.cache['r'] = r_1

        o_1 = self.Geto().detach().cpu().Numpy()
        if self.cache.get('o') is not None:
            o_0 = self.cache['o']
            f_change_rate = np.sum(abs(o_1 - o_0)) / np.sum(np.abs(o_0))
            result += 'f_change_rate: %.3e '%f_change_rate
        self.cache['o'] = o_1

        if hasattr(self, 'Geti'):
            i_1 = self.Geti().detach().cpu().Numpy()
            if self.cache.get('i') is not None:
                i_0 = self.cache['i']
                i_change_rate = np.sum(abs(i_1 - i_0)) / np.sum(np.abs(i_0))
                result += 'i_change_rate: %.3e '%i_change_rate
            self.cache['i'] = i_1
        if verbose:
            print(result)
        return result
    def anal_gradient(self, verbose=True):
        result = ''
        for name in ['i', 'r', 'o']:
            if hasattr(self, name):
                weight = getattr(self, name)
                if weight.grad is not None:
                    ratio = torch.sum(torch.abs(weight.grad)) / torch.sum(torch.abs(weight))
                    result += '%s: ratio_grad_weight: %.3e ' % (name, ratio)
        if verbose:
            print(result)
        return result
    def plot_Recurrent_weight(self, ax, cmap):
        weight_r = self.Getr().detach().cpu().Numpy()
        weight_r_mapped, weight_min, weight_max = norm_and_map(weight_r, cmap=cmap, return_min_max=True) # weight_r_mapped: [N_num, res_x, res_y, (r,g,b,a)]
        
        ax.set_title('Recurrent weight')
        ax.imshow(weight_r_mapped, extent=(0, self.N_num, 0, self.N_num))

        norm = mpl.colors.Normalize(vmin=weight_min, vmax=weight_max)
        ax_ = ax.inset_axes([1.05, 0.0, 0.12, 0.8]) # left, bottom, width, height. all are ratios to sub-canvas of ax.
        cbar = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
            cax=ax_, # occupy ax_ 
            ticks=np.linspace(weight_min, weight_max, num=5),
            orientation='vertical')
        cbar.set_label('Connection strength', loc='center')
        
        if self.separate_ei:
            #ax.set_xticklabels('')
            #ax.set_yticklabels('')
            ax.set_xticks([0, self.E_num, self.N_num])
            ax.set_yticks([0, self.E_num, self.N_num])

            ax.set_xticks([(0 + self.E_num)/2, (self.E_num + self.N_num)/2], minor=True)
            ax.set_xticklabels(['to E', 'to I'], minor=True)

            ax.set_yticks([(0 + self.E_num)/2, (self.E_num + self.N_num)/2], minor=True)
            ax.set_yticklabels(['from E', 'from I'], minor=True)
            
            ax.tick_param(axis='both', which='minor', length=0)

        else:
            ax.set_xticks([0, self.N_num])
            ax.set_yticks([0, self.N_num])            

        ax.set_xlabel('Postsynaptic neuron index')
        ax.set_ylabel('Presynaptic neuron index')
    
    def plot_weight(self, ax=None, save=True, save_path='./', save_name='RNN_Navi_weight_plot.png', cmap='jet'):
        if ax is None:
            plt.close('all')
            row_num, col_num = 2, 2
            fig, axes = plt.subplots(nrows=row_num, ncols=col_num, figsize=(5*col_num, 5*row_num)) # figsize unit: inches

        fig.suptitle('Weight Visualization of 1-layer RNN')

        # plot Recurrent weight
        ax = axes[0, 0] # raises error is row_num==col_num==1
        
        self.plot_Recurrent_weight(ax, cmap)

        # plot input_weight
        if self.init_method in ['linear']:
            ax = axes[0, 1]
        elif self.init_method in ['mlp']:
            pass
        else:
            pass
        plt.tight_layout()
        if save:
            utils_torch.EnsureFileDir(save_path)
            plt.savefig(save_path + save_name)
    def GetTrainWeight(self):
        return self.cache.TrainWeight
    def SetTrainWeight(self):
        return utils_torch.model.SetTrainWeightForModel(self)
    def ClearTrainWeight(self):
        utils_torch.model.ClearTrainWeightForModel(self)
    # def SetLogger(self, logger):
    #     return utils_torch.model.SetLoggerForModel(self, logger)
    # def GetLogger(self):
    #     return utils_torch.model.GetLoggerForModel(self)
    # def Log(self, data, Name="Undefined"):
    #     return utils_torch.model.LogForModel(self, data, Name)
    def SetFullName(self, FullName):
        utils_torch.model.SetFullNameForModel(self, FullName)

__MainClass__ = RNNLIF
utils_torch.model.SetMethodForModelClass(__MainClass__)