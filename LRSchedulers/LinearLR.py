import matplotlib.pyplot as plt
from utils_torch.utils import EnsurePath

class LinearLR(object): # linear lr scheduler log lr = a + b log epoch lr = e ^ a + e ^ b lr
    def __init__(self, optimizer, milestones=None, epochs=None, lr_decays=None, epoch_num=None, verbose=False):
        self.optimizer = optimizer
        self.milestones = milestones
        
        self.base_lrs = []
        for group in optimizer.param_groups:
            self.base_lrs.append(group['lr'])

        self.epoch_now = -1
        if milestones[0][0]>0:
            milestones.insert(0, [0, 1.0])

        self.point_num = point_num = len(milestones)
        #print("point_num:%d"%point_num)
        if point_num==0:
            raise Exception("LinearLR: milstones is empty")
        
        for i in range(point_num-1):
            if milestones[i][0] >= milestones[i+1][0]:
                raise Exception("LinearLR: milstone epochs must be in ascending order.")
            if len(milestones[i])!=2:
                raise Exception("LinearLR: milstone elements must contain 2 and only 2 elementes: epoch and learning rate decay at this epoch.")

        for point in milestones:
            if isinstance(point[0], float):
                point[0] = int(point[0] * epoch_num)
    
        self.init_key_index()
        self.optimizer._step_count = 0
        self.verbose = verbose
        self.step()
    def init_key_index(self):
        # init last and next key index.
        if self.epoch_now < self.milestones[0][0]:
            self.last_index = -1
            #self.next_index = 1
        else:
            self.last_index = 0
            #self.next_index = 1
        
        if self.last_index==-1:
            self.last_decay = self.milestones[0][1]
        else:
            self.last_decay = self.milestones[self.last_index][1]
        if self.last_index==-2:
            self.next_decay = self.milestones[-1][1]
        else:
            self.next_decay = self.milestones[self.last_index+1][1]

    def update_key_index(self):
        if self.last_index!=-2:
            if self.epoch_now >= self.milestones[self.last_index + 1][0]:
                self.last_index += 1
                if self.last_index >= self.point_num-1:
                    self.last_index = -2
                elif self.last_decay==-1:
                    self.last_decay = self.next_decay = self.milestones[0][1]
                else:
                    self.last_decay = self.milestones[self.last_index][1]
                    self.next_decay = self.milestones[self.last_index + 1][1]
                    self.last_epoch = self.milestones[self.last_index][0]
                    self.next_epoch = self.milestones[self.last_index + 1][0]
    def cal_lr_decay(self):
        if self.last_index==-1:
            lr_decay = 1.0
        elif self.last_index==-2 or self.last_index>=self.point_num-1:
            lr_decay = self.milestones[-1][1]
        else:
            lamda = ( self.epoch_now - self.last_epoch ) / (self.next_epoch - self.last_epoch)
            lr_decay = lamda * self.next_decay + ( 1.0 - lamda ) * self.last_decay
        
        return lr_decay

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def Getlast_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def Getlast_lr_decay(self):
        return self._last_lr_decay

    def Getlr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """Display the current learning rate.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(group, lr))
            else:
                print('Epoch {:5d}: adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(epoch, group, lr))
    def step(self, epoch=None):
        self.epoch_now += 1
        self.update_key_index()
        lr_decay = self.cal_lr_decay()
        _last_lr = []
        for i, group in enumerate(self.optimizer.param_groups):
            self.update_key_index()
            lr_new = self.base_lrs[i] * lr_decay
            group['lr'] = lr_new
            #_last_lr.append(lr_new)
            _last_lr.append(group['lr'])
        self._last_lr = _last_lr
        self._last_lr_decay = lr_decay
        
        if self.verbose:
            print("epoch:%d lr_decay:%.1e lr:%s"%(self.epoch_now, lr_decay, str(_last_lr)))

    def reset(self):
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[i]
        
        self.epoch_now = -1
        self.init_key_index()
            
    def plot_lr_decay(self, epoch_num=None, save=True, save_path="./", save_name="LinearLR_plot.png"):
        if epoch_num is None:
            epoch_num = self.milestones[-1][0] * 2
        lr_decays = []
        for i in range(epoch_num):
            self.step()
            lr_decays.append(self.Getlast_lr_decay())
        
        self.reset()
        figure, ax = plt.subplots()
        ax.plot(range(len(lr_decays)), lr_decays)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate Decay")
        ax.set_yscale("log")
        # set xticks
        ax_xticks = [0]
        for stone in self.milestones:
            ax_xticks.append(stone[0])
        if epoch_num-1 not in ax_xticks:
            ax_xticks.append(epoch_num-1)
        ax_xticks_label = list( map( lambda x:str(x), ax_xticks ) )
        #ax_xticks_label = ["tick" for _ in range(len(ax_xticks))]
        ax.set_xticks(ax_xticks)
        ax.set_xticklabels(ax_xticks_label)

        # set yticks
        ax_yticks = [1.0]
        for stone in self.milestones:
            ax_yticks.append(stone[1])
        ax.set_ylim([min(ax_yticks)/10, max(ax_yticks)*10])
        #ax_yticks += [0.0]
        ax_yticks_label = list( map( lambda x:str(x), ax_yticks ) )
        ax.set_yticks(ax_yticks)
        ax.set_yticklabels(ax_yticks_label)

        ax.set_title("Learning Rate Decay - Epoch")
        if save:
            EnsurePath(save_path)
            plt.savefig(save_path + save_name)

        return ax