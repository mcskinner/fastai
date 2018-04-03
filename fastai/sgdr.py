from .imports import *
from .layer_optimizer import *
import copy


class Callback:
    def on_train_begin(self): pass
    def on_batch_begin(self): pass
    def on_epoch_end(self, metrics): pass
    def on_batch_end(self, metrics): pass
    def on_train_end(self): pass


class LoggingCallback(Callback):
    """Useful for maintaining status of a long-running job.

    Usage:
        learn.fit(0.01, 1, callbacks=[LoggingCallback(save_path="/tmp/log")])
    """

    def __init__(self, save_path):
        super().__init__()
        self.save_path=save_path
    def on_train_begin(self):
        self.batch = 0
        self.epoch = 0
        self.f = open(self.save_path, "a", 1)
        self.log("\ton_train_begin")
    def on_batch_begin(self):
        self.log(str(self.batch)+"\ton_batch_begin")
    def on_epoch_end(self, metrics):
        self.log(str(self.epoch)+"\ton_epoch_end: "+str(metrics))
        self.epoch += 1
    def on_batch_end(self, metrics):
        self.log(str(self.batch)+"\ton_batch_end: "+str(metrics))
        self.batch += 1
    def on_train_end(self):
        self.log("\ton_train_end")
        self.f.close()
    def log(self, string):
        self.f.write(time.strftime("%Y-%m-%dT%H:%M:%S")+"\t"+string+"\n")


class LossRecorder(Callback):
    """Record learning rate and loss by iteration."""

    def __init__(self, layer_opt, save_path=''):
        super().__init__()
        self.layer_opt=layer_opt
        self.save_path=save_path

    def on_train_begin(self):
        self.losses,self.lrs,self.iterations = [],[],[]
        self.iteration = 0
        self.epoch = 0

    def on_epoch_end(self, metrics):
        self.epoch += 1

    def on_batch_end(self, loss):
        self.iteration += 1
        self.lrs.append(self.layer_opt.lr)
        self.iterations.append(self.iteration)
        self.losses.append(loss)

    def plot_loss(self):
        if not in_ipynb(): plt.switch_backend('agg')
        plt.plot(self.iterations[10:], self.losses[10:])
        if not in_ipynb():
            plt.savefig(os.path.join(self.save_path, 'loss_plot.png'))
            np.save(os.path.join(self.save_path, 'losses.npy'), self.losses[10:])

    def plot_lr(self):
        if not in_ipynb(): plt.switch_backend('agg')
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.plot(self.iterations, self.lrs)
        if not in_ipynb():
            plt.savefig(os.path.join(self.save_path, 'lr_plot.png'))


class LR_Updater(LossRecorder):
    """Update the learning rate each batch, recording that and loss."""

    def __init__(self, layer_opt, save_path=''):
        super().__init__(layer_opt, save_path)
        self.init_lrs = np.array(layer_opt.lrs)

    def on_train_begin(self):
        super().on_train_begin()
        self.update_lr()

    def on_batch_end(self, loss):
        res = super().on_batch_end(loss)
        self.update_lr()
        return res

    def update_lr(self):
        new_lrs = self.calc_lr(self.init_lrs)
        self.layer_opt.set_lrs(new_lrs)

    @abstractmethod
    def calc_lr(self, init_lrs):
        raise NotImplementedError


class LR_Finder(LR_Updater):
    """Increase the learning rate linearly or multiplicatively.

    Eventually the learning rate will be sufficiently large that the
    training loss diverges. This callback will stop the training loop
    when it thinks that happens, and then plot the loss against the
    learning rate to help humans pick a good learning rate. That is,
    one which is as high as possible without diverging.
    """

    def __init__(self, layer_opt, nb, end_lr=10, linear=False):
        super().__init__(layer_opt)
        self.linear = linear
        ratio = end_lr/layer_opt.lr
        self.lr_mult = (ratio/nb) if linear else ratio**(1/nb)

    def on_train_begin(self):
        super().on_train_begin()
        self.best = 1e9

    def on_batch_end(self, loss):
        if math.isnan(loss) or loss>self.best*4: return True
        if loss<self.best and self.iteration>10: self.best = loss
        return super().on_batch_end(loss)

    def calc_lr(self, init_lrs):
        mult = self.lr_mult*self.iteration if self.linear else self.lr_mult**self.iteration
        return init_lrs * mult

    def plot(self, n_skip=10, n_skip_end=5):
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip:-n_skip_end], self.losses[n_skip:-n_skip_end])
        plt.xscale('log')


class CosAnneal(LR_Updater):
    """Decrease the learning rate on a cosine schedule, with restarts.

    The schedule looks something like:
      __        ____            ______
        `.     |    `*,        |      `'-.,
          \    |       \       |            ...
           `.__|        `-.____|
    """

    def __init__(self, layer_opt, nb, on_cycle_end=None, cycle_mult=1):
        super().__init__(layer_opt)
        self.nb,self.on_cycle_end,self.cycle_mult = nb,on_cycle_end,cycle_mult

    def on_train_begin(self):
        super().on_train_begin()
        self.cycle_iter,self.cycle_count = 0,0

    def calc_lr(self, init_lrs):
        if self.iteration < self.nb/20:
            self.cycle_iter += 1
            return init_lrs/100.

        cos_out = (1+np.cos(np.pi*self.cycle_iter/self.nb))/2
        self.cycle_iter += 1
        if self.cycle_iter==self.nb:
            self.cycle_iter = 0
            self.nb *= self.cycle_mult
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return init_lrs * cos_out


class CircularLR(LR_Updater):
    """Increase and decrease the learning rate on a sort of sawtooth schedule.

    The schedule looks something like:

        /`*,        /`*,        /`*,
       /    `*,    /    `*,    /     ...
      /        `*,/        `*,/
    """

    def __init__(self, layer_opt, nb, div=4, cut_div=8, on_cycle_end=None):
        super().__init__(layer_opt)
        self.nb,self.div,self.cut_div,self.on_cycle_end = nb,div,cut_div,on_cycle_end

    def on_train_begin(self):
        super().on_train_begin()
        self.cycle_iter,self.cycle_count = 0,0

    def calc_lr(self, init_lrs):
        cut_pt = self.nb//self.cut_div
        if self.cycle_iter<=cut_pt:
            # Interpolate from 0 to 1, over 1/cut_div of the data (default 1/8).
            pct = self.cycle_iter/cut_pt
        else: 
            # Interpolate back from 1 to 0 over the rest of the data (e.g. 7/8 of it).
            pct = 1 - (self.cycle_iter - cut_pt)/(cut_pt*(self.cut_div-1))

        # Interpolation actually goes 1/div to 1 instead of 0 to 1, default 1/4.
        res = init_lrs * (1 + pct*(self.div-1)) / self.div

        self.cycle_iter += 1
        if self.cycle_iter==self.nb:
            self.cycle_iter = 0
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return res


class SaveBestModel(LossRecorder):
    """Save weights of the best model based during training.

    If metrics are provided, the first metric in the list is used to find the best model. 
    If no metrics are provided, the loss is used.
        
    Args:
        model: the fastai model
        lr: indicate to use test images; otherwise use validation images
        name: the name of filename of the weights without '.h5'

    Usage:
        Briefly, you have your model 'learn' variable and call fit.
        >>> learn.fit(lr, 2, cycle_len=2, cycle_mult=1, best_save_name='mybestmodel')
        ....
        >>> learn.load('mybestmodel')
        
        For more details see http://forums.fast.ai/t/a-code-snippet-to-save-the-best-model-during-training/12066
    """

    def __init__(self, model, layer_opt, metrics, name='best_model'):
        super().__init__(layer_opt)
        self.name = name
        self.model = model
        self.best_loss = None
        self.best_acc = None
        self.save_method = self.save_when_only_loss if metrics is None else self.save_when_acc
        
    def on_epoch_end(self, metrics):
        super().on_epoch_end(metrics)
        self.save_method(metrics)
        
    def save_when_only_loss(self, metrics):
        loss = metrics[0]
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.model.save(f'{self.name}')
    
    def save_when_acc(self, metrics):
        loss, acc = metrics[0], metrics[1]
        if self.best_acc is None or acc > self.best_acc or acc == self.best_acc and loss < self.best_loss:
            self.best_acc = acc
            self.best_loss = loss
            self.model.save(f'{self.name}')


class WeightDecaySchedule(Callback):
    def __init__(self, layer_opt, batch_per_epoch, cycle_len, cycle_mult, n_cycles, norm_wds=False, wds_sched_mult=None):
        """Implements the weight decay schedule as mentioned in https://arxiv.org/abs/1711.05101

        Args:
          layer_opt: The LayerOptimizer
          batch_per_epoch: number of batches in an epoch
          cycle_len: number of epochs in the initial cycle
          cycle_mult: cycle length multiplier, cycle_len(t) = cycle_len(t-1) * cycle_mult
          n_cycles: number of cycles to execute
        """
        super().__init__()

        self.layer_opt = layer_opt
        self.batch_per_epoch = batch_per_epoch
        self.init_wds = np.array(layer_opt.wds)  # Weight decays as set by user
        self.init_lrs = np.array(layer_opt.lrs)  # Learning rates as set by user
        self.new_wds = None                      # Holds the new weight decay factors, calculated in on_batch_begin()
        self.param_groups_old = None             # Caches the old parameter values in on_batch_begin()
        self.iteration = 0
        self.epoch = 0
        self.wds_sched_mult = wds_sched_mult
        self.norm_wds = norm_wds
        self.wds_history = list()

        # Pre calculating the number of epochs in the cycle of current running epoch
        self.epoch_to_num_cycles, i = {}, 0
        for _ in range(n_cycles):
            for _ in range(cycle_len):
                self.epoch_to_num_cycles[i] = cycle_len
                i += 1
            cycle_len *= cycle_mult

    def on_train_begin(self):
        self.iteration = 0
        self.epoch = 0

    def on_batch_begin(self):
        # Prepare for decay of weights

        # Default weight decay (as provided by user)
        wdn = self.init_wds

        # Weight decay multiplier (The 'eta' in the paper). Optional.
        wdm = 1.0
        if self.wds_sched_mult is not None:
            wdm = self.wds_sched_mult(self)

        # Weight decay normalized. Optional.
        if self.norm_wds:
            wdn = wdn / np.sqrt(self.batch_per_epoch * self.epoch_to_num_cycles[self.epoch])

        # Final wds
        self.new_wds = wdm * wdn

        # Record the wds
        self.wds_history.append(self.new_wds)

        # Set weight_decay with zeros so that it is not applied in Adam, we will apply it outside in on_batch_end()
        self.layer_opt.set_wds(torch.zeros(self.new_wds.size))
        # We have to save the existing weights before the optimizer changes the values
        self.param_groups_old = copy.deepcopy(self.layer_opt.opt.param_groups)
        self.iteration += 1

    def on_batch_end(self, loss):
        # Decay the weights
        for group, group_old, wds in zip(self.layer_opt.opt.param_groups, self.param_groups_old, self.new_wds):
            for p, p_old in zip(group['params'], group_old['params']):
                if p.grad is None:
                    continue
                p.data = p.data.add(-wds, p_old.data)  # x.add(a, y) = x+a*y

    def on_epoch_end(self, metrics):
        self.epoch += 1
