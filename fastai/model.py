from .imports import *
from .torch_imports import *
from .core import *
from .layer_optimizer import *

def cut_model(m, cut):
    return list(m.children())[:cut] if cut else [m]

def predict_to_bcolz(m, gen, arr, workers=4):
    arr.trim(len(arr))
    lock=threading.Lock()
    m.eval()
    for x,*_ in tqdm(gen):
        y = to_np(m(VV(x)).data)
        with lock:
            arr.append(y)
            arr.flush()

def num_features(m):
    c=children(m)
    if len(c)==0: return None
    for l in reversed(c):
        if hasattr(l, 'num_features'): return l.num_features
        res = num_features(l)
        if res is not None: return res

# assume main output first, others e.g. RNN hidden after.
def split_out(o): return o[0], o[1:] if isinstance(o, tuple) else o, []

class Stepper():
    def __init__(self, m, opt, crit, clip=0, reg_fn=None):
        self.m,self.opt,self.crit,self.clip,self.reg_fn = m,opt,crit,clip,reg_fn
        self.reset(True)

    def reset(self, train=True):
        if train: apply_leaf(self.m, set_train_mode)
        else: self.m.eval()
        if hasattr(self.m, 'reset'): self.m.reset()

    def step(self, xs, y, epoch):
        output,xtra = split_out(self.m(*xs))
        self.opt.zero_grad()
        loss = raw_loss = self.crit(output, y)
        if self.reg_fn: loss = self.reg_fn(output, xtra, raw_loss)
        loss.backward()
        if self.clip: nn.utils.clip_grad_norm(trainable_params(self.m), self.clip)
        self.opt.step()
        return raw_loss.data[0]

    def evaluate(self, xs, y):
        preds,_ = split_out(self.m(*xs))
        return preds, self.crit(preds, y)

def is_frozen_batchnorm(m):
    if not hasattr(m, 'running_mean'): return False
    return getattr(m, 'bn_freeze', False) or not getattr(m, 'trainable', False)
def is_frozen_dropout(m):
    if not getattr(m, 'drop_freeze', False): return False
    return hasattr(m, 'p') and ('drop' in type(m).__name__.lower())
def set_train_mode(m):
    if is_frozen_batchnorm(m) or is_frozen_dropout(m): m.eval()
    else: m.train()

def fit(model, data, epochs, opt, crit, metrics=None, callbacks=None, stepper=Stepper, **kwargs):
    """ Fits a model

    Arguments:
       model (model): any pytorch module
           net = to_gpu(net)
       data (ModelData): see ModelData class and subclasses
       opt: optimizer. Example: opt=optim.Adam(net.parameters())
       epochs(int): number of epochs
       crit: loss function to optimize. Example: F.cross_entropy
    """
    stepper = stepper(model, opt, crit, **kwargs)
    metrics = metrics or []
    callbacks = callbacks or []
    avg_mom = 0.98
    batch_num,avg_loss = 0,0.
    for cb in callbacks: cb.on_train_begin()
    names = ['epoch', 'trn_loss', 'val_loss'] + [f.__name__ for f in metrics]
    layout = '{!s:10} ' * len(names)

    num_batch = len(data.trn_dl)
    if epochs<1:  # Allow partial epochs.
        num_batch = int(num_batch*epochs)
        epochs = 1

    for epoch in tnrange(epochs, desc='Epoch'):
        stepper.reset(True)
        t = tqdm(iter(data.trn_dl), leave=False, total=num_batch)
        for i, (*x,y) in enumerate(t):
            batch_num += 1
            for cb in callbacks: cb.on_batch_begin()
            loss = stepper.step(V(x),V(y), epoch)
            avg_loss += (1-avg_mom) * (loss-avg_loss)
            debias_loss = avg_loss / (1 - avg_mom**batch_num)
            t.set_postfix(loss=debias_loss)
            stop = np.any([cb.on_batch_end(debias_loss) for cb in callbacks])
            if stop: return
            if i>num_batch: break  # seems weird, but used for partial epochs above.

        vals = validate(stepper, data.val_dl, metrics)
        if epoch == 0: print(layout.format(*names))
        print_stats(epoch, [debias_loss] + vals)
        stop = np.any([cb.on_epoch_end(vals) for cb in callbacks])
        if stop: break

    for cb in callbacks: cb.on_train_end()
    return vals


def print_stats(epoch, values, decimals=6):
    layout = '{!s:^10}' + ' {!s:10}' * len(values)
    values = [epoch] + list(np.round(values, decimals))
    print(layout.format(*values))

def validate(stepper, dl, metrics):
    loss,res = [],[]
    stepper.reset(False)
    for (*x,y) in iter(dl):
        preds,l = stepper.evaluate(VV(x), VV(y))
        loss.append(to_np(l))
        res.append([f(preds.data,y) for f in metrics])

    # Not quite right, in general the last batch is overweighted.
    return [np.mean(loss)] + list(np.mean(np.stack(res), 0))

def predict(m, dl):
    preda,_ = predict_with_targs_(m, dl)
    return to_np(torch.cat(preda))

def predict_with_targs(m, dl):
    preda,targa = predict_with_targs_(m, dl)
    return to_np(torch.cat(preda)), to_np(torch.cat(targa))

def predict_with_targs_(m, dl):
    m.eval()
    if hasattr(m, 'reset'): m.reset()
    pred_n_targs = [(get_prediction_(m(*VV(x))),y) for *x,y in iter(dl)]
    return zip(*pred_n_targs)  # transpose

def get_prediction_(x):
    if listy(x): x=x[0]  # assume main output first, others e.g. RNN hidden after.
    return x.data

# Only used by courses/dl2/pascal.ipynb, probably worth moving there.
def predict_batch(m, x):
    m.eval()
    if hasattr(m, 'reset'): m.reset()
    return m(VV(x))

# From https://github.com/ncullen93/torchsample
# Another thing that is only used by learner.py, maybe move.
def model_summary(m, input_size):
    hooks = []
    summary = OrderedDict()
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx+1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['output_shape'] = list(output.size())
            # Assume first dimension is batch size.
            summary[m_key]['input_shape'][0] = -1
            summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and module.bias is not None:
                params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        is_other_base_module = not (module == m
                                    or isinstance(module, nn.Sequential)
                                    or isinstance(module, nn.ModuleList))
        if is_other_base_module:
            hooks.append(module.register_forward_hook(hook))

    m.apply(register_hook)

    if listy(input_size[0]):
        x = [to_gpu(Variable(torch.rand(3,*in_size))) for in_size in input_size]
    else: x = [to_gpu(Variable(torch.rand(3,*input_size)))]
    m(*x)

    for h in hooks: h.remove()
    return summary
