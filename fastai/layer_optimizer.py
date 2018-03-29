from .core import listy, trainable_params

class LayerOptimizer():
    def __init__(self, opt_fn, layer_groups, lrs, wds=None):
        if not listy(layer_groups): layer_groups = [layer_groups]
        lrs = broadcast(lrs, layer_groups)
        wds = broadcast(wds, layer_groups, default=0.)
        self.layer_groups,self.lrs,self.wds = layer_groups,lrs,wds
        self.opt = opt_fn([
            {'params': trainable_params(lg), 'lr': lr, 'weight_decay': wd}
            for lg,lr,wd in zip(self.layer_groups, self.lrs, self.wds)
        ])

    @property
    def lr(self): return self.lrs[-1]

    def set_lrs(self, lrs):
        self.lrs = lrs
        update_opt('lr', self.opt, lrs)

    def set_wds(self, wds):
        self.wds = wds
        update_opt('weight_decay', self.opt, wds)


def broadcast(xs, along, default=None):
    if xs is None: xs = default
    if not isinstance(xs, Iterable): xs = [xs]
    if len(xs)==1: xs = xs*len(along)
    return xs

def zip_along(xs, along):
    return zip(broadcast(xs, along), along)

def update_opt(param, opt, xs):
    for x, pg in zip_along(xs, opt.param_groups): pg[param] = x
