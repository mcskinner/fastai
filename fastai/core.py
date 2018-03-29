from .imports import *
from .torch_imports import *

def noop(*args, **kwargs): pass

# Only used by learner; likely for cycle_mult e.g. 2x needs 2^n-1 iterations.
def sum_geom(a,r,n): return a*n if r==1 else math.ceil(a*(1-r**n)/(1-r))

conv_dict = {
    np.dtype('int8'): torch.LongTensor, np.dtype('int16'): torch.LongTensor,
    np.dtype('int32'): torch.LongTensor, np.dtype('int64'): torch.LongTensor,
    np.dtype('float32'): torch.FloatTensor, np.dtype('float64'): torch.FloatTensor,
}

# Made up. Like R's sapply for simple apply that deals with all shapes.
def sapply(x, f): return [f(o) for o in x] if listy(x) else f(x)
def listy(x): return isinstance(x, (list,tuple))

def A(*a):
    return np.array(a[0]) if len(a)==1 else [np.array(o) for o in a]

def T_(a):
    if torch.is_tensor(a): return a
    a = np.array(np.ascontiguousarray(a))
    if a.dtype in (np.int8, np.int16, np.int32, np.int64):
        return torch.LongTensor(a.astype(np.int64))
    if a.dtype in (np.float32, np.float64):
        return torch.FloatTensor(a.astype(np.float32))
    raise NotImplementedError(a.dtype)
def T(a): return to_gpu(T_(a), async=True)

def create_variable(x, vol, grad=False):
    if isinstance(x, Variable):
        return x
    return Variable(T(x), vol=vol, grad=grad)

def V_(x, grad=False, vol=False): return create_variable(x, vol, grad=grad)
def VV_(x):                       return create_variable(x, True)
def V(x, grad=False, vol=False):  return sapply(x, partial(V_, grad=grad, vol=vol))
def VV(x):                        return sapply(x, VV_)

def to_np(v):
    if listy(v): return [to_np(o) for o in v]
    if isinstance(v, Variable): v=v.data
    return v.cpu().numpy()

USE_GPU=True
def has_cuda(): return torch.cuda.is_available()
def to_gpu(x, *args, **kwargs): return x.cuda(*args, **kwargs) if has_cuda() and USE_GPU else x

def split_by_idxs(seq, idxs):
    last = 0
    for idx in idxs:
        yield seq[last:idx]
        last = idx
    yield seq[last:]

def trainable_params_(m):
    return [p for p in m.parameters() if p.requires_grad]

def trainable_params(p):
    if listy(p):
        return list(chain(*[trainable_params_(o) for o in p]))
    return trainable_params_(p)

# Presumably `m` means "module" in the torch.nn sense.
def set_trainable_attr(m,b):
    m.trainable=b
    for p in m.parameters(): p.requires_grad=b

def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module): f(m)  # not returned, f is a sink.
    if len(c)>0:
        for l in c: apply_leaf(l,f)
    # `len` duck typing breaks before anything can get here.

def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m,b))

def SGD_Momentum(momentum): return partial(optim.SGD, momentum=momentum)

def one_hot(a,c): return np.eye(c)[a]

def partition(a,sz): return [a[i:i+sz] for i in range(0, len(a), sz)]
def partition_by_cores(a): return partition(a, len(a)//num_cpus() + 1)

def chunk_iter(it, sz):
    while True:
        chunk = []
        try:
            for _ in range(sz): chunk.append(next(it))
            yield chunk
        except StopIteration:
            if chunk: yield chunk
            break

def num_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


class BasicModel():
    """Must specify a `children` method."""
    def __init__(self, model, name='unnamed'): self.model,self.name = model,name
    def get_layer_groups(self, do_fc=False): return children(self.model)

class SingleModel(BasicModel):
    def get_layer_groups(self): return [self.model]

class SimpleNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(l1, l2) for l1, l2 in zip(layers, layers[1:])])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return F.log_softmax(l_x, dim=-1)


def save(fn, a): pickle.dump(a, open(fn,'wb'))
def load(fn): return pickle.load(open(fn,'rb'))
def load2(fn): return pickle.load(open(fn,'rb'), encoding='iso-8859-1')

def load_array(fname): return bcolz.open(fname)[:]
