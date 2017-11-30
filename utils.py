
import chainer
import chainer.functions as F
from chainer import cuda

xp = cuda.cupy

def exp_mask(val, mask):
    xp = cuda.get_array_module(val)
    return val + (1 - xp.asarray(mask, xp.float32)) * -1e-30  # add is ok?
    
def softmax(logits, mask=None):
    if mask is not None:
        logits = exp_mask(logits, mask)
    flat_logits = flatten(logits)
    flat_out = F.softmax(flat_logits)
    out = flat_out.reshape(list(logits.shape[:-1]) + [-1])
    return out

def softsel(target, logits, mask=None, scope=None):
    a = softmax(logits, mask=mask)
    target_rank = len(target.shape)
    out = F.sum(F.tile(F.expand_dims(a, -1), target.shape[-1]) * target, target_rank - 2)
    return out

def make_mask(sequence):
    xp = cuda.get_array_module(sequence)
    populated = xp.sign(xp.abs(xp.sum(sequence, axis=2)))
    mask = F.expand_dims(populated, -1)
    return mask

def flatten(args):
    if type(args) is not list:
        args = [args]
    flat_arg = F.concat([arg.reshape((-1,arg.shape[-1])) for arg in args], axis=1)
    return flat_arg

def flat_linear(layer, args, squeeze=False):
    flat_arg = flatten(args)
    flat_out = layer(flat_arg)
    if type(args) is not list:
        reconst_shape = list(args.shape[:-1])
    else:
        reconst_shape = list(args[0].shape[:-1])
    out = flat_out.reshape(reconst_shape + [-1])
    if squeeze:
        out = F.squeeze(out)
    return out

def linear_logits(layer, args, mask=None):
    logits = flat_linear(layer, args, squeeze=True)
    if mask is not None:
        logits = exp_mask(logits, mask) # mask=(N,48,48)
    return logits

def get_logits(layer, args, mask=None):
    new_arg = args[0] * args[1]
    return linear_logits(layer, [args[0], args[1], new_arg], mask=mask)

def interaction_layer(x, y):
    x_len = x.shape[1]
    y_len = y.shape[1]
    x_aug = F.tile(F.expand_dims(x, 2), (1, 1, y_len, 1))
    y_aug = F.tile(F.expand_dims(y, 1), (1, x_len, 1, 1))
    h_logits = x_aug * y_aug
    return h_logits

