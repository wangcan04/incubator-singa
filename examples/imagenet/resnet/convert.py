import torchfile
import numpy as np
import cPickle as pickle
from argparse import ArgumentParser


import model

verbose=False

def add_param(name, val, params):
    if name in params:
        assert type(params[name]) == tensor.Tensor, 'duplicated param %s' % name
        assert params[name].size() == val.size, 'dif size for param %s' % name
        params[name].copy_from_numpy(val)
    else:
        params[name] = val

    if verbose:
        print name, params[name].shape


def conv(m, idx, params, param_names):
    outplane = m['weight'].shape[0]
    name = param_names[idx]
    val = np.reshape(m['weight'], (outplane, -1))
    add_param(name, val, params)
    return idx + 1


def batchnorm(m, idx, params, param_names):
    add_param(param_names[idx], m['weight'], params)
    add_param(param_names[idx + 1], m['bias'], params)
    add_param(param_names[idx + 2], m['running_mean'], params)
    add_param(param_names[idx + 3], m['running_var'], params)
    return idx + 4


def linear(m, idx, params, param_names):
    add_param(param_names[idx], np.transpose(m['weight']), params)
    add_param(param_names[idx + 1], m['weight'], params)
    return idx + 2


def traverse(m, idx, params, param_names):
    ''' Traverse all modules of the torch checkpoint file to extract params.

    Args:
        m, a TorchObject
        idx, index for the current cursor of param_names
        params, an empty dictionary (name->numpy) to dump the params via pickle;
            or a list of tensor objects which should be in the same order as
            param_names, called to initialize net created in Singa directly
            using param values from torch checkpoint file.

    Returns:
        the updated idx
    '''
    module_type = m.__dict__['_typename']
    if module_type == ['nn.Sequential', 'nn.ConcatTable'] :
        for x in m.modules:
            idx = traverse(x, idx, params, param_names)
    elif 'SpatialConvolution' in module_type:
        idx = conv(m, idx, params, param_names)
    elif 'SpatialBatchNormalization' in module_type:
        idx = batchnorm(m, idx, params, param_names)
    elif 'Linear' in module_type:
        idx = liner(m, idx, params, param_names)
    return idx


if __name__ == '__main__':
    parser = ArgumentParser(description='Convert params from torch to python dict')
    parser.add_argument("infile", help="torch checkpoint file")
    parser.add_argument("model", choices = ['resnet', 'wrn', 'preact'])
    parser.add_argument("depth", choices = [18, 34, 50, 101, 152])

    net = model.create_net(arrgs.model, args.depth)
    m = torchfile.load(args.infile)
    param_vals = {}
    traverse(m, net.param_names(), param_vals)
    if len(param_vals) != len(param_names):
        print 'The following params are missing from torch file'
        miss = [name if name not in param_vals for name in param_names]
        print miss

    outfile = os.path.splitext(infile)[0] + '.pickle'
    with open(outfile, 'wb') as fd:
        pickle.dump(params, fd)
