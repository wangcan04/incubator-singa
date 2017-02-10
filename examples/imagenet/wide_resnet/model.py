# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
''' This model is created following Caffe implementation of GoogleNet
https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/
'''
from singa.layer import Conv2D, Activation, MaxPooling2D, AvgPooling2D,\
        Split, Merge, Flatten, Dense, BatchNormalization, Softmax
from singa import net as ffnet
from singa import initializer

ffnet.verbose=True

def conv(net, src, name, num, kernel, stride=1, pad=0):
    net.add(Conv2D(name + '-conv', num, kernel, stride, pad=pad, use_bias=False), src)
    return net.add(BatchNormalization(name + '-bn'))

def create_net(weight_path=None):
    net = ffnet.FeedForwardNet()
    net.add(Conv2D('input-conv', 64, 7, 2, pad=3, use_bias=False, input_sample_shape=(3, 224, 224)))
    net.add(BatchNormalization('input-bn'))
    net.add(Activation('input_relu'))
    net.add(MaxPooling2D('input_pool', 3, 2, pad=1))

    stage(0, net, 3, 64, 256)
    stage(1, net, 4, 256, 512)
    stage(2, net, 6, 512, 1024)
    stage(3, net, 3, 1024, 2048)

    net.add(AvgPooling2D('avg_pool', 7, 1, pad=0))
    net.add(Flatten('flag'))
    net.add(Dense('dense', 1000))
    # net.add(Softmax('softmax'))

    if weight_path == None:
        for pname, pval in zip(net.param_names(), net.param_values()):
            print pname, pval.shape
            if 'conv' in pname and len(pval.shape) > 1:
                initializer.gaussian(pval, 0, pval.shape[1])
            elif 'dense' in pname:
                if len(pval.shape) > 1:
                    initializer.gaussian(pval, 0, pval.shape[0])
                else:
                    pval.set_value(0)
            # init params from batch norm layer
            elif 'mean' in pname or 'beta' in pname:
                pval.set_value(0)
            elif 'var' in pname:
                pval.set_value(1)
            elif 'gamma' in pname:
                initializer.uniform(pval, 0, 1)
    else:
        net.load(weight_path, use_pickle = 'pickle' in weight_path)
    return net

def stage(sid, net, num_blk, inplane, outplane):
    block('stage%d-blk%d' % (sid, 0), net, inplane, outplane, 1 if sid == 0 else 2)
    for i in range(1, num_blk - 1):
        block('stage%d-blk%d' % (sid, i), net, outplane, outplane)
    return block('stage%d-blk%d' % (sid, num_blk - 1), net, outplane, outplane)


def block(blk_name, net, inplane, outplane, stride=1):
    split = net.add(Split(blk_name + '-split', 2))
    conv1 = conv(net, split, blk_name + '-1', outplane/2, 1, 1)
    relu1 = net.add(Activation(blk_name + '-1-relu'))
    conv2 = conv(net, relu1, blk_name + '-2', outplane/2, 3, stride, pad=1)
    relu2 = net.add(Activation(blk_name + '-2-relu'))
    conv3 = conv(net, relu2, blk_name + '-3', outplane, 1, 1)
    branch = split
    if inplane != outplane:
        branch = conv(net, split, blk_name + '-shortcut', outplane, 1, stride)
    net.add(Merge(blk_name + '-add'), [conv3, branch])
    return net.add(Activation(blk_name + '-relu'))

if __name__ == '__main__':
    create_net('wrn-50-2.pickle')
