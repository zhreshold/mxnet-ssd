"""References:
Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang,
Tobias Weyand, Marco Andreetto, Hartwig Adam.
"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications."
arXiv preprint arXiv:1704.04861
"""
import mxnet as mx

def depthwise_conv(data, kernel, pad, num_filter, num_group, stride, name):
    conv = mx.symbol.Convolution(data=data, kernel=kernel, pad=pad, stride=stride,
        num_filter=num_group, name=name+'_depthwise', num_group=num_group)
    bn = mx.symbol.BatchNorm(data=conv)
    relu = mx.symbol.Activation(data=bn, act_type='relu')
    conv2 = mx.symbol.Convolution(data=relu, kernel=(1, 1), num_filter=num_filter,
        name=name+'_pointwise')
    bn2 = mx.symbol.BatchNorm(data=conv2)
    relu2 = mx.symbol.Activation(data=bn2, act_type='relu')
    return relu2


def get_symbol(num_classes, **kwargs):
    data = mx.sym.Variable(name='data')

    # first standard conv
    conv1 = mx.sym.Convolution(data=data, num_filter=32, kernel=(3, 3), pad=(1, 1),
        stride=(2, 2), name='conv1')
    bn1 = mx.sym.BatchNorm(data=conv1)
    relu1 = mx.sym.Activation(data=bn1, act_type='relu')

    # separable convolutions
    filters = [64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
    strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
    last_filter = 32
    index = 2
    x = relu1
    for nf, ns in zip(filters, strides):
        x = depthwise_conv(data=x, kernel=(3, 3), pad=(1, 1), num_filter=nf,
            num_group=last_filter, stride=(ns, ns), name='conv{}'.format(index))
        last_filter = nf
        index += 1

    # avg pool
    pool = mx.sym.Pooling(data=x, pool_type='avg', global_pool=True, kernel=(7, 7))
    flat = mx.sym.Flatten(data=pool)
    fc = mx.sym.FullyConnected(data=flat, num_hidden=num_classes, name='fc')
    softmax = mx.sym.SoftmaxOutput(data=fc, name='softmax')
    return softmax
