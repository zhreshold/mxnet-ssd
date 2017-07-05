from __future__ import print_function, absolute_import
import os.path as osp
from . import find_mxnet
import mxnet as mx
import argparse
from symbol import symbol_factory



parser = argparse.ArgumentParser(description='network visualization')
parser.add_argument('--network', type=str, default='vgg16_reduced',
                    help = 'the cnn to use')
parser.add_argument('--num-classes', type=int, default=20,
                    help='the number of classes')
parser.add_argument('--data-shape', type=int, default=300,
                    help='set image\'s shape')
parser.add_argument('--train', action='store_true', default=False, help='show train net')
args = parser.parse_args()

if not args.train:
    net = symbol_factory.get_symbol(args.network, args.data_shape, num_classes=args.num_classes)
    a = mx.viz.plot_network(net, shape={"data":(1,3,args.data_shape,args.data_shape)}, \
        node_attrs={"shape":'rect', "fixedsize":'false'})
    filename = "ssd_" + args.network + '_' + str(args.data_shape)
    a.render(osp.join(osp.dirname(__file__), filename))
else:
    net = symbol_factory.get_symbol_train(args.network, args.data_shape, num_classes=args.num_classes)
    print(net.tojson())
