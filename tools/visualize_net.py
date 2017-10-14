from __future__ import print_function
import find_mxnet
import mxnet as mx
import argparse
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'symbol'))
import symbol_factory

def parse_args():
    parser = argparse.ArgumentParser(description='network visualization')
    parser.add_argument('--network', dest='network', type=str, default='vgg16_reduced',
                        help='the cnn to use')
    parser.add_argument('--num-classes', dest='num_classes', type=int, default=20,
                        help='the number of classes')
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=300,
                        help='set image\'s shape')
    parser.add_argument('--train', dest='train', type=bool, default=False, help='show train net')
    parser.add_argument('--output-dir', dest='output_dir', type=str, default=os.path.dirname(__file__),
                        help='path of the output visualized net')
    parser.add_argument('--print-net', dest='print_net', type=bool, default=False,
                        help='print the network as json')
    args = parser.parse_args()
    return args

def net_visualization(network=None,
                      num_classes=None,
                      data_shape=None,
                      train=None,
                      output_dir=None,
                      print_net=False,
                      net=None):
    # if you specify your net, this means that you are calling this function from somewhere else..
    if net is None:
        if not train:
            net = symbol_factory.get_symbol(network, data_shape, num_classes=num_classes)
        else:
            net = symbol_factory.get_symbol_train(network, data_shape, num_classes=num_classes)

    if not train:
        a = mx.viz.plot_network(net, shape={"data": (1, 3, data_shape, data_shape)}, \
                                node_attrs={"shape": 'rect', "fixedsize": 'false'})
        filename = "ssd_" + network + '_' + str(data_shape)+'_'+'test'
    else:
        a = mx.viz.plot_network(net, shape=None, \
                                node_attrs={"shape": 'rect', "fixedsize": 'false'})
        filename = "ssd_" + network + '_' + 'train'

    a.render(os.path.join(output_dir, filename))
    if print_net:
        print(net.tojson())

if __name__ == '__main__':
    args = parse_args()
    net_visualization(network=args.network,
                      num_classes=args.num_classes,
                      data_shape=args.data_shape,
                      train=args.train,
                      output_dir=args.output_dir,
                      print_net=args.print_net)
