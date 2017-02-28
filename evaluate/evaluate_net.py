from __future__ import print_function
import os
import sys
import importlib
import mxnet as mx
from dataset.iterator import DetRecordIter
from config.config import cfg
from evaluate.eval_metric import MApMetric, VOC07MApMetric
import logging

def evaluate_net(net, path_imgrec, num_classes, mean_pixels, data_shape,
                 model_prefix, epoch, ctx=mx.cpu(), batch_size=1,
                 path_imglist="", nms_thresh=0.45, force_nms=False,
                 ovp_thresh=0.5, use_difficult=False, class_names=None,
                 voc07_metric=False):
    """

    """
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # args
    if isinstance(data_shape, int):
        data_shape = (3, data_shape, data_shape)
    assert len(data_shape) == 3 and data_shape[0] == 3
    model_prefix += '_' + str(data_shape[1])

    # iterator
    eval_iter = DetRecordIter(path_imgrec, batch_size, data_shape,
                              path_imglist=path_imglist, **cfg.valid)
    # network
    sys.path.append(os.path.join(cfg.ROOT_DIR, 'symbol'))
    net = importlib.import_module("symbol_" + net) \
        .get_symbol_eval(num_classes, nms_thresh, force_nms)
    # model params
    _, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
    # init module
    mod = mx.mod.Module(net, label_names=('label',), logger=logger, context=ctx)
    mod.bind(data_shapes=eval_iter.provide_data, label_shapes=eval_iter.provide_label)
    mod.set_params(args, auxs, allow_missing=False, force_init=True)

    # run evaluation
    if voc07_metric:
        metric = VOC07MApMetric(ovp_thresh, use_difficult, class_names)
    else:
        metric = MApMetric(ovp_thresh, use_difficult, class_names)
    results = mod.score(eval_iter, metric, num_batch=None)
    for k, v in results:
        print("{}: {}".format(k, v))
