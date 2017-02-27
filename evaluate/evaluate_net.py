import os
import sys
import importlib
from dataset.pascal_voc import PascalVoc
from dataset.iterator import DetIter, DetRecordIter
from detect.detector import Detector
from config.config import cfg
from eval_metric import MApMetric
import logging
from __future__ import print_function

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
    prefix += '_' + str(data_shape[1])

    # iterator
    eval_iter = DetRecordIter(path_imgrec, batch_size, data_shape,
                              path_imglist=path_imglist, **cfg.valid)
    # network
    sys.path.append(os.path.join(cfg.ROOT_DIR, 'symbol'))
    net = importlib.import_module("symbol_" + net) \
        .get_symbol(num_classes, nms_thresh, force_nms)
    # model params
    _, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
    # init module
    mod = mx.mod.Module(net, label_names=('label',), logger=logger, context=ctx)
    mod.set_params(arg_params, aux_params, allow_missing=False)

    # run evaluation
    if voc07_metric:
        metric = VOC07MApMetric(ovp_thresh, use_difficult, class_names)
    else:
        metric = MApMetric(ovp_thresh, use_difficult, class_names)
    results = mod.score(eval_iter, metric)
    for k, v in results:
        print("{}: {}".format(k, v))

def evaluate_net1(net, dataset, devkit_path, mean_pixels, data_shape,
                 model_prefix, epoch, ctx, year=None, sets='test',
                 batch_size=1, nms_thresh=0.5, force_nms=False):
    """
    Evaluate entire dataset, basically simple wrapper for detections

    Parameters:
    ---------
    dataset : str
        name of dataset to evaluate
    devkit_path : str
        root directory of dataset
    mean_pixels : tuple of float
        (R, G, B) mean pixel values
    data_shape : int
        resize input data shape
    model_prefix : str
        load model prefix
    epoch : int
        load model epoch
    ctx : mx.ctx
        running context, mx.cpu() or mx.gpu(0)...
    year : str or None
        evaluate on which year's data
    sets : str
        evaluation set
    batch_size : int
        using batch_size for evaluation
    nms_thresh : float
        non-maximum suppression threshold
    force_nms : bool
        force suppress different categories
    """
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if dataset == "pascal":
        if not year:
            year = '2007'
        imdb = PascalVoc(sets, year, devkit_path, shuffle=False, is_train=False)
        data_iter = DetIter(imdb, batch_size, data_shape, mean_pixels,
            rand_samplers=[], rand_mirror=False, is_train=False, shuffle=False)
        sys.path.append(os.path.join(cfg.ROOT_DIR, 'symbol'))
        net = importlib.import_module("symbol_" + net) \
            .get_symbol(imdb.num_classes, nms_thresh, force_nms)
        model_prefix += "_" + str(data_shape)
        detector = Detector(net, model_prefix, epoch, data_shape, mean_pixels, batch_size, ctx)
        logger.info("Start evaluation with {} images, be patient...".format(imdb.num_images))
        detections = detector.detect(data_iter)
        imdb.evaluate_detections(detections)
    else:
        raise NotImplementedError, "No support for dataset: " + dataset
