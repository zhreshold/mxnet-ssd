import tools.find_mxnet
import mxnet as mx
import logging
import sys
import os
import importlib
from initializer import CustomInitializer
from dataset.iterator import DetRecordIter
from train.metric import MultiBoxMetric
from evaluate.eval_metric import MApMetric, VOC07MApMetric
from config.config import cfg

def convert_pretrained(name, args):
    """
    Special operations need to be made due to name inconsistance, etc

    Parameters:
    ---------
    args : dict
        loaded arguments

    Returns:
    ---------
    processed arguments as dict
    """
    if name == 'vgg16_reduced':
        args['conv6_bias'] = args.pop('fc6_bias')
        args['conv6_weight'] = args.pop('fc6_weight')
        args['conv7_bias'] = args.pop('fc7_bias')
        args['conv7_weight'] = args.pop('fc7_weight')
        del args['fc8_weight']
        del args['fc8_bias']
    return args

def train_net(net, train_path, val_path, devkit_path, num_classes, batch_size,
              data_shape, mean_pixels, brightness, contrast, saturation,
              pca_noise, resume, finetune, pretrained, epoch,
              prefix, ctx, begin_epoch, end_epoch, frequent, learning_rate,
              momentum, weight_decay, lr_refactor_step, lr_refactor_ratio,
              eval_batch_size=8, eval_interval=1, nms_thresh=0.45, force_nms=False,
              ovp_thresh=0.5, use_difficult=False, class_names=None,
              voc07_metric=False,
              train_list="", val_list="", iter_monitor=0, log_file=None):
    """

    """
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if log_file:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)

    # check args
    if isinstance(data_shape, int):
        data_shape = (3, data_shape, data_shape)
    assert len(data_shape) == 3 and data_shape[0] == 3
    prefix += '_' + str(data_shape[1])

    if isinstance(mean_pixels, (int, float)):
        mean_pixels = [mean_pixels, mean_pixels, mean_pixels]
    assert len(mean_pixels) == 3, "must provide all RGB mean values"

    # train_iter = mx.image.ImageDetIter(
    #     batch_size, data_shape, path_imglist=train_path, path_root=devkit_path,
    #     mean=mean_pixels, brightness=brightness,
    #     contrast=contrast, saturation=saturation, pca_noise=pca_noise,
    #     shuffle=cfg.TRAIN.EPOCH_SHUFFLE, rand_crop=cfg.TRAIN.RAND_CROPS,
    #     rand_pad=cfg.TRAIN.RAND_PAD, rand_mirror=cfg.TRAIN.RAND_MIRROR)
    # train_iter = mx.image.ImageDetIter(
    #     batch_size, data_shape, path_imgrec=train_path.replace('.lst', '.rec'), path_imgidx=train_path.replace('.lst', '.idx'),
    #     mean=mean_pixels, brightness=brightness,
    #     contrast=contrast, saturation=saturation, pca_noise=pca_noise,
    #     shuffle=cfg.TRAIN.EPOCH_SHUFFLE, rand_crop=cfg.TRAIN.RAND_CROPS,
    #     rand_pad=cfg.TRAIN.RAND_PAD, rand_mirror=cfg.TRAIN.RAND_MIRROR)
    train_iter = DetRecordIter(train_path, batch_size, data_shape,
                               path_imglist=train_list, **cfg.train)

    if val_path:
        # val_iter = mx.image.ImageDetIter(
        #     batch_size, data_shape, path_imgrec=val_path.replace('.lst', '.rec'), path_imgidx=val_path.replace('.lst', '.idx'),
        #     mean=mean_pixels)
        val_iter = DetRecordIter(val_path, eval_batch_size, data_shape,
                                 path_imglist=val_list, **cfg.valid)
        # synchronize label_shape to avoid reshaping executer
        # label_shape = map(max, train_iter.label_shape, val_iter.label_shape)
        # train_iter.reshape(label_shape=label_shape)
        # val_iter.reshape(label_shape=label_shape)
    else:
        val_iter = None

    # load symbol
    sys.path.append(os.path.join(cfg.ROOT_DIR, 'symbol'))
    symbol_module = importlib.import_module("symbol_" + net)
    net = symbol_module.get_symbol_train(num_classes)

    # define layers with fixed weight/bias
    fixed_param_names = [name for name in net.list_arguments() \
        if name.startswith('conv1_') or name.startswith('conv2_')]

    # load pretrained or resume from previous state
    ctx_str = '('+ ','.join([str(c) for c in ctx]) + ')'
    if resume > 0:
        logger.info("Resume training with {} from epoch {}"
            .format(ctx_str, resume))
        _, args, auxs = mx.model.load_checkpoint(prefix, resume)
        begin_epoch = resume
    elif finetune > 0:
        logger.info("Start finetuning with {} from epoch {}"
            .format(ctx_str, finetune))
        _, args, auxs = mx.model.load_checkpoint(prefix, finetune)
        begin_epoch = finetune
        # the prediction convolution layers name starts with relu, so it's fine
        fixed_param_names = [name for name in net.list_arguments() \
            if name.startswith('conv')]
    elif pretrained:
        logger.info("Start training with {} from pretrained model {}"
            .format(ctx_str, pretrained))
        _, args, auxs = mx.model.load_checkpoint(pretrained, epoch)
        args = convert_pretrained(pretrained, args)
    else:
        logger.info("Experimental: start training from scratch with {}"
            .format(ctx_str))
        args = None
        auxs = None
        fixed_param_names = None

    # helper information
    if fixed_param_names:
        logger.info("Freezed parameters: [" + ','.join(fixed_param_names) + ']')

    # init training module
    mod = mx.mod.Module(net, label_names=('label',), logger=logger, context=ctx,
                        fixed_param_names=fixed_param_names)
    # init evaluation module
    val_net = symbol_module.get_symbol_eval(num_classes, nms_thresh, force_nms)
    val_mod = mx.mod.Module(net, label_names=('label',), logger=logger, context=ctx)
    val_mod.bind(data_shapes=val_iter.provide_data, label_shapes=val_iter.provide_label,
                 for_training=False)

    # fit
    batch_end_callback = mx.callback.Speedometer(train_iter.batch_size, frequent=frequent)
    epoch_end_callback = mx.callback.do_checkpoint(prefix)
    iter_refactor = [int(r) for r in lr_refactor_step.split(',')]
    lr_refactor_ratio = [float(r) for r in lr_refactor_ratio.split(',')]
    # lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(iter_refactor, lr_refactor_ratio)
    lr_scheduler = None
    optimizer_params={'learning_rate':learning_rate,
                      'momentum':momentum,
                      'wd':weight_decay,
                      'lr_scheduler':lr_scheduler,
                      'clip_gradient':None,
                      'rescale_grad': 1.0}
    monitor = mx.mon.Monitor(iter_monitor, pattern=".*") if iter_monitor > 0 else None

    # run fit net, every n epochs we run evaluation network to get mAP
    if voc07_metric:
        eval_metric = VOC07MApMetric(ovp_thresh, use_difficult, class_names)
    else:
        eval_metric = MApMetric(ovp_thresh, use_difficult, class_names)
    eval_interval = max(1, int(eval_interval))
    eval_checkpoints = range(begin_epoch, end_epoch, eval_interval)[1:] + [end_epoch]
    for end_epoch in eval_checkpoints:
        mod.fit(train_iter,
                eval_metric=MultiBoxMetric(),
                batch_end_callback=batch_end_callback,
                epoch_end_callback=epoch_end_callback,
                optimizer='sgd',
                optimizer_params=optimizer_params,
                begin_epoch=begin_epoch,
                num_epoch=end_epoch,
                initializer=CustomInitializer(factor_type="in", magnitude=1),
                arg_params=args,
                aux_params=auxs,
                allow_missing=True,
                monitor=monitor)
        begin_epoch = end_epoch
        args, auxs = mod.get_params()
        val_mod.set_params(args, auxs, allow_missing=False)
        results = val_mod.score(val_iter,
                      eval_metric=eval_metric,
                      score_end_callback=mx.callback.Speedometer(val_iter.batch_size),
                      epoch=end_epoch)
        for k, v in results:
            print("{}: {}".format(k, v))
