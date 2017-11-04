import logging
import os
import scipy.misc
import numpy as np


class ParseLogCallback(object):
    """
    1. log distribution's std to tensorboard (as distribution)
    This function make use of mxnet's "monitor" module, and it's output to a log file.
    while training, it is possible to specify layers to be monitored.
    these layers will be printed to a given log file,
    their values are computed **asynchronously**.

    2. log training loss to tensorboard (as scalar)

    Currently - does not support resume training..
    """
    def __init__(self, dist_logging_dir=None, scalar_logging_dir=None,
                 logfile_path=None, batch_size=None, iter_monitor=None,
                 frequent=None, prefix='ssd'):
        self.scalar_logging_dir = scalar_logging_dir
        self.dist_logging_dir = dist_logging_dir
        self.logfile_path = logfile_path
        self.batch_size = batch_size
        self.iter_monitor = iter_monitor
        self.frequent = frequent
        self.prefix = prefix
        self.batch = 0
        self.line_idx = 0
        try:
            from tensorboard import SummaryWriter
            self.dist_summary_writer = SummaryWriter(dist_logging_dir)
            self.scalar_summary_writer = SummaryWriter(scalar_logging_dir)
        except ImportError:
            logging.error('You can install tensorboard via `pip install tensorboard`.')

    def __call__(self, param):
        """Callback to parse a log file and and add params to TensorBoard."""

        # save distributions from the monitor output log
        if self.iter_monitor is not None and self.batch % self.iter_monitor == 0:
            with open(self.logfile_path) as fp:
                for i in range(self.line_idx):
                    fp.next()
                for line in fp:
                    if line.startswith('Batch'):
                        line = line.split(' ')
                        line = [x for x in line if x]
                        layer_name = line[2]
                        layer_value = np.array(float(line[3].split('\t')[0])).flatten()
                        if np.isfinite(layer_value):
                            self.dist_summary_writer.add_histogram(layer_name, layer_value)
                    self.line_idx += 1

        # save training loss
        if self.batch % self.frequent == 0:
            if param.eval_metric is None:
                return
            name_value = param.eval_metric.get_name_value()
            for name, value in name_value:
                if self.prefix is not None:
                    name = '%s-%s' % (self.prefix, name)
                self.scalar_summary_writer.add_scalar(name, value, global_step=self.batch)
        self.batch += 1

class LogROCCallback(object):
    """save roc graphs periodically in TensorBoard.
        write TensorBoard event file, holding the roc graph for every epoch
        logging_dir : str
        this function can only be executed after 'eval_metric.py', since that function is responsible for the graph creation
            where the tensorboard file will be created
        roc_path : list[str]
            list of paths to future roc's
        class_names : list[str]
            list of class names.
        """
    def __init__(self, logging_dir=None, prefix='val', roc_path=None, class_names=None):
        self.prefix = prefix
        self.roc_path = roc_path
        self.class_names = class_names
        try:
            from tensorboard import SummaryWriter
            self.summary_writer = SummaryWriter(logging_dir)
        except ImportError:
            logging.error('You can install tensorboard via `pip install tensorboard`.')

    def __call__(self, param):
        """Callback to log ROC graph as an image in TensorBoard."""
        for class_name in self.class_names:
            roc = os.path.join(self.roc_path, 'roc_'+class_name+'.png')
            if not os.path.exists(roc):
                continue
            im = scipy.misc.imread(roc)
            self.summary_writer.add_image(self.prefix+'_'+class_name, im)

class LogDistributionsCallback(object):
    """
    This function has been deprecated because it consumes too much time.
    The faster way is to use "ParseLogCallback" with a 'iter_monitor' flag

    Log metrics periodically in TensorBoard.
    This callback works almost same as `callback.Speedometer`, but write TensorBoard event file
    for visualization.
    logging_dir : str
        where the tensorboard file will be created
    layers_list : list[str]
        list of layers to be tracked
    """
    def __init__(self, logging_dir, prefix=None, layers_list=None):
        self.prefix = prefix
        self.layers_list = layers_list
        try:
            from tensorboard import SummaryWriter
            self.summary_writer = SummaryWriter(logging_dir)
        except ImportError:
            logging.error('You can install tensorboard via `pip install tensorboard`.')

    def __call__(self, param):
        """Callback to log layers' distributions in TensorBoard."""
        if param.locals is None:
            return
        for name, value in param.locals['arg_params'].iteritems():
            # TODO - implement layer to choose from..
            if self.layers_list is None:
                continue
            if self.prefix is not None:
                name = '%s-%s' % (self.prefix, name)
            self.summary_writer.add_histogram(name, value.asnumpy().flatten())