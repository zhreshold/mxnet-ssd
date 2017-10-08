import logging
import os
import scipy.misc
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
        """Callback to log training speed and metrics in TensorBoard."""
        for class_name in self.class_names:
            roc = os.path.join(self.roc_path, 'roc_'+class_name+'.png')
            if not os.path.exists(roc):
                continue
            im = scipy.misc.imread(roc)
            self.summary_writer.image(self.prefix+'_'+class_name, im)

class LogDistributionsCallback(object):
    """Log metrics periodically in TensorBoard.
    This callback works almost same as `callback.Speedometer`, but write TensorBoard event file
    for visualization. For more usage, please refer https://github.com/dmlc/tensorboard
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
        """Callback to log training speed and metrics in TensorBoard."""
        if param.locals is None:
            return
        for name, value in param.locals['arg_params'].iteritems():
            # TODO - implement layer to choose from..
            if self.layers_list is None:
                continue
            if self.prefix is not None:
                name = '%s-%s' % (self.prefix, name)
            self.summary_writer.add_histogram(name, value.asnumpy().flatten())