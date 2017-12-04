import logging
import os
import scipy.misc
import numpy as np
import random
import matplotlib.pyplot as plt

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
                 logfile_path=None, batch_size=None, iter_monitor=0,
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
        if not self.iter_monitor == 0 and self.batch % self.iter_monitor == 0:
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

class LogDetectionsCallback(object):
    """ TODO complete
    """
    def __init__(self, logging_dir=None, prefix='val', images_path=None,
                 class_names=None, batch_size=None, mean_pixels=None, det_thresh=0.5):

        self.logging_dir = logging_dir
        self.prefix = prefix
        if not os.path.exists(images_path):
            os.mkdir(images_path)
        self.images_path = images_path
        self.class_names = class_names
        self.batch_size = batch_size
        self.mean_pixels = mean_pixels
        self.det_thresh = det_thresh
        try:
            from tensorboard import SummaryWriter
            self.summary_writer = SummaryWriter(logging_dir)
        except ImportError:
            logging.error('You can install tensorboard via `pip install tensorboard`.')

    def __call__(self, param):
        """Callback to log detections and gt-boxes as an image in TensorBoard."""
        if param.locals is None:
            return

        result = []
        pad = param.locals['eval_batch'].pad
        images = param.locals['eval_batch'].data[0][0:self.batch_size-pad].asnumpy()
        labels = param.locals['eval_batch'].label[0][0:self.batch_size - pad].asnumpy()
        outputs = [out[0:out.shape[0] - pad] for out in param.locals['self'].get_outputs()]
        # 'det' variable can be in different positions depending with train/test symbols
        if len(outputs) > 1:
            det_idx = [idx for idx,f in enumerate(param.locals['self'].output_names) if f.startswith('det')][0]
            detections = outputs[det_idx].asnumpy()
        else:
            detections = outputs[0].asnumpy()
        for i in range(detections.shape[0]):
            det = detections[i, :, :]
            det = det[np.where(det[:, 0] >= 0)[0]]
            label = labels[i,:,:]
            label = label[np.where(label[:, 0] >= 0)[0]]
            img = images[i,:,:,:] + np.reshape(self.mean_pixels, (3,1,1))
            img = img.astype(np.uint8)
            img = img.transpose([1,2,0])
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            self._visualize_detection_and_labels(img, det, label=label,
                                                 classes=self.class_names, thresh=self.det_thresh,
                                                 plt_path=os.path.join(self.images_path, 'image'+str(i)+'.png'))
            # save to tensorboard
            img_det_graph = scipy.misc.imread(os.path.join(self.images_path, 'image'+str(i)+'.png'))
            self.summary_writer.add_image('image'+str(i)+'.png', img_det_graph)
        return result

    def _visualize_detection_and_labels(self, img, dets, label, classes=[], thresh=None, plt_path=None):
        """
        visualize detections in one image

        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """
        fig = plt.figure()
        plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        colors = dict()
        # Visualize ground-truth boxes
        gt_color = (1.0, 0.0, 0.0)
        for i in range(label.shape[0]):
            cls_id = int(label[i, 0])
            if cls_id >= 0:
                xmin = int(label[i, 1] * width)
                ymin = int(label[i, 2] * height)
                xmax = int(label[i, 3] * width)
                ymax = int(label[i, 4] * height)
                rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                     ymax - ymin, fill=False,
                                     edgecolor=gt_color,
                                     linewidth=2)
                plt.gca().add_patch(rect)
                class_name = str(cls_id)
                if classes and len(classes) > cls_id:
                    class_name = classes[cls_id]
                plt.gca().text(xmin, ymin - 2,
                               'gt',
                               bbox=dict(facecolor=gt_color, alpha=0.5),
                               fontsize=8, color='white')
        # visualize predictions
        for i in range(dets.shape[0]):
            cls_id = int(dets[i, 0])
            if cls_id >= 0:
                score = dets[i, 1]
                if score > thresh:
                    if cls_id not in colors:
                        colors[cls_id] = (random.random(), random.random(), random.random())
                    xmin = int(dets[i, 2] * width)
                    ymin = int(dets[i, 3] * height)
                    xmax = int(dets[i, 4] * width)
                    ymax = int(dets[i, 5] * height)
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                         ymax - ymin, fill=False,
                                         edgecolor=colors[cls_id],
                                         linewidth=3.5)
                    plt.gca().add_patch(rect)
                    class_name = str(cls_id)
                    if classes and len(classes) > cls_id:
                        class_name = classes[cls_id]
                    plt.gca().text(xmin, ymin - 2,
                                    '{:s} {:.3f}'.format(class_name, score),
                                    bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                                    fontsize=8, color='white')
        plt.savefig(plt_path)
        plt.close(fig)



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