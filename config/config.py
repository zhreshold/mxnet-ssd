import os
# from easydict import EasyDict as edict
# from tools.rand_sampler import RandCropper, RandPadder
from utils import DotDict, namedtuple_with_defaults, zip_namedtuple, config_as_dict

RandCropper = namedtuple_with_defaults('RandCropper',
    'min_crop_scales, max_crop_scales, \
    min_crop_aspect_ratios, max_crop_aspect_ratios, \
    min_crop_overlaps, max_crop_overlaps, \
    min_crop_sample_coverages, max_crop_sample_coverages, \
    min_crop_object_coverages, max_crop_object_coverages, \
    max_crop_trials',
    [0.0, 1.0,
    0.5, 2.0,
    0.0, 1.0,
    0.0, 1.0,
    0.0, 1.0,
    25])

RandPadder = namedtuple_with_defaults('RandPadder',
    'rand_pad_prob, max_pad_scale, fill_value',
    [0.0, 1.0, 127])

ColorJitter = namedtuple_with_defaults('ColorJitter',
    'random_hue_prob, max_random_hue, \
    random_saturation_prob, max_random_saturation, \
    random_illumination_prob, max_random_illumination, \
    random_contrast_prob, max_random_contrast',
    [0.0, 18,
    0.0, 32,
    0.0, 32,
    0.0, 0.5])


cfg = DotDict()
cfg.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# training configs
cfg.train = DotDict()
# random cropping samplers
cfg.train.rand_crop_samplers = [
    RandCropper(min_crop_scales=0.3, min_crop_overlaps=0.1),
    RandCropper(min_crop_scales=0.3, min_crop_overlaps=0.3),
    RandCropper(min_crop_scales=0.3, min_crop_overlaps=0.5),
    RandCropper(min_crop_scales=0.3, min_crop_overlaps=0.7),
    RandCropper(min_crop_scales=0.3, min_crop_overlaps=0.9),]
# random padding
cfg.train.rand_pad = RandPadder(rand_pad_prob=0.5, max_pad_scale=4.0)
# random color jitter
cfg.train.color_jitter = ColorJitter(random_hue_prob=0.5, random_saturation_prob=0.5,
    random_illumination_prob=0.5, random_contrast_prob=0.5)
cfg.train.rand_mirror_prob = 0.5
cfg.train.shuffle = True
cfg.train.seed = 233
cfg.train.preprocess_threads = 6
cfg.train = config_as_dict(cfg.train)  # convert to normal dict

# validation
cfg.valid = DotDict()
cfg.valid.rand_crop_samplers = []
cfg.valid.rand_pad = RandPadder()
cfg.valid.color_jitter = ColorJitter()
cfg.valid.rand_mirror_prob = 0
cfg.valid.shuffle = False
cfg.valid.seed = 0
cfg.valid = config_as_dict(cfg.valid)  # convert to normal dict



# cfg = edict()
# cfg.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
#
# # training
# cfg.train = edict()
# cfg.train.RAND_CROPS = [
#     {
#     'min_area': 0.3,
#     'ratio': (0.5, 2.0),
#     'min_overlap': 0.1,
#     },
#     {
#     'min_area': 0.3,
#     'ratio': (0.5, 2.0),
#     'min_overlap': 0.3,
#     },
#     {
#     'min_area': 0.3,
#     'ratio': (0.5, 2.0),
#     'min_overlap': 0.5,
#     },
#     {
#     'min_area': 0.3,
#     'ratio': (0.5, 2.0),
#     'min_overlap': 0.7,
#     },
#     {
#     'min_area': 0.3,
#     'ratio': (0.5, 2.0),
#     'min_overlap': 0.9,
#     },
#     {
#     'min_area': 0.3,
#     'ratio': (0.5, 2.0),
#     'max_overlap': 0.1,
#     },
#     ]
# cfg.train.RAND_PAD = {
#     'p': 0.5,
#     'max_area': 4.0,
#     'padval': 128
#     }
# cfg.train.RAND_SAMPLERS = [RandCropper(min_scale=1., max_trials=1, max_sample=1),
#     RandCropper(min_scale=.3, min_aspect_ratio=.5, max_aspect_ratio=2., min_overlap=.1),
#     RandCropper(min_scale=.3, min_aspect_ratio=.5, max_aspect_ratio=2., min_overlap=.3),
#     RandCropper(min_scale=.3, min_aspect_ratio=.5, max_aspect_ratio=2., min_overlap=.5),
#     RandCropper(min_scale=.3, min_aspect_ratio=.5, max_aspect_ratio=2., min_overlap=.7),
#     RandPadder(max_scale=2., min_aspect_ratio=.5, max_aspect_ratio=2., min_gt_scale=.05),
#     RandPadder(max_scale=3., min_aspect_ratio=.5, max_aspect_ratio=2., min_gt_scale=.05),
#     RandPadder(max_scale=4., min_aspect_ratio=.5, max_aspect_ratio=2., min_gt_scale=.05),]
# # cfg.train.RAND_SAMPLERS = []
# cfg.train.RAND_MIRROR = True
# cfg.train.INIT_SHUFFLE = True
# cfg.train.EPOCH_SHUFFLE = True # shuffle training list after each epoch
# cfg.train.RAND_SEED = None
# cfg.train.RESIZE_EPOCH = 1 # save model every N epoch
#
#
# # validation
# cfg.valid = edict()
# cfg.valid.RAND_SAMPLERS = []
# cfg.valid.RAND_MIRROR = False
# cfg.valid.INIT_SHUFFLE = False
# cfg.valid.EPOCH_SHUFFLE = False
# cfg.valid.RAND_SEED = None
