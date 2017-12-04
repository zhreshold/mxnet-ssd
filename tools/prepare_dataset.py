from __future__ import print_function
import sys, os
import argparse
import subprocess
import mxnet
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))
from dataset.pascal_voc import PascalVoc
from dataset.mscoco import Coco
from dataset.concat_db import ConcatDB

def load_pascal(image_set, year, devkit_path, shuffle=False, class_names=None, true_negative=None):
    """
    wrapper function for loading pascal voc dataset

    Parameters:
    ----------
    image_set : str
        train, trainval...
    year : str
        2007, 2012 or combinations splitted by comma
    devkit_path : str
        root directory of dataset
    shuffle : bool
        whether to shuffle initial list

    Returns:
    ----------
    Imdb
    """
    image_set = [y.strip() for y in image_set.split(',')]
    assert image_set, "No image_set specified"
    year = [y.strip() for y in year.split(',')]
    assert year, "No year specified"

    # make sure (# sets == # years)
    if len(image_set) > 1 and len(year) == 1:
        year = year * len(image_set)
    if len(image_set) == 1 and len(year) > 1:
        image_set = image_set * len(year)
    assert len(image_set) == len(year), "Number of sets and year mismatch"

    imdbs = []
    for s, y in zip(image_set, year):
        imdbs.append(PascalVoc(s, y, devkit_path, shuffle, is_train=True, class_names=class_names, true_negative_images=true_negative))
    if len(imdbs) > 1:
        return ConcatDB(imdbs, shuffle)
    else:
        return imdbs[0]

def load_coco(image_set, dirname, shuffle=False):
    """
    wrapper function for loading ms coco dataset

    Parameters:
    ----------
    image_set : str
        train2014, val2014, valminusminival2014, minival2014
    dirname: str
        root dir for coco
    shuffle: boolean
        initial shuffle
    """
    anno_files = ['instances_' + y.strip() + '.json' for y in image_set.split(',')]
    assert anno_files, "No image set specified"
    imdbs = []
    for af in anno_files:
        af_path = os.path.join(dirname, 'annotations', af)
        imdbs.append(Coco(af_path, dirname, shuffle=shuffle))
    if len(imdbs) > 1:
        return ConcatDB(imdbs, shuffle)
    else:
        return imdbs[0]

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare lists for dataset')
    parser.add_argument('--dataset', dest='dataset', help='dataset to use',
                        default='pascal', type=str)
    parser.add_argument('--year', dest='year', help='which year to use',
                        default='2007,2012', type=str)
    parser.add_argument('--set', dest='set', help='train, val, trainval, test',
                        default='trainval', type=str)
    parser.add_argument('--target', dest='target', help='output list file',
                        default=None,
                        type=str)
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default=None, help='string of comma separated names, or text filename')
    parser.add_argument('--root', dest='root_path', help='dataset root path',
                        default=os.path.join(curr_path, '..', 'data', 'VOCdevkit'),
                        type=str)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle list',
                        type=bool, default=True)
    parser.add_argument('--true-negative', dest='true_negative', help='use images with no GT as true_negative',
                        type=bool, default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.class_names is not None:
        assert args.target is not None, 'for a subset of classes, specify a target path. Its for your own safety'
    if args.dataset == 'pascal':
        db = load_pascal(args.set, args.year, args.root_path, args.shuffle, args.class_names, args.true_negative)
        print("saving list to disk...")
        db.save_imglist(args.target, root=args.root_path)
    elif args.dataset == 'coco':
        db = load_coco(args.set, args.root_path, args.shuffle)
        print("saving list to disk...")
        db.save_imglist(args.target, root=args.root_path)
    else:
        raise NotImplementedError("No implementation for dataset: " + args.dataset)

    print("List file {} generated...".format(args.target))

    im2rec_path = os.path.join(mxnet.__path__[0], 'tools/im2rec.py')
    # final validation - sometimes __path__ (or __file__) gives 'mxnet/python/mxnet' instead of 'mxnet'
    if not os.path.exists(im2rec_path):
        im2rec_path = os.path.join(os.path.dirname(os.path.dirname(mxnet.__path__[0])), 'tools/im2rec.py')
    subprocess.check_call(["python", im2rec_path,
        os.path.abspath(args.target), os.path.abspath(args.root_path),
        "--shuffle", str(int(args.shuffle)), "--pack-label", "1"])

    print("Record file {} generated...".format(args.target.split('.')[0] + '.rec'))
