# SSD: Single Shot MultiBox Detector

SSD is an unified framework for object detection with a single network.

You can use the code to train/evaluate/test for object detection task.

*This repo is still under construction.*

### Disclaimer
This is a re-implementation of original SSD which is based on caffe. The official
repository is available [here](https://github.com/weiliu89/caffe/tree/ssd).
The arXiv paper is available [here](http://arxiv.org/abs/1512.02325).

This example is intended for reproducing the nice detector while fully utilize the
remarkable traits of MXNet. However:
* The model is not compatible with caffe version due to the implementation details.

### Getting started
* You will need python modules: `easydict`, `cv2`, `matplotlib` and `numpy`.
You can install them via pip or package manegers, such as `apt-get`:
```
sudo apt-get install python-opencv python-matplotlib python-numpy
sudo pip install easydict
```
* Build MXNet with extra layers. Follow the official instructions
[here](http://mxnet.readthedocs.io/en/latest/how_to/build.html), and add extra
layers in `config.mk` by pointing `EXTRA_OPERATORS = example/ssd/operator/`.
Remember to enable CUDA if you want to be able to train, since CPU training is
insanely slow. Using CUDNN is not fully tested but should be fine.

### Try the demo
* Download the pretrained model: `to_be_added`, and extract to `model/` directory.
* Run `python demo.py`
* Check `python demo.py --help` for more options.

### Train the model
This example only covers training on Pascal VOC dataset. Other datasets should
be easily supported by adding subclass derived from class `Imdb` in `dataset/imdb.py`.
See example of `dataset/pascal_voc.py` for details.
* Download the converted vgg16_reduced model: , put `.param` and `.json` files
into `model/` directory by default.
* Download the PASCAL VOC dataset, skip this step if you already have one.
```
cd /path/to/where_you_store_datasets/
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
# Extract the data.
tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
```
* We are goint to use `trainval` set in VOC2007/2012 as a common strategy.
The suggested directory structure is to store `VOC2007` and `VOC2012` directories
in the same `VOCdevkit` folder.
* Then link `VOCdevkit` folder to `data/VOCdevkit` by default:
`ln -s /path/to/VOCdevkit /path/to/this_example/data/VOCdevkit`.
Use hard link instead of copy could save us a bit disk space.
* Start training: `python train.py`
* By default, this example will use `batch-size=32` and `learning_rate=0.004`.
You might need to change the parameters a bit if you have different configurations.
Check `python train.py --help` for more training options. For example, if you have 4 GPUs, use:
```
python train.py --gpus 0,1,2,3 --batch-size 128 --lr 0.005
```

### Note
* First run might take a while to initialize, the reason is still unknown.
