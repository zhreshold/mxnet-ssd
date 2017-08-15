# SSD: Single Shot MultiBox Object Detector

SSD is an unified framework for object detection with a single network.

You can use the code to train/evaluate/test for object detection task.

### Disclaimer
This is a re-implementation of original SSD which is based on caffe. The official
repository is available [here](https://github.com/weiliu89/caffe/tree/ssd).
The arXiv paper is available [here](http://arxiv.org/abs/1512.02325).

This example is intended for reproducing the nice detector while fully utilize the
remarkable traits of MXNet.
* The model is fully compatible with caffe version.
* Model [converter](#convert-caffemodel) from caffe is available now!
* The result is almost identical to the original version. However, due to different implementation details, the results might differ slightly.

### What's new
* Added multiple trained models.
* Added a much simpler way to compose network from mainstream classification networks (resnet, inception...) and [Guide](symbol/README.md).
* Update to the latest version according to caffe version, with 5% mAP increase.
* Use C++ record iterator based on back-end multi-thread engine to achieve huge speed up on multi-gpu environments.
* Monitor validation mAP during training.
* More network symbols under development and test.
* Extra operators are now in `mxnet/src/operator/contrib`, symbols are modified. Please use [Release-v0.2-beta](https://github.com/zhreshold/mxnet-ssd/releases/tag/v0.2-beta) for old models.

### Demo results
![demo1](https://cloud.githubusercontent.com/assets/3307514/19171057/8e1a0cc4-8be0-11e6-9d8f-088c25353b40.png)
![demo2](https://cloud.githubusercontent.com/assets/3307514/19171063/91ec2792-8be0-11e6-983c-773bd6868fa8.png)
![demo3](https://cloud.githubusercontent.com/assets/3307514/19171086/a9346842-8be0-11e6-8011-c17716b22ad3.png)

### mAP
|        Model          | Training data    | Test data |  mAP | Note |
|:-----------------:|:----------------:|:---------:|:----:|:-----|
| [VGG16_reduced 300x300](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.5-beta/vgg16_ssd_300_voc0712_trainval.zip) | VOC07+12 trainval| VOC07 test| 77.8| fast |
| [VGG16_reduced 512x512](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.5-beta/vgg16_ssd_512_voc0712_trainval.zip) | VOC07+12 trainval | VOC07 test| 79.9| slow |
| [Inception-v3 512x512](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.6/inceptionv3_ssd_512_voc0712_trainval.zip) | VOC07+12 trainval| VOC07 test| 78.9 | fastest |
| [Resnet-50 512x512](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.6/resnet50_ssd_512_voc0712_trainval.zip) | VOC07+12 trainval| VOC07 test| 78.9 | fast |

*More to be added*

### Speed
|         Model         |   GPU            | CUDNN | Batch-size | FPS* |
|:---------------------:|:----------------:|:-----:|:----------:|:----:|
| VGG16_reduced 300x300 | TITAN X(Maxwell) | v5.1  |     16     | 95   |
| VGG16_reduced 300x300 | TITAN X(Maxwell) | v5.1  |     8      | 95   |
| VGG16_reduced 300x300 | TITAN X(Maxwell) | v5.1  |     1      | 64   |
| VGG16_reduced 300x300 | TITAN X(Maxwell) |  N/A  |     8      | 36   |
| VGG16_reduced 300x300 | TITAN X(Maxwell) |  N/A  |     1      | 28   |

*Forward time only, data loading and drawing excluded.*

### Getting started
* You will need python modules: `cv2`, `matplotlib` and `numpy`.
If you use mxnet-python api, you probably have already got them.
You can install them via pip or package manegers, such as `apt-get`:
```
sudo apt-get install python-opencv python-matplotlib python-numpy
```
* Clone this repo:
```
# if you don't have git, install it via apt or homebrew/yum based on your system
sudo apt-get install git
# cd where you would like to clone this repo
cd ~
git clone --recursive https://github.com/zhreshold/mxnet-ssd.git
# make sure you clone this with --recursive
# if not done correctly or you are using downloaded repo, pull them all via:
# git submodule update --recursive --init
cd mxnet-ssd/mxnet
```
* Build MXNet: `cd /path/to/mxnet-ssd/mxnet`. Follow the official instructions [here](http://mxnet.io/get_started/install.html).
```
# for Ubuntu/Debian
cp make/config.mk ./config.mk
# modify it if necessary
```
Remember to enable CUDA if you want to be able to train, since CPU training is
insanely slow. Using CUDNN is optional, but highly recommanded.

### Try the demo
* Download the pretrained model: [`ssd_resnet50_0712.zip`](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.6/resnet50_ssd_512_voc0712_trainval.zip), and extract to `model/` directory.
* Run
```
# cd /path/to/mxnet-ssd
python demo.py --gpu 0
# play with examples:
python demo.py --epoch 0 --images ./data/demo/dog.jpg --thresh 0.5
python demo.py --cpu --network resnet50 --data-shape 512
# wait for library to load for the first time
```
* Check `python demo.py --help` for more options.

### Train the model
This example only covers training on Pascal VOC dataset. Other datasets should
be easily supported by adding subclass derived from class `Imdb` in `dataset/imdb.py`.
See example of `dataset/pascal_voc.py` for details.
* Download the converted pretrained `vgg16_reduced` model [here](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.2-beta/vgg16_reduced.zip), unzip `.param` and `.json` files
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
* We are going to use `trainval` set in VOC2007/2012 as a common strategy.
The suggested directory structure is to store `VOC2007` and `VOC2012` directories
in the same `VOCdevkit` folder.
* Then link `VOCdevkit` folder to `data/VOCdevkit` by default:
```
ln -s /path/to/VOCdevkit /path/to/this_example/data/VOCdevkit
```
Use hard link instead of copy could save us a bit disk space.
* Create packed binary file for faster training:
```
# cd /path/to/mxnet-ssd
bash tools/prepare_pascal.sh
# or if you are using windows
python tools/prepare_dataset.py --dataset pascal --year 2007,2012 --set trainval --target ./data/train.lst
python tools/prepare_dataset.py --dataset pascal --year 2007 --set test --target ./data/val.lst --shuffle False
```
* Start training:
```
python train.py
```
* By default, this example will use `batch-size=32` and `learning_rate=0.004`.
You might need to change the parameters a bit if you have different configurations.
Check `python train.py --help` for more training options. For example, if you have 4 GPUs, use:
```
# note that a perfect training parameter set is yet to be discovered for multi-gpu
python train.py --gpus 0,1,2,3 --batch-size 128 --lr 0.001
```
* Memory usage: MXNet is very memory efficient, training on `VGG16_reduced` model with `batch-size` 32 takes around 4684MB without CUDNN(conv1_x and conv2_x fixed).

### Evalute trained model
Use:
```
# cd /path/to/mxnet-ssd
python evaluate.py --gpus 0,1 --batch-size 128 --epoch 0
```
### Convert model to deploy mode
This simply removes all loss layers, and attach a layer for merging results and non-maximum suppression.
Useful when loading python symbol is not available.
```
# cd /path/to/mxnet-ssd
python deploy.py --num-class 20
# then you can run demo with new model without loading python symbol
python demo.py --prefix model/ssd_300_deploy --epoch 0 --deploy
```

### Convert caffemodel
Converter from caffe is available at `/path/to/mxnet-ssd/tools/caffe_converter`

This is specifically modified to handle custom layer in caffe-ssd. Usage:
```
cd /path/to/mxnet-ssd/tools/caffe_converter
make
python convert_model.py deploy.prototxt name_of_pretrained_caffe_model.caffemodel ssd_converted
# you will use this model in deploy mode without loading from python symbol
python demo.py --prefix ssd_converted --epoch 1 --deploy
```
There is no guarantee that conversion will always work, but at least it's good for now.

### Legacy models
Since the new interface for composing network is introduced, the old models have inconsistent names for weights.
You can still load the previous model by rename the symbol to `legacy_xxx.py`
and call with `python train/demo.py --network legacy_xxx `
For example:
```
python demo.py --network 'legacy_vgg16_ssd_300.py' --prefix model/ssd_300 --epoch 0
```
