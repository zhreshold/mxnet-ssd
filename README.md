# SSD based Object Detection for Country Flag Cards 

This is an example of Object Detection done for country flag cards making use of SSD Network in MXNet Framework.

You can read the detailed post about the approach used in this project in my [Medium post](https://medium.com/@prasad.pai/implementing-object-detection-in-machine-learning-for-flag-cards-with-mxnet-6bc276bb0b14).

This repository is the forked version of [Zhreshold's MXNet-SSD](https://github.com/zhreshold/mxnet-ssd) which is a generic version of MXNet-SSD and some of the instructions of installation are copied from there.

The trained network at present runs for detecting 25 country flags printed on rectangular placards. The countries' flags used are of Argentina, Australia, Bhutan, Brazil, Canada, China, Cuba, France, Germany, Greece, India, Kenya, Mexico, Norway, Portugal, Saudi Arabia, South Africa, Sri Lanka, Sweden, Thailand, Turkey, Ukraine, United Arab Emirates, United Kingdom and United States of America. The evaluation of the trained network can be done on still images or a pre-recorded video or even a live video feed.  This is the result of the network trained using VGG model.

![Detected Result](https://user-images.githubusercontent.com/13696749/32447111-13a44b6a-c331-11e7-9968-9c10343d3e31.png)

## Instructions to run the code
### Getting started
* Option #1 - install using 'Docker'. if you are not familiar with this technology, there is a 'Docker' section below.
you can get the latest image:
```
docker pull daviddocker78:mxnet-ssd:gpu_0.12.0_cuda9
```
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
git clone --recursive https://github.com/Prasad9/mxnet-ssd.git
# make sure you clone this with --recursive
# if not done correctly or you are using downloaded repo, pull them all via:
# git submodule update --recursive --init
cd mxnet-ssd/mxnet
```
* (Skip this step if you have offcial MXNet installed.) Build MXNet: `cd /path/to/mxnet-ssd/mxnet`. Follow the official instructions [here](http://mxnet.io/get_started/install.html).
```
# for Ubuntu/Debian
cp make/config.mk ./config.mk
# modify it if necessary
```

### Trying the demo
The example output image was run on VGG network for only 4 epochs. You will have to download this network's pretrained weight and symbol files from this [dropbox link](https://www.dropbox.com/s/qvu8q4nqm7z3k5u/VGG_SSD_Flags25_epoch4.zip?dl=0). Paste the two files (without changing names) in `model` folder present in root directory of this repository. After that you can try the demo in three formats:

* ### Pre-recorded Video
To try out the pre-recorded video, run the following command:
```
python object_detection.py ./flags/demo_data/video/demo.mp4 --epoch=4
```
The above command will create another file named as `demo_output.mp4` present in same folder as input. You can test the output run on all the 25 flag cards in this [youtube video](https://www.youtube.com/watch?v=QC3GULk9ngU). I encourage you to go through [other options](https://github.com/Prasad9/Detect-Flags-SSD/blob/master/object_detection.py#L37) present in the command.

* ### Still Images
Place all the images which you wish to run the network upon in a common folder containing no other files. Then run the following command:
```
python object_detection.py ./flags/demo_data/images --epoch=4 --thresh=0.6 --plot-prob=0
```
For each of the image present in the folder, it will create an `_output` file containing the predictions. I encourage you to go through [other options](https://github.com/Prasad9/Detect-Flags-SSD/blob/master/object_detection.py#L37) present in the command.

* ### Live Feed Video
At present, the live feed video isn't working properly. This is a work in progress at the moment but you are free to test the video feed directly. To try it, run the following command:
```
python object_detection_app.py
```

### Training your own network
To train your own network, collect the 25 country flags you are interested in (and name the files with country names if you wish to generate label file as well) and place them in a common folder. Then run the following command.
```
cd flags
python data_utils/preprocess.py <src_folder_path> <dst_folder_path> --create-label-file=1
```
The output folder will contain all the resized images at dimension of 224x144 pixels. Update the contents of your label names in this [class_names.txt file](https://github.com/Prasad9/Detect-Flags-SSD/blob/master/flags/input_data/class_names.txt) if you have not generated label file automatically (Arrange labels alphabetically).  

Next you will have to download a dataset which mimics the best background situation you will have while you put your model for testing. I have used [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) which comprises of 2 lakh plus images of celebrity faces. Remember, you can't use CelebA dataset for commericial purposes. Resize all these background images to dimension of 224x224 pixels using the following command.
```
python data_utils/preprocess.py <src_folder_path> <dst_folder_path> --is-background=1
```
As these background image dataset is very likely to contain large number of images, I have optimised the code a bit and set the number of images in the dataset in [constants.py](https://github.com/Prasad9/Detect-Flags-SSD/blob/master/flags/data_utils/constants.py#L13). 

Next, we are going to superimpose the flag files on these background images and add some random noise. You will get an output something like this after this step.
```
python generate_data.py <flag_folder> <bg_img_folder>
```
![Sample Dataset](https://user-images.githubusercontent.com/13696749/32482145-5ec559c8-c3bc-11e7-942f-78c36b7adbea.png)

Next, we have to generate the record file for training and validation. Run the following command.
```
python generate_rec.py
```

Now with the data ready, we can start the training procedure. Our base model can comprise of any model like VGG, Resnet, Inception etc. Whichever model you choose, download the weight and symbol file of that model trained on ImageNet from this MXNet models [website](http://data.mxnet.io/models/imagenet/). Place the downloaded files in `model` folder present in root directory. 

Lastly, train your model.
```
cd ..
python train.py 
```
Depending upon your network, epoch no, batch size etc, you may very well like to add extra options while training your network. Hence, I encourage you to look into the [various options](https://github.com/Prasad9/Detect-Flags-SSD/blob/master/train.py#L12) present while training.
