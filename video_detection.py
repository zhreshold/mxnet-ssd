import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tools.find_mxnet
import mxnet as mx
import sys
from detect.image_detector import ImageDetector
from detect.detector import Detector
from symbol.symbol_factory import get_symbol

from utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool

from moviepy.editor import VideoFileClip

class_names = 'Argentina, Australia, Bhutan, Brazil, Canada, China, Cuba, France, Germany, Greece, India, \
 			   Kenya, Mexico, Norway, Portugal, Saudi Arabia, South Africa, Sri Lanka, Sweden, Thailand, \
			   Turkey, Ukraine, U.A.E., U.K., U.S.A.'	
detector = None
thresh = None

def get_detector(net, prefix, epoch, data_shape, mean_pixels, ctx, class_names,
                 nms_thresh=0.5, force_nms=True, nms_topk=400):

	if net is not None:
		net = get_symbol(net, data_shape, num_classes=len(class_names), nms_thresh=nms_thresh,
			force_nms=force_nms, nms_topk=nms_topk)
	detector = ImageDetector(net, prefix, epoch, data_shape, mean_pixels, class_names, ctx=ctx)
	return detector

def process_image(image_frame):
	# run detection
	detected_img = detector.detect_and_layover_image(image_frame, thresh, False)
	return detected_img

def parse_args():
	parser = argparse.ArgumentParser(description='Detect objects in the video')
	parser.add_argument('video_path', help = 'Where video is present', type = str)
	parser.add_argument('--network', dest='network', type=str, default='vgg16_reduced',
						help='which network to use')
	parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
						default=1, type=int)
	parser.add_argument('--prefix', dest='prefix', help='Trained model prefix',
						default=os.path.join(os.getcwd(), 'model', 'ssd'), type=str)
	parser.add_argument('--thresh', dest='thresh', help='Threshold of confidence level',
						default=0.43, type=float)
	parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.45,
						help='non-maximum suppression threshold')
	parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
						help='red mean value')
	parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
						help='green mean value')
	parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
						help='blue mean value')
	parser.add_argument('--data-shape', dest='data_shape', type=int, default=300,
						help='set image shape')
	parser.add_argument('--class-names', dest='class_names', type=str,
						default = class_names, help='string of comma separated names')
	parser.add_argument('--force', dest='force_nms', type=bool, default=True,
						help='force non-maximum suppression on different class')
	parser.add_argument('--has-gpu', dest='gpu', help='GPU device 1 if present else 0',
						default=1, type=int)
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	global detector, thresh
	args = parse_args()

	color_subtract = (args.mean_r, args.mean_g, args.mean_b)
	ctx = mx.gpu(0) if args.gpu == 1 else mx.cpu(0)
	class_names = [class_name.strip() for class_name in args.class_names.split(',')]
	detector = get_detector(args.network, args.prefix, args.epoch, args.data_shape, color_subtract, ctx,
                            class_names, args.nms_thresh, args.force_nms)
	thresh = args.thresh

	video_path_comp = args.video_path.split('.')
	output_at = video_path_comp[0] + '_output.' + video_path_comp[1] 
	clip1 = VideoFileClip(args.video_path)
	
	white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
	white_clip.write_videofile(output_at, audio=False)
	