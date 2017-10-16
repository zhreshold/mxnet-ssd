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
from IPython.display import HTML

IS_CLASSES_30 = True
INPUT_VIDEO_AT = './model/project_video.MOV'
OUTPUT_VIDEO_AT = './model/output_video.mp4'

network = 'vgg16_reduced'
class_names_30 = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria',\
	 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia',\
	  'BosniaandHerzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'BurkinaFaso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada']

class_names_194 = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria',\
	 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia',\
	  'BosniaandHerzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'BurkinaFaso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada',\
	   'CapeVerde', 'CentralAfricanRepublic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'CostaRica', 'Croatia', 'Cuba', 'Cyprus',\
	    'CzechRepublic', 'DemocraticRepublicoftheCongo', 'Denmark', 'Djibouti', 'DominicanRepublic', 'EastTimor', 'Ecuador', 'Egypt',\
		 'ElSalvador', 'EquatorialGuinea', 'Eritrea', 'Estonia', 'Ethiopia', 'FalklandIslands', 'FederatedStatesofMicronesia', 'Fiji',\
		  'Finland', 'France', 'FrenchGuiana', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Greenland', 'Guatemala',\
		   'Guinea', 'GuineaBissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq',\
		    'Ireland', 'Israel', 'Italy', 'IvoryCoast', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kuwait',\
			 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg',\
			  'Macedonia', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'MarshallIslands', 'Mauritania',\
			   'Mauritius', 'Mexico', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia',\
			    'Nauru', 'Nepal', 'Netherlands', 'NewCaledonia', 'NewZealand', 'Nicaragua', 'Niger', 'Nigeria', 'NorthKorea', 'Norway',\
				 'Oman', 'Pakistan', 'Palau', 'Palestine', 'Panama', 'PapuaNewGuinea', 'Paraguay', 'Peru', 'Philippines', 'Poland',\
				  'Portugal', 'PuertoRico', 'Qatar', 'RepublicofCongo', 'Romania', 'Russia', 'Rwanda', 'Samoa', 'SanMarino',\
				   'SaoTomeandPrincipe', 'SaudiArabia', 'Senegal', 'Serbia', 'Seychelles', 'SierraLeone', 'Singapore', 'Slovakia',\
				    'Slovenia', 'SolomonIslands', 'Somalia', 'SouthAfrica', 'SouthKorea', 'SouthSudan', 'Spain', 'SriLanka', 'Sudan',\
					 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria', 'Tajikistan', 'Tanzania', 'Thailand', 'Togo', 'Tonga',\
					  'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'UnitedArabEmirates', 'UnitedKingdom',\
					   'UnitedStates', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'VaticanCity', 'Venezuela', 'Vietnam', 'WesternSahara',\
					    'Yemen', 'Zambia', 'Zimbabwe']


epoch = 4 if IS_CLASSES_30 else 13
class_names = class_names_30 if IS_CLASSES_30 else class_names_194
cwd = os.getcwd()
prefix = cwd + '/model/ssd'
data_shape = 300
color_subtract = (123, 117, 104)
nms_thresh = 0.5
force_nms = True
show_timer = True
thresh = 0.43

def get_detector(net, prefix, epoch, data_shape, mean_pixels, ctx, class_names,
                 nms_thresh=0.5, force_nms=True, nms_topk=400):
	"""
	wrapper for initialize a detector

	Parameters:
	----------
	net : str
		test network name
	prefix : str
		load model prefix
	epoch : int
		load model epoch
	data_shape : int
		resize image shape
	mean_pixels : tuple (float, float, float)
		mean pixel values (R, G, B)
	ctx : mx.ctx
		running context, mx.cpu() or mx.gpu(?)
	num_class : int
		number of classes
	nms_thresh : float
		non-maximum suppression threshold
	force_nms : bool
		force suppress different categories
	"""
	if net is not None:
		net = get_symbol(net, data_shape, num_classes=len(class_names), nms_thresh=nms_thresh,
			force_nms=force_nms, nms_topk=nms_topk)
	detector = ImageDetector(net, prefix, epoch, data_shape, mean_pixels, class_names, ctx=ctx)
	#detector = Detector(net, prefix, epoch, data_shape, mean_pixels, ctx=ctx)
	return detector

detector = get_detector(network, prefix, epoch, data_shape, color_subtract, mx.gpu(0),
                            class_names, nms_thresh, force_nms)

def process_image(image_frame):
	# run detection
	detected_img = detector.detect_and_layover_image(image_frame, thresh, show_timer)
	return detected_img

if __name__ == '__main__':
	output_at = OUTPUT_VIDEO_AT
	clip1 = VideoFileClip(INPUT_VIDEO_AT)
	white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
	white_clip.write_videofile(output_at, audio=False)
	