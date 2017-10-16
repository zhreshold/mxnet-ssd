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

network = 'vgg16_reduced'
class_names = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria',\
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

class_names = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria',\
	 'Azerbaijan']
epoch = 6
cwd = os.getcwd()
prefix = cwd + '/model/ssd'
data_shape = 300
color_subtract = (123, 117, 104)
nms_thresh = 0.5
force_nms = True
show_timer = True
thresh = 0.5

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
	detected_img = detector.detect_and_visualize_image(image_frame, thresh, show_timer)
	#detected_img = detector.detect_and_layover_image(image_frame, thresh, show_timer)
	return detected_img

def worker(input_q, output_q):
	fps = FPS().start()
	while True:
		fps.update()
		frame = input_q.get()
		#frame = np.transpose(frame, (2, 0, 1))
		#output_q.put(detect_objects(frame, sess, detection_graph))
		detected_img = process_image(frame)
		output_q.put(detected_img)

	fps.stop()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-src', '--source', dest='video_source', type=int,
						default=0, help='Device index of the camera.')
	parser.add_argument('-wd', '--width', dest='width', type=int,
						default=480, help='Width of the frames in the video stream.')
	parser.add_argument('-ht', '--height', dest='height', type=int,
						default=360, help='Height of the frames in the video stream.')
	parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
						default=1, help='Number of workers.')
	parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
						default=5, help='Size of the queue.')
	args = parser.parse_args()

	#time.sleep(1)

	#'''
	video_capture = WebcamVideoStream(src=args.video_source,
										width=args.width,
										height=args.height).start()
	fps = FPS().start()

	while True:  # fps._numFrames < 120
		frame = video_capture.read()
		detected_img = process_image(frame)

		t = time.time()

		cv2.imshow('Video', detected_img)
		fps.update()

		#print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break		
		

	fps.stop()
	#print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
	#print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

	video_capture.stop()
	cv2.destroyAllWindows()
	#'''


	'''
	logger = multiprocessing.log_to_stderr()
	#logger.setLevel(multiprocessing.SUBDEBUG)

	input_q = Queue(maxsize=args.queue_size)
	output_q = Queue(maxsize=args.queue_size)
	pool = Pool(args.num_workers, worker, (input_q, output_q))

	video_capture = WebcamVideoStream(src=args.video_source,
										width=args.width,
										height=args.height).start()
	fps = FPS().start()

	while True:  # fps._numFrames < 120
		frame = video_capture.read()
		input_q.put(frame)

		t = time.time()

		cv2.imshow('Video', output_q.get())
		fps.update()

		#print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break		
		

	fps.stop()
	#print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
	#print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

	pool.terminate()
	video_capture.stop()
	cv2.destroyAllWindows()
	'''
