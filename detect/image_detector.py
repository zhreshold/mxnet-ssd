from __future__ import print_function
import mxnet as mx
import numpy as np
from timeit import default_timer as timer
from dataset.iterator import DetTestImageIter
import cv2

class ImageDetector(object):
	"""
	SSD detector which hold a detection network and wraps detection API

	Parameters:
	----------
	symbol : mx.Symbol
		detection network Symbol
	model_prefix : str
		name prefix of trained model
	epoch : int
		load epoch of trained model
	data_shape : int
		input data resize shape
	mean_pixels : tuple of float
		(mean_r, mean_g, mean_b)
	batch_size : int
		run detection with batch size
	ctx : mx.ctx
		device to use, if None, use mx.cpu() as default context
	"""
	def __init__(self, symbol, model_prefix, epoch, data_shape, mean_pixels, \
					classes, batch_size=1, ctx=None):
		self.ctx = ctx
		if self.ctx is None:
			self.ctx = mx.cpu()
		load_symbol, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
		if symbol is None:
			symbol = load_symbol
		self.mod = mx.mod.Module(symbol, label_names=None, context=ctx)
		self.data_shape = data_shape
		self.mod.bind(data_shapes=[('data', (batch_size, 3, data_shape, data_shape))])
		self.mod.set_params(args, auxs)
		self.data_shape = data_shape
		self.mean_pixels = mean_pixels
		self.classes = classes
		self.colors = []
		self.fill_random_colors()

	def fill_random_colors(self):
		import random
		for i in range(len(self.classes)):
			self.colors.append((random.random(), random.random(), random.random()))

		print(self.colors)

	def fill_random_colors_int(self):
		import random
		for i in range(len(self.classes)):
			self.colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

		print(self.colors)


	def detect(self, det_iter, show_timer=False):
		"""
		detect all images in iterator

		Parameters:
		----------
		det_iter : DetIter
			iterator for all testing images
		show_timer : Boolean
			whether to print out detection exec time

		Returns:
		----------
		list of detection results
		"""
		num_images = det_iter._size
		result = []
		detections = []
		#if not isinstance(det_iter, mx.io.PrefetchingIter):
		#	det_iter = mx.io.PrefetchingIter(det_iter)
		start = timer()
		for pred, _, _ in self.mod.iter_predict(det_iter):
			detections.append(pred[0].asnumpy())
		time_elapsed = timer() - start
		if show_timer:
			print("Detection time for {} images: {:.4f} sec".format(num_images, time_elapsed))
		for output in detections:
			for i in range(output.shape[0]):
				det = output[i, :, :]
				res = det[np.where(det[:, 0] >= 0)[0]]
				result.append(res)
		resized_img = det_iter.current_data()
		return result, resized_img

	def im_detect(self, img, show_timer=False):
		"""
		wrapper for detecting multiple images

		Parameters:
		----------
		im_list : list of str
			image path or list of image paths
		root_dir : str
			directory of input images, optional if image path already
			has full directory information
		extension : str
			image extension, eg. ".jpg", optional

		Returns:
		----------
		list of detection results in format [det0, det1...], det is in
		format np.array([id, score, xmin, ymin, xmax, ymax]...)
		"""
		im_list = [img]
		test_iter = DetTestImageIter(im_list, 1, self.data_shape, self.mean_pixels)
		return self.detect(test_iter, show_timer)

	def plot_rects(self, img, dets, thresh=0.6):
		img_shape = img.shape
		for i in range(dets.shape[0]):
			cls_id = int(dets[i, 0])
			if cls_id >= 0:
				score = dets[i, 1]
				#print('Score is {}, class {}'.format(score, cls_id))
				if score > thresh:
					xmin = int(dets[i, 2] * img_shape[1])
					ymin = int(dets[i, 3] * img_shape[0])
					xmax = int(dets[i, 4] * img_shape[1])
					ymax = int(dets[i, 5] * img_shape[0])

					cv2.rectangle(img, (xmin, ymin), (xmax, ymax), self.colors[cls_id], 4)

					class_name = self.classes[cls_id]
					cv2.putText(img, class_name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
					print('Class id = {}, Score = {}, Country = {}, rect = ({}, {}, {}, {})'.format(cls_id, score, class_name, xmin, ymin, xmax, ymax))

	def detect_and_visualize_image(self, img, thresh=0.6, show_timer=False):
		"""
		wrapper for im_detect and visualize_detection

		Parameters:
		----------
		im_list : list of str or str
		image path or list of image paths
		root_dir : str or None
		directory of input images, optional if image path already
		has full directory information
		extension : str or None
		image extension, eg. ".jpg", optional

		Returns:
		----------

		"""
		dets, resized_img = self.im_detect(img, show_timer=show_timer)
		resized_img = resized_img.asnumpy()
		resized_img /= 255.0
		for k, det in enumerate(dets):
			self.plot_rects(resized_img, det, thresh)
		return resized_img

	def scale_and_plot_rects(self, img, dets, thresh=0.6):
		img_shape = img.shape
		for i in range(dets.shape[0]):
			cls_id = int(dets[i, 0])
			if cls_id >= 0:
				score = dets[i, 1]
				#print('Score is {}, class {}'.format(score, cls_id))
				if score > thresh:
					xmin = int(dets[i, 2] * img_shape[1])
					ymin = int(dets[i, 3] * img_shape[0])
					xmax = int(dets[i, 4] * img_shape[1])
					ymax = int(dets[i, 5] * img_shape[0])

					cv2.rectangle(img, (xmin, ymin), (xmax, ymax), self.colors[cls_id], 4)

					class_name = self.classes[cls_id]
					cv2.putText(img, class_name, (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)
					score_color = (0, 255, 0) if score > 0.5 else (255, 0, 0)
					cv2.putText(img, '{:.3f}'.format(score), (xmax - 60, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, score_color, 1)
					if score < 0.5:
						print('Class id = {}, Score = {}, Thresh = {}'.format(cls_id, score, thresh))

	def detect_and_layover_image(self, img, thresh=0.6, show_timer=False):
		"""
		wrapper for im_detect and visualize_detection

		Parameters:
		----------
		im_list : list of str or str
		image path or list of image paths
		root_dir : str or None
		directory of input images, optional if image path already
		has full directory information
		extension : str or None
		image extension, eg. ".jpg", optional

		Returns:
		----------

		"""
		dets, _ = self.im_detect(img, show_timer=show_timer)
		for k, det in enumerate(dets):
			self.scale_and_plot_rects(img, det, thresh)
		return img
