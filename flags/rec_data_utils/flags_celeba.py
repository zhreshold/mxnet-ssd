from __future__ import print_function
import os
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from sklearn.utils import shuffle

class FlagsCeleba(object):
	def __init__(self, image_path, annotation_path, class_name_path, shuffle = True):
		self.image_path = image_path
		self.annotation_path = annotation_path

		self.classes = self.load_class_names(class_name_path)
		self.num_classes = len(self.classes)
		self.image_set_index = self.load_image_set_indexes(self.annotation_path, shuffle)
		self.num_images = len(self.image_set_index)
		self.labels = self._load_image_labels()

	@property
	def cache_path(self):
		"""
		make a directory to store all caches

		Returns:
		---------
			cache path
		"""
		cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache')
		if not os.path.exists(cache_path):
			os.mkdir(cache_path)
		return cache_path

	def load_class_names(self, class_name_path):
		with open(class_name_path, 'r') as file:
			class_names = file.readlines()
		class_names = [class_name.strip() for class_name in class_names]
		return class_names
	
	def load_image_set_indexes(self, annotation_path, should_shuffle):
		index_set = os.listdir(annotation_path)
		index_set = [name[:6] for name in index_set]
		if should_shuffle:
			index_set = shuffle(index_set)
		return index_set

	def _load_image_labels(self):
		"""
		preprocess all ground-truths

		Returns:
		----------
		labels packed in [num_images x max_num_objects x 5] tensor
		"""
		temp = []

		# load ground-truth from xml annotations
		for idx in self.image_set_index:
			label_file = self._label_path_from_index(idx)
			tree = ET.parse(label_file)
			root = tree.getroot()
			size = root.find('size')
			width = float(size.find('width').text)
			height = float(size.find('height').text)
			label = []

			for obj in root.iter('object'):
				difficult = int(obj.find('difficult').text)
				cls_id = obj.find('name').text
				xml_box = obj.find('bndbox')
				xmin = float(xml_box.find('xmin').text) / width
				ymin = float(xml_box.find('ymin').text) / height
				xmax = float(xml_box.find('xmax').text) / width
				ymax = float(xml_box.find('ymax').text) / height
				label.append([cls_id, xmin, ymin, xmax, ymax, difficult])
			temp.append(np.array(label, dtype = np.float32))
			if len(temp) % 4000 == 0:
				print("Reading at {}".format(len(temp)))
		return temp

	def image_path_from_index(self, index):
		"""
		load image full path given specified index
		pascal_voc import PascalVoc
		Parameters:
		----------
		index : int
			index of image requested in dataset

		Returns:
		----------
		full path of specified image
		"""
		image_name = self.image_set_index[index]
		assert self.image_set_index is not None, "Dataset not initialized"
		image_file = os.path.join(self.image_path, image_name + '.png')
		assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
		return image_file

	def label_from_index(self, index):
		assert self.labels is not None, "Labels not processed"
		return self.labels[index]

	def _label_path_from_index(self, index):
		"""
		load ground-truth of image given specified index

		Parameters:
		----------
		index : int
			index of image requested in dataset

		Returns:
		----------
		object ground-truths, in format
		numpy.array([id, xmin, ymin, xmax, ymax]...)
		"""
		label_file = os.path.join(self.annotation_path, index + '.xml')
		assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)
		return label_file

	def save_imglist(self, fname=None, root=None, shuffle=False):
		"""
		save imglist to disk

		Parameters:
		----------
		fname : str
			saved filename
		"""
		str_list = []
		for index in range(self.num_images):
			label = self.label_from_index(index)
			path = self.image_path_from_index(index)
			if root:
			    path = os.path.relpath(path, root)
			str_list.append('\t'.join([str(index), str(2), str(label.shape[1])] \
				+ ["{0:.4f}".format(x) for x in label.ravel()] + [path,]) + '\n')
		if str_list:
			if shuffle:
				import random
				random.shuffle(str_list)
			if not fname:
				fname = self.name + '.lst'
			with open(fname, 'w') as f:
				for line in str_list:
					f.write(line)
		else:
			raise RuntimeError("No image in imdb")