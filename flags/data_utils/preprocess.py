import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import os
import argparse
import shutil

from write_xml_file import write_xml_file

FLAG_HEIGHT = 144
FLAG_WIDTH = 224

def tf_resize_images(img, image_width, image_height):
	tf.reset_default_graph()
	tf_img = tf.image.resize_images(img, (image_height, image_width),
								tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		resized_img = sess.run(tf_img)
	return resized_img


def base_resize_images(src_folder, dest_folder):
	shutil.rmtree(dest_folder, ignore_errors = True)
	os.mkdir(dest_folder)

	flag_files = os.listdir(src_folder)
	resize_batch_size = 32
	for offset in range(0, len(flag_files), resize_batch_size):
		batch_file_names = flag_files[offset: offset + resize_batch_size]
		batch_files = []
		for file_name in batch_file_names:
			file_path = os.path.join(src_folder, file_name)
			img = mpimg.imread(file_path)[:, :, :3]
			batch_files.append(img)
		batch_files = np.array(batch_files)

		batch_files = tf_resize_images(batch_files, FLAG_WIDTH, FLAG_HEIGHT)
		for file_name, file in zip(batch_file_names, batch_files):
			file_path = os.path.join(dest_folder, file_name)
			mpimg.imsave(file_path, file)

def create_label_file(dest_folder):
	dest_path = os.path.abspath(dest_folder)
	country_name_file = os.path.join(dest_path, '..', 'class_names.txt')

	files = os.listdir(dest_path)
	files = ['{}\n'.format(file.split('.')[0]) for file in files]
	files.sort()

	with open(country_name_file, mode = 'w') as label_file:
		for country_name in files:
			label_file.write(country_name)

def parse_args():
	# Make sure the file names are the country names preferrably without any spaces.
	parser = argparse.ArgumentParser(description = 'Resize data and prepare labels')
	parser.add_argument('src_folder', help = 'Where raw sized flags are present', type = str)
	parser.add_argument('dest_folder', help = 'Where to store resized flags', type = str)	

	parser.add_argument('--create-label-file', dest = 'create_label',
			 help = 'Should create label file(0=No, 1=Yes)', 
			 default = 1, type = int)
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()

	base_resize_images(args.src_folder, args.dest_folder)
	
	if args.create_label == 1:
		create_label_file(args.dest_folder)
