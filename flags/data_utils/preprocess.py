import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import os
import argparse
import shutil

from tqdm import tqdm
from constants import FLAG_HEIGHT, FLAG_WIDTH, IMAGE_SIZE

def is_file_png(img_path):
	return os.path.splitext(img_path)[1] == '.png'

def base_resize_images(src_folder, dest_folder, img_size, should_rename):
	shutil.rmtree(dest_folder, ignore_errors = True)
	os.mkdir(dest_folder)
	src_files = os.listdir(src_folder)

	tf.reset_default_graph()
	img_placeholder = tf.placeholder(tf.float32, (None, None, 3))
	tf_img = tf.image.resize_images(img_placeholder, img_size,
								tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		with tqdm(total = len(src_files))  as pbar:
			for file_index, file_name in enumerate(src_files):
				file_path = os.path.abspath(os.path.join(src_folder, file_name))
				img = mpimg.imread(file_path)[:, :, :3]
				# Jpeg images are read as uint type with data lying from 0 to 255
				# Convert them into float with values from 0.0 to 1.0
				if not is_file_png(file_path):
					img = img / 255.0
				resized_img = sess.run(tf_img, feed_dict = {img_placeholder: img})

				if should_rename:
					save_path = os.path.abspath(os.path.join(dest_folder, '{:06d}.png'.format(file_index + 1)))
				else:
					file_base_name = os.path.basename(file_name).split('.')[0]
					save_path = os.path.abspath(os.path.join(dest_folder, file_base_name + '.png'))
				mpimg.imsave(save_path, resized_img)

				pbar.update(1)

	return
	
def create_label_file(dest_folder):
	dest_path = os.path.abspath(dest_folder)
	country_name_file = os.path.abspath(os.path.join(dest_path, '..', 'class_names.txt'))

	files = os.listdir(dest_path)
	files = ['{}\n'.format(file.split('.')[0]) for file in files]
	files.sort()

	with open(country_name_file, mode = 'w') as label_file:
		for country_name in files:
			label_file.write(country_name)
	return country_name_file

def parse_args():
	# Make sure the file names are the country names preferrably without any spaces.
	parser = argparse.ArgumentParser(description = 'Resize data and prepare labels')
	parser.add_argument('src_folder', help = 'Where raw sized flags or background images are present', type = str)
	parser.add_argument('dest_folder', help = 'Where to store resized flags or background images', type = str)	
	
	parser.add_argument('--create-label-file', dest = 'create_label',
			 help = 'Should create label file (0=No, 1=Yes)', 
			 default = 1, type = int)
	parser.add_argument('--is-background', dest = 'is_bg_img',
			 help = 'Is background image folder (0=No, 1=Yes)', 
			 default = 0, type = int)
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()

	if args.is_bg_img == 1:
		base_resize_images(args.src_folder, args.dest_folder, (IMAGE_SIZE, IMAGE_SIZE), should_rename = True)
		print('Resizing of background image is complete')
	else:
		base_resize_images(args.src_folder, args.dest_folder, (FLAG_HEIGHT, FLAG_WIDTH), should_rename = False)
		print('Resizing of flag image is complete')

		if args.create_label == 1:
			label_path = create_label_file(args.dest_folder)
			print('Label file has been created at {}'.format(label_path))
