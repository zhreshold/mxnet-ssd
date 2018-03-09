from __future__ import print_function
import sys, os
import argparse
import subprocess

from data_utils.constants import TRAIN_FOLDER, VAL_FOLDER, GENERATED_DATA, XML_FOLDER
from rec_data_utils.flags_celeba import FlagsCeleba

def load_flags(image_path, annotation_path, class_name_path, shuffle = True):
	flags_celeba = FlagsCeleba(image_path, annotation_path, class_name_path)
	return flags_celeba

def parse_args():
	parser = argparse.ArgumentParser(description='Prepare lists for dataset')
	parser.add_argument('--data-path', dest = 'data_path', help = 'Give path where your image folders are present', 
            default = os.path.join(os.getcwd(), 'input_data'), type = str)
	# Ensure that this path has GeneratedData_<Train,Val> folder, Annotations_<Train,Val> and class_names.txt
	args = parser.parse_args()
	return args

def get_paths(dataset, base_path):
	image_path, annotation_path, list_save_name = '', '', ''
	# For train dataset
	if dataset == 0:
		image_path = os.path.join(base_path, '{}_{}'.format(GENERATED_DATA, TRAIN_FOLDER))
		annotation_path = os.path.join(base_path, '{}_{}'.format(XML_FOLDER, TRAIN_FOLDER))
		list_save_name = 'train.lst'
	# For Validation dataset
	elif dataset == 1:
		image_path = os.path.join(base_path, '{}_{}'.format(GENERATED_DATA, VAL_FOLDER))
		annotation_path = os.path.join(base_path, '{}_{}'.format(XML_FOLDER, VAL_FOLDER))
		list_save_name = 'val.lst'
	return image_path, annotation_path, list_save_name

if __name__ == "__main__":
	args = parse_args()
	input_data_path = args.data_path
	for dataset in range(2):
		image_path, annotation_path, list_save_name = get_paths(dataset, input_data_path)		

		class_name_path = os.path.join(input_data_path, 'class_names.txt')
		lst_path = input_data_path + '/' + list_save_name

		db = load_flags(image_path, annotation_path, class_name_path)
		db.save_imglist(lst_path, image_path)

		print("List file {} generated...".format(lst_path))

		curr_path = os.path.abspath(os.path.dirname(__file__))
		subprocess.check_call(["python",
			os.path.join(curr_path, "../mxnet/tools/im2rec.py"),
			os.path.abspath(lst_path), os.path.abspath(image_path),
			"--pack-label"])

		file_name = list_save_name.split('.')[0]
		print("Record file {} generated...".format(file_name + '.rec'))

		base_path_name = os.path.join(input_data_path, file_name)
		target_path = curr_path + '/../data/' + file_name
		os.rename(base_path_name + '.rec', target_path + '.rec')
		os.rename(base_path_name + '.idx', target_path + '.idx')
		os.rename(base_path_name + '.lst', target_path + '.lst')
		print("Record file moved to {}".format(target_path))
