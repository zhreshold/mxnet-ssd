from __future__ import print_function
import sys, os
import argparse
import subprocess

from data_utils.folder_names import TRAIN_FOLDER, VAL_FOLDER, GENERATED_DATA, XML_FOLDER
from rec_data_utils.flags_celeba import FlagsCeleba

def load_flags(image_path, annotation_path, class_name_path, shuffle = True):
	flags_celeba = FlagsCeleba(image_path, annotation_path, class_name_path)
	return flags_celeba

def parse_args():
	parser = argparse.ArgumentParser(description='Prepare lists for dataset')
	parser.add_argument('data_path', help = 'Give path where your image folders are present', type = str)
	# Ensure that this path has GeneratedData_<Train,Val> folder, Annotations_<Train,Val> and class_names.txt
	#parser.add_argument('--dataset', dest = 'dataset', help = '1 for Train, 2 for Validation', default = 1, type = int)
	args = parser.parse_args()
	return args

def get_paths(dataset):
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
	base_path = args.data_path
	for dataset in range(2):
		image_path, annotation_path, list_save_name = get_paths(dataset)		

		class_name_path = os.path.join(base_path, 'class_names.txt')
		lst_path = base_path + '/' + list_save_name

		db = load_flags(image_path, annotation_path, class_name_path)
		db.save_imglist(lst_path, image_path)

		print("List file {} generated...".format(lst_path))

		curr_path = os.path.abspath(os.path.dirname(__file__))
		subprocess.check_call(["python",
			os.path.join(curr_path, "../mxnet/tools/im2rec.py"),
			os.path.abspath(lst_path), os.path.abspath(image_path),
			"--shuffle", str(1), "--pack-label", "1"])

		print("Record file {} generated...".format(list_save_name.split('.')[0] + '.rec'))
