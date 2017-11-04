import argparse
from math import ceil, floor, pi, sqrt
from tqdm import tqdm
import numpy as np
import os
from sklearn.utils import shuffle
import shutil
import random

from data_utils.operations import tf_generate_images, write_label_file_entries
from data_utils.folder_names import XML_FOLDER, GENERATED_DATA, TRAIN_FOLDER, VAL_FOLDER, TEST_FOLDER, LABEL

# There are 202599 images in my CelebA dataset. Give this value appropriately.
CELEBA_TOTAL_FILES = 202599     # Directly hardcoded to save memory

BATCH_SIZE = 16

MIN_FLAGS = 1
MAX_FLAGS = 2  # Currently supports upto 2 Maximum flags in one image. 

BORDER_WHITE_AREA = 40 # How much percent of card should be covered with white area.

# Dimensions of the raw flag height and width
FLAG_HEIGHT = 144
FLAG_WIDTH = 224

def get_filenames_and_labels(flag_path):
    flag_file_names = os.listdir(flag_path)
    flag_file_names.sort()
    flag_file_names = ['{}/{}'.format(flag_path, flag_file) for flag_file in flag_file_names]

    labels = list(range(len(flag_file_names)))
    return flag_file_names, labels

def generate_image_pipeline(X_files, y_data, save_folder, folder_type, bg_img_folder,
							start_celeb_index, total_base_images,
                            scales = [0.40, 0.43, 0.46, 0.48, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
                            angles = [0], angle_repeat_ratio = [1]):
    # Folder for saving generated images.
    save_img_folder = '{}/{}_{}'.format(save_folder, GENERATED_DATA, folder_type)
    shutil.rmtree(save_img_folder, ignore_errors = True)
    os.mkdir(save_img_folder)
    # Folder for saving annotation XML files.
    save_xml_folder = '{}/{}_{}'.format(save_folder, XML_FOLDER, folder_type)
    shutil.rmtree(save_xml_folder, ignore_errors = True)
    os.mkdir(save_xml_folder)
    # File for saving labels.
    file_name = '{}/{}_{}.txt'.format(save_folder, LABEL, folder_type)
    if os.path.exists(file_name):
        os.unlink(file_name)
        
    # Counter indexes.
    save_index = 0  # Index for maintaining saved file number of newly generated file.
    celeb_index = start_celeb_index # Index for maintaining at which file index of bg_img is at currently. Loops after finishing.
    scale_index = 0  # Index for at which scale position of flag image. Loops after finishing.
    data_index = 0 # Index for maintaining at which flag image is currently at. Loops after finishing.
    
    data_samples = len(y_data)
    # Calculate the number of images to generate for each angle.
    angle_images = [ceil(total_base_images * ratio) for ratio in angle_repeat_ratio]
    total_images = sum(angle_images)
    with tqdm(total = total_images) as pbar:
        # Generate total images needed at each angle.
        for angle_at, images_at_angle in zip(angles, angle_images):
            save_image_at = 0
            while save_image_at < images_at_angle:
                # Get the scale index.
                if scale_index == len(scales):
                    scale_index = 0
                scale_at = scales[scale_index]
                scale_index += 1
                
                if data_index >= data_samples:
                    data_index = 0
                    
                no_of_files_array = []
                # Keep the ability of putting multiple flag files in one image only if scaling is below 0.5. 
                if scale_at <= 0.5:
                    for batch_counter in range(min(images_at_angle - save_image_at, BATCH_SIZE)):
                        files_to_pick = random.randint(MIN_FLAGS, MAX_FLAGS)
                        no_of_files_array.append(files_to_pick)
                else:
                    no_of_files_array = [MIN_FLAGS] * min(images_at_angle - save_image_at, BATCH_SIZE)
                no_of_files = sum(no_of_files_array)
                
                # Collect the needed number of flag files. 
                if data_index + no_of_files > data_samples:
                    # This condition deals with in case the looping of flag files array has to be done.
                    batch_X_files = X_files[data_index: ]
                    batch_y_data = y_data[data_index: ]
                    data_index = no_of_files - len(batch_y_data)
                    batch_X_files.extend(X_files[: data_index])
                    batch_y_data = np.concatenate((batch_y_data, y_data[: data_index]))
                    # If the data is not filled still.
                    if len(batch_y_data) != no_of_files:
                        data_index = no_of_files - len(batch_y_data)
                        batch_X_files.extend(X_files[: data_index])
                        batch_y_data = np.concatenate((batch_y_data, y_data[: data_index]))
                else:
                    batch_X_files = X_files[data_index: data_index + no_of_files]
                    batch_y_data = y_data[data_index: data_index + no_of_files]
                    data_index += no_of_files
                    
                # Some check to see if required number of flags files are collected.
                # Ideally, the assert condition should never fail.
                assert no_of_files == len(batch_X_files), 'Length mismatch in data files'
                assert no_of_files == len(batch_y_data), 'Length mismatch in label array'

                # As there are large number of parameters to pass, pass it in a dictionary.
                parameter_dict = {'scale_at': scale_at, 
                                    'angle_at': angle_at,
                                    'celeb_index_at': celeb_index,
                                    'save_index': save_index,
                                    'raw_flag_size': (FLAG_HEIGHT, FLAG_WIDTH),
                                    'no_of_files_array': no_of_files_array,
                                    'border_area': BORDER_WHITE_AREA,
                                    'bg_total_files': CELEBA_TOTAL_FILES}

                # Generate the batch of images.
                celeb_index, no_of_files_array, batch_y_data = tf_generate_images(batch_X_files, batch_y_data,
                                                                        bg_img_folder, save_img_folder,
                                                                        save_xml_folder, parameter_dict)

                write_label_file_entries(batch_y_data, no_of_files_array, save_folder, folder_type)
                save_index += len(no_of_files_array)
                save_image_at += len(no_of_files_array)
                pbar.update(len(no_of_files_array))


def parse_args():
	parser = argparse.ArgumentParser(description = 'Resize data and prepare labels')
	parser.add_argument('flag_folder', help = 'Where flags are present', type = str)
	parser.add_argument('bg_img_folder', help = 'Where background images are present', type = str)
	parser.add_argument('save_folder', help = 'Where the generated files are to be stored', type = str)
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()

	label_file_names, labels = get_filenames_and_labels(args.flag_folder)
	label_file_names, labels = shuffle(label_file_names, labels)

	generate_image_pipeline(label_file_names, labels, args.save_folder, TRAIN_FOLDER, args.bg_img_folder,
							1, total_base_images = 120000)
	generate_image_pipeline(label_file_names, labels, args.save_folder, TEST_FOLDER, args.bg_img_folder,
							190000, total_base_images = 10000)