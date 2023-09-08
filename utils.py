import os
import random
from dataset import DatasetGenerator
import cv2
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

CWD = os.path.dirname(__file__)

IMAGE_SIZE = (160, 160)

def train_test_split(images_dir, split):
    """
    Splits the dataset into training and testing sets based on a specified split ratio.

    Args:
        images_dir (str): Directory containing the dataset.
        split (float): Ratio of data to allocate for training (e.g., 0.8 for an 80/20 split).

    Returns:
        train_list (dict): Dictionary containing training folder names as keys and the number of files in each folder as values.
        test_list (dict): Dictionary containing testing folder names as keys and the number of files in each folder as values.
    """
    folders = os.listdir(images_dir)
    train_len = int(len(folders) * split)

    random.shuffle(folders)

    train_list, test_list = {}, {}

    for folder in folders[:train_len]:
        os_path = os.path.join(images_dir, folder)
        if os.path.isdir(os_path):
            num_files = len(os.listdir(os_path))
            train_list[folder] = num_files

    for folder in folders[train_len:]:
        os_path = os.path.join(images_dir, folder)
        if os.path.isdir(os_path):
            num_files = len(os.listdir(os_path))
            test_list[folder] = num_files

    return train_list, test_list


def cluster_data(image_dict):

    """
    Returns dataset for clustering.

    Args:
        image_dict (dict): Dictionary containing folder names as keys and the number of images in each folder as values.

    Returns:
        data_dict (list): List of folder names that meet the specified criteria.
        images (list): List of preprocessed images that meet the specified criteria.
        images_dir (list): List of tuples where each tuple contains folder name and image filename for selected images.
    """

    data_dict = [k for k, v in image_dict.items() if int(v) >= 10 and int(v) < 12]
    images = []
    images_dir = []
    for folder in data_dict:
        folder_path = os.path.join(CWD, 'data/Extracted Faces', folder)
        image_list = os.listdir(folder_path)
        for image in image_list:
            images_dir.append((folder, image))
            image_path = os.path.join(folder_path, image)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE)
            image = (image - 127.5)/128.0
            images.append(image)

    return data_dict, images, images_dir


# def read_image(image_path):

#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, IMAGE_SIZE)
#     image = (image - 127.5) / 128.0
#     return image