import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import re
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers
import yaml
from keras.applications import efficientnet
from keras.layers import TextVectorization

keras.utils.set_random_seed(111)

with open("EchoLens/src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Path to the images
IMAGES_PATH = config["data"]["IMAGES_PATH"]
# Desired image dimensions
#IMAGE_SIZE = config["image_preprocessing"]["IMAGE_SIZE"]
IMAGE_SIZE = (299, 299)
# Vocabulary size
VOCAB_SIZE = config["training"]["VOCAB_SIZE"]
# Fixed length allowed for any sequence
SEQ_LENGTH = config["training"]["SEQ_LENGTH"]
# Dimension for the image embeddings and token embeddings
EMBED_DIM = config["training"]["EMBED_DIM"]
# Per-layer units in the feed-forward network
FF_DIM = config["training"]["FF_DIM"]
# Other training parameters
BATCH_SIZE = config["training"]["BATCH_SIZE"]
EPOCHS = config["training"]["EPOCHS"]
AUTOTUNE = tf.data.AUTOTUNE

def load_captions_data(filename):
    """Loads captions (text) data and maps them to corresponding images.

    Args:
        filename: Path to the text file containing caption data.

    Returns:
        caption_mapping: Dictionary mapping image names and the corresponding captions
        text_data: List containing all the available captions
    """

    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = {}
        text_data = []
        images_to_skip = set()

        for line in caption_data:
            line = line.rstrip("\n")
            # Image name and captions are separated using a tab
            img_name, caption = line.strip().lower().split(",", 1)
            img_name = '/mnt/d/DIT/First Sem/Computer Vision/EchoLens/DataSet/Images/' + img_name
            # Each image is repeated five times for the five different captions.
            # Each image name has a suffix `#(caption_number)`
            # img_name = img_name.split("#")[0]
            # img_name = os.path.join(IMAGES_PATH, img_name.strip())

            # We will remove caption that are either too short to too long
            tokens = caption.strip().split()

            if len(tokens) < 5 or len(tokens) > SEQ_LENGTH:
                images_to_skip.add(img_name)
                continue

            if img_name.endswith("jpg") and img_name not in images_to_skip:
                # We will add a start and an end token to each caption
                caption = "<start> " + caption.strip() + " <end>"
                text_data.append(caption)

                if img_name in caption_mapping:
                    caption_mapping[img_name].append(caption)
                else:
                    caption_mapping[img_name] = [caption]

        for img_name in images_to_skip:
            if img_name in caption_mapping:
                del caption_mapping[img_name]

        return caption_mapping, text_data
    
def train_val_split(caption_data, train_size=0.8, shuffle=True):
    """Split the captioning dataset into train and validation sets.

    Args:
        caption_data (dict): Dictionary containing the mapped caption data
        train_size (float): Fraction of all the full dataset to use as training data
        shuffle (bool): Whether to shuffle the dataset before splitting

    Returns:
        Traning and validation datasets as two separated dicts
    """

    # 1. Get the list of all image names
    all_images = list(caption_data.keys())

    # 2. Shuffle if necessary
    if shuffle:
        np.random.shuffle(all_images)

    # 3. Split into training and validation sets
    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }

    # 4. Return the splits
    return training_data, validation_data

def custom_standardization(input_string):
    strip_chars = r"!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    strip_chars = strip_chars.replace("<", "")
    strip_chars = strip_chars.replace(">", "")
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img