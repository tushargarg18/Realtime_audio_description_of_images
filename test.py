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
from image_loader import load_captions_data, train_val_split, custom_standardization, decode_and_resize
from decoder import TransformerDecoderBlock
from encoder import TransformerEncoderBlock, get_cnn_model, PositionalEmbedding
from image_captioning_model import ImageCaptioningModel
import pickle
from keras.layers import TextVectorization

#from train import LRSchedule

from keras.models import load_model

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


# Initialize the model
image_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.3),
    ]
    )
checkpoint_path = "/mnt/d/DIT/First Sem/Computer Vision/EchoLens-Pretrained/caption_model.weights.h5"
cnn_model = get_cnn_model()
encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)
caption_model = ImageCaptioningModel(
cnn_model=cnn_model,
encoder=encoder,
decoder=decoder,
image_aug=image_augmentation,
)
#caption_model = ImageCaptioningModel()
caption_model.build([(None, 2048), (None, 20)])
caption_model.load_weights(checkpoint_path)

def custom_standardization(input_string):
    strip_chars = r"!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    strip_chars = strip_chars.replace("<", "")
    strip_chars = strip_chars.replace(">", "")
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

# Initialize the TextVectorization layer
with open("/mnt/d/DIT/First Sem/Computer Vision/EchoLens-Pretrained/vectorization_layer_state.pkl", "rb") as f:
    vectorization_loaded_data = pickle.load(f)
config, vocab = vectorization_loaded_data['config'], vectorization_loaded_data['vocab']
print(len(vectorization_loaded_data))
print(vectorization_loaded_data.keys())
print("vocab", len(vocab))
print("config", config)



# if callable(config.get("standardize")): # Check if it's a function object
#     config["standardize"] = 'custom_standardization' # Set it to the string name

# # The key is to provide the actual function object in `custom_objects`
# # where the key is the name Keras expects (often the function's name as a string).
# custom_objects = {
#     "custom_standardization": 'custom_standardization',
#     # It's good practice to also include the layer class itself if you're deserializing
#     # a built-in layer, though often not strictly necessary if it's a standard Keras layer
#     # and the module path is correct. Including it ensures it's found.
#     #"TextVectorization": TextVectorization
# }

# full_keras_object_config = {
#         'class_name': 'TextVectorization', # The exact class name of the layer
#         'config': config,                  # The internal config dictionary you loaded
#         'module': 'keras.layers.preprocessing.text_vectorization', # The module path
#         'vocabulary': vocab, # The vocabulary list
#     }

# standardize_arg_from_config = config.pop("standardize", None)

#     # Manually create the TextVectorization layer instance
#     # Pass the actual custom_standardization function directly to the standardize argument
#     # Use the remaining config items as kwargs
# if standardize_arg_from_config == 'custom_standardization':
#     # If the saved config indicated custom standardization, use the actual function
#     vectorization = TextVectorization(standardize=custom_standardization, **config)
# elif standardize_arg_from_config is not None:
#     # If it was a built-in standardization string (e.g., 'lower_and_strip_punctuation')
#     vectorization = TextVectorization(standardize=standardize_arg_from_config, **config)
# else:
#     # If standardize was not specified in the original config
#     vectorization = TextVectorization(**config)

# print(vectorization.get_config())

# if weights and isinstance(weights, list) and len(weights) > 0:
#         vocab_weights = weights[0] # The vocabulary list is usually the first item
#         vectorization.set_vocabulary(vocab_weights)
# else:
#     print("Warning: Could not extract vocabulary from 'vectorizer.pkl'. TextVectorization table might not be fully initialized.")

# vectorization = tf.keras.layers.deserialize(full_keras_object_config, custom_objects=custom_objects)
# vectorization = TextVectorization.from_config(config, custom_objects=custom_objects)
# vectorization.set_weights(weights)

vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
    vocabulary=vocab,
)

# vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 1

# Load the dataset
captions_mapping, text_data = load_captions_data("/mnt/d/DIT/First Sem/Computer Vision/EchoLens/DataSet/captions.txt")

# Split the dataset into training and validation sets
train_data, valid_data = train_val_split(captions_mapping)
print("Number of training samples: ", len(train_data))
print("Number of validation samples: ", len(valid_data))

valid_images = list(valid_data.keys())

def generate_caption():
    # Select a random image from the validation dataset
    sample_img = np.random.choice(valid_images)

    # Read the image from the disk
    sample_img = decode_and_resize(sample_img)
    img = sample_img.numpy().clip(0, 255).astype(np.uint8)
    print("Captioning image: ", img)
    plt.imshow(img)
    plt.show()
    # Data augmentation for image data
    
    #custom_objects = {'LRSchedule': LRSchedule}
    #caption_model = load_model("caption_model.keras", custom_objects={
    #"ImageCaptioningModel": ImageCaptioningModel,
    #"LRSchedule": LRSchedule,
    #})
    #caption_model = keras.saving.load_model("caption_model.keras")

    # Pass the image to the CNN
    img = tf.expand_dims(sample_img, 0)
    img = caption_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        print("prediction", predictions.shape)
        sampled_token_index = np.argmax(predictions[0, i, :])
        
        sampled_token = index_lookup.get(sampled_token_index, "<unk>")
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    print("Predicted Caption: ", decoded_caption)


# Check predictions for a few samples
generate_caption()
generate_caption()
generate_caption()
