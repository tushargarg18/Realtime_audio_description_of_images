import pickle
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


# -----------------------------------------------------------------------------------------------
# Caption Text Processing & Data Preparation
# -----------------------------------------------------------------------------------------------
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

    # Get the list of all image names
    all_images = list(caption_data.keys())

    # Shuffle if necessary
    if shuffle:
        np.random.shuffle(all_images)

    # Split into training and validation sets
    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }

    # Return the splits
    return training_data, validation_data


# Load the dataset
captions_mapping, text_data = load_captions_data("/mnt/d/DIT/First Sem/Computer Vision/EchoLens/DataSet/captions.txt")

# Split the dataset into training and validation sets
train_data, valid_data = train_val_split(captions_mapping)
print("Number of training samples: ", len(train_data))
print("Number of validation samples: ", len(valid_data))


def custom_standardization(input_string):
    """
    Applies custom text standardization by converting text to lowercase
    and removing specified characters using regular expressions.

    Args:
        input_string (tf.Tensor): A scalar string tensor (typically a line of text).

    Returns:
        tf.Tensor: A standardized string tensor with lowercase letters and stripped characters.
    """
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

# Define the characters to strip from the text
strip_chars = r"!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
strip_chars = strip_chars.replace("<", "")
strip_chars = strip_chars.replace(">", "")

# Create a TextVectorization layer
# This layer will convert text to sequences of integers
vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
)
vectorization.adapt(text_data)

# Data augmentation for image data
image_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.3),
    ]
)


def decode_and_resize(img_path):
    """
    Reads an image from a file path, decodes it, resizes it, and normalizes pixel values.

    Args:
        img_path (tf.Tensor): A scalar string tensor representing the file path to the image.

    Returns:
        tf.Tensor: A float32 tensor of shape (IMAGE_SIZE[0], IMAGE_SIZE[1], 3) with pixel values in [0, 1].
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def process_input(img_path, captions):
    """
    Processes a single data sample consisting of an image path and its corresponding caption.

    Args:
        img_path (tf.Tensor): A scalar string tensor representing the path to an image file.
        captions (tf.Tensor): A scalar string tensor containing the text caption for the image.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing:
            - The preprocessed image tensor (float32, resized, normalized).
            - The vectorized caption tensor (integer sequence).
    """
    return decode_and_resize(img_path), vectorization(captions)


def make_dataset(images, captions):
    """
    Creates a TensorFlow dataset pipeline from lists of image paths and captions.

    Args:
        images (List[str] or tf.Tensor): List or tensor of image file paths.
        captions (List[str] or tf.Tensor): List or tensor of corresponding image captions.

    Returns:
        tf.data.Dataset: A batched, shuffled, and preprocessed dataset ready for training.
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.shuffle(BATCH_SIZE * 8)
    dataset = dataset.map(process_input, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset


# Pass the list of images and the list of corresponding captions
train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))

valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()))
# -----------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------
# Encoder Block
# -----------------------------------------------------------------------------------------------
def get_cnn_model():
    """
    Loads a pre-trained EfficientNetB0 model, freezes its weights, and adapts its output
    for use in image captioning by reshaping the feature map.

    Returns:
        keras.Model: A CNN model that outputs a reshaped feature map of the input image.
    """
    base_model = efficientnet.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    # We freeze our feature extractor
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model

@keras.utils.register_keras_serializable()
class TransformerEncoderBlock(layers.Layer):
    """
    A custom Transformer encoder block with:
    - Layer normalization before attention and feedforward layers (Pre-LN architecture)
    - Multi-head self-attention
    - Position-wise dense feedforward layer

    This block can be stacked to form a Transformer encoder.

    Args:
        embed_dim (int): Dimensionality of the input embeddings.
        dense_dim (int): Dimensionality of the internal dense layer (unused in current code but often used in FFN).
        num_heads (int): Number of attention heads.

    Methods:
        call(inputs, training, mask=None): Forward pass through the encoder block.
    """
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.0
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(embed_dim, activation="relu")

    def call(self, inputs, training, mask=None):
        """
        Forward pass of the Transformer Encoder Block.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            training (bool): Flag indicating training mode (for dropout, if used)
            mask (tf.Tensor or None): Optional attention mask (currently unused)

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        inputs = self.layernorm_1(inputs)
        inputs = self.dense_1(inputs)

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=None,
            training=training,
        )
        out_1 = self.layernorm_2(inputs + attention_output_1)
        return out_1

@keras.utils.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    """
    This layer combines token embeddings with learned positional embeddings.
    It maps input token IDs to embeddings and adds positional information
    so the model can understand token order.

    Args:
        sequence_length (int): Maximum length of input sequences.
        vocab_size (int): Size of the vocabulary (number of unique tokens).
        embed_dim (int): Dimensionality of the token embeddings.

    Methods:
        call(inputs): Computes combined token and position embeddings.
        compute_mask(inputs, mask=None): Creates a mask to ignore padding tokens.
    """
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, inputs):
        """
        Forward pass for the positional embedding layer.

        Args:
            inputs (tf.Tensor): Tensor of shape (batch_size, seq_len) with token IDs.

        Returns:
            tf.Tensor: Combined token + position embeddings, shape (batch_size, seq_len, embed_dim).
        """
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_tokens = embedded_tokens * self.embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        """
        Compute the mask tensor to ignore padding tokens (assumed to be 0).

        Args:
            inputs (tf.Tensor): Input token IDs.

        Returns:
            tf.Tensor: Boolean mask tensor where True indicates non-padding tokens.
        """
        return tf.math.not_equal(inputs, 0)
# -----------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------
# Decoder Block
# -----------------------------------------------------------------------------------------------
@keras.utils.register_keras_serializable()
class TransformerDecoderBlock(layers.Layer):
    """
    Transformer Decoder Block that includes:
    - Positional embedding layer for inputs
    - Masked multi-head self-attention (causal attention to prevent future lookahead)
    - Multi-head attention over encoder outputs (encoder-decoder attention)
    - Feed-forward network with dropout and residual connections
    - Layer normalizations applied pre/post attention and FFN layers

    Args:
        embed_dim (int): Dimensionality of embeddings.
        ff_dim (int): Dimensionality of the feed-forward network intermediate layer.
        num_heads (int): Number of attention heads.

    Methods:
        call(inputs, encoder_outputs, training, mask=None): Forward pass.
        get_causal_attention_mask(inputs): Generates mask to prevent attention to future tokens.
    """
    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        self.embedding = PositionalEmbedding(
            embed_dim=EMBED_DIM,
            sequence_length=SEQ_LENGTH,
            vocab_size=VOCAB_SIZE,
        )
        self.out = layers.Dense(VOCAB_SIZE, activation="softmax")

        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, training, mask=None):
        """
        Forward pass of the Transformer decoder block.

        Args:
            inputs (tf.Tensor): Decoder input token IDs (batch_size, seq_len).
            encoder_outputs (tf.Tensor): Output from encoder (batch_size, enc_seq_len, embed_dim).
            training (bool): Whether in training mode (enables dropout).
            mask (tf.Tensor or None): Padding mask for inputs.

        Returns:
            tf.Tensor: Predicted token probabilities (batch_size, seq_len, vocab_size).
        """
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)

        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=combined_mask,
            training=training,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
            training=training,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        ffn_out = self.layernorm_3(ffn_out + out_2, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds

    def get_causal_attention_mask(self, inputs):
        """
        Generate a causal mask to prevent attention to future tokens.

        Args:
            inputs (tf.Tensor): Input tensor (batch_size, seq_len, embed_dim).

        Returns:
            tf.Tensor: A mask tensor of shape (batch_size, seq_len, seq_len) where
                       positions corresponding to future tokens are masked out.
        """
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [
                tf.expand_dims(batch_size, -1),
                tf.constant([1, 1], dtype=tf.int32),
            ],
            axis=0,
        )
        return tf.tile(mask, mult)


@keras.utils.register_keras_serializable()
class ImageCaptioningModel(keras.Model):
    """
    Custom Keras Model for Image Captioning.
    Combines a CNN-based image encoder, a sequence encoder, and a decoder
    to generate captions for images. Supports training on multiple captions
    per image, optional image augmentation, and handles loss and accuracy metrics.
    """
    def __init__(
        self,
        cnn_model,
        encoder,
        decoder,
        num_captions_per_image=5,
        image_aug=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.num_captions_per_image = num_captions_per_image
        self.image_aug = image_aug

    def calculate_loss(self, y_true, y_pred, mask):
        """
        Compute the masked loss between true and predicted captions.

        Args:
            y_true: Ground truth sequences.
            y_pred: Predicted sequences (logits or probabilities).
            mask: Boolean mask indicating valid tokens (non-padding).

        Returns:
            Scalar loss value averaged over valid tokens.
        """
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        """
        Compute the masked accuracy between true and predicted captions.

        Args:
            y_true: Ground truth sequences.
            y_pred: Predicted sequences (logits or probabilities).
            mask: Boolean mask indicating valid tokens (non-padding).

        Returns:
            Scalar accuracy value averaged over valid tokens.
        """
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):
        """
        Helper function to compute loss and accuracy for a batch of captions.

        Args:
            img_embed: Image embeddings from the CNN.
            batch_seq: Batch of captions (token sequences).
            training: Whether the call is for training or inference.

        Returns:
            Tuple of (loss, accuracy) for the batch.
        """
        encoder_out = self.encoder(img_embed, training=training)
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        mask = tf.math.not_equal(batch_seq_true, 0)
        batch_seq_pred = self.decoder(
            batch_seq_inp, encoder_out, training=training, mask=mask
        )
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        return loss, acc

    def train_step(self, batch_data):
        """
        Custom training step implementation.

        Args:
            batch_data: Tuple of (images, captions) for the batch.

        Returns:
            Dictionary with loss and accuracy metrics.
        """
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        if self.image_aug:
            batch_img = self.image_aug(batch_img)

        # 1. Get image embeddings
        img_embed = self.cnn_model(batch_img)

        # 2. Pass each of the five captions one by one to the decoder
        # along with the encoder outputs and compute the loss as well as accuracy
        # for each caption.
        for i in range(self.num_captions_per_image):
            with tf.GradientTape() as tape:
                loss, acc = self._compute_caption_loss_and_acc(
                    img_embed, batch_seq[:, i, :], training=True
                )

                # 3. Update loss and accuracy
                batch_loss += loss
                batch_acc += acc

            # 4. Get the list of all the trainable weights
            train_vars = (
                self.encoder.trainable_variables + self.decoder.trainable_variables
            )

            # 5. Get the gradients
            grads = tape.gradient(loss, train_vars)

            # 6. Update the trainable weights
            self.optimizer.apply_gradients(zip(grads, train_vars))

        # 7. Update the trackers
        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # 8. Return the loss and accuracy values
        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_tracker.result(),
        }

    def test_step(self, batch_data):
        """
        Custom testing/validation step implementation.

        Args:
            batch_data: Tuple of (images, captions) for the batch.

        Returns:
            Dictionary with loss and accuracy metrics.
        """
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        # 1. Get image embeddings
        img_embed = self.cnn_model(batch_img)

        # 2. Pass each of the five captions one by one to the decoder
        # along with the encoder outputs and compute the loss as well as accuracy
        # for each caption.
        for i in range(self.num_captions_per_image):
            loss, acc = self._compute_caption_loss_and_acc(
                img_embed, batch_seq[:, i, :], training=False
            )

            # 3. Update batch loss and batch accuracy
            batch_loss += loss
            batch_acc += acc

        batch_acc /= float(self.num_captions_per_image)

        # 4. Update the trackers
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # 5. Return the loss and accuracy values
        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_tracker.result(),
        }

    @property
    def metrics(self):
        """
        List of metrics to reset at the start of each epoch.

        Returns:
            List of Keras metrics.
        """
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]
    
    def get_config(self):
        """
        Return the configuration of the model for saving and loading.

        Returns:
            Dictionary containing model configuration.
        """
        config = super().get_config()
        config.update({
            "cnn_model": keras.saving.serialize_keras_object(self.cnn_model),
            "encoder": keras.saving.serialize_keras_object(self.encoder),
            "decoder": keras.saving.serialize_keras_object(self.decoder),
            "num_captions_per_image": self.num_captions_per_image,
            "image_aug": keras.saving.serialize_keras_object(self.image_aug),
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Instantiate the model from its configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            An instance of ImageCaptioningModel.
        """
        # Extract and deserialize components
        cnn_model = keras.saving.deserialize_keras_object(config.pop("cnn_model"))
        encoder = keras.saving.deserialize_keras_object(config.pop("encoder"))
        decoder = keras.saving.deserialize_keras_object(config.pop("decoder"))
        image_aug = keras.saving.deserialize_keras_object(config.pop("image_aug"))
        num_captions_per_image = config.pop("num_captions_per_image")

        # Pass remaining config (base_config like name, trainable, etc.) via kwargs
        return cls(
            cnn_model=cnn_model,
            encoder=encoder,
            decoder=decoder,
            num_captions_per_image=num_captions_per_image,
            image_aug=image_aug,
            **config  # base keras.Model args like name, dtype, trainable
        )
    
    def build(self, input_shape, seq_len=SEQ_LENGTH):
        """
        Builds the model by running dummy data through all components.

        Args:
            input_shape: Shape tuple of the image input (batch, H, W, C).
            seq_len: Sequence length of captions (default SEQ_LENGTH).
        """
        # input_shape is a tuple like (batch_size, height, width, channels)
        dummy_images = tf.zeros(input_shape)
        
        if self.image_aug:
            dummy_images = self.image_aug(dummy_images)
        
        # 1. Pass through CNN to get image embeddings
        img_embed = self.cnn_model(dummy_images)
        
        # 2. Pass through encoder
        encoder_out = self.encoder(img_embed, training=False)
        
        # 3. Create dummy caption input: (batch_size, seq_len)
        dummy_caption = tf.zeros((input_shape[0], seq_len), dtype=tf.int32)
        dummy_mask = tf.ones_like(dummy_caption, dtype=tf.bool)
        
        # 4. Pass through decoder
        _ = self.decoder(dummy_caption, encoder_out, training=False, mask=dummy_mask)

        # Now the model is built
        super().build(input_shape)
#--- --------------------------------------------------------------------------------------------

# Initialize the model
cnn_model = get_cnn_model()
encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model,
    encoder=encoder,
    decoder=decoder,
    image_aug=image_augmentation,
)

# Define the loss function
cross_entropy = keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
    reduction=None,
)

# EarlyStopping criteria
#early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)


# Learning Rate Scheduler for the optimizer
@keras.utils.register_keras_serializable()
class LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_progress = global_step / warmup_steps
        warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
        return tf.cond(
            global_step < warmup_steps,
            lambda: warmup_learning_rate,
            lambda: self.post_warmup_learning_rate,
        )
    def get_config(self):
        return {
            "post_warmup_learning_rate": float(self.post_warmup_learning_rate),
            "warmup_steps": int(self.warmup_steps),
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Create a learning rate schedule
num_train_steps = len(train_dataset) * EPOCHS
num_warmup_steps = num_train_steps // 15
lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps)

# Compile the model
#caption_model.compile(optimizer=keras.optimizers.Adam(), loss=cross_entropy)
caption_model.compile(optimizer=keras.optimizers.Adam(lr_schedule), loss=cross_entropy)

caption_model.build(input_shape=(4, 299, 299, 3))  # batch size 4, Inception-style input
#caption_model.build(input_shape=[(None, 2048), (None, 20)])

# Fit the model
caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=valid_dataset,
    #callbacks=[early_stopping],
)


# Save the model weights
caption_model.save_weights("/mnt/d/DIT/First Sem/Computer Vision/EchoLens-Pretrained/caption_model.weights.h5")

#print(caption_model.weights)
caption_model.save('caption_model.keras')


# save the vectorization layer
# Save them using pickle
vectorization_save_path = "vectorization_layer_state.pkl"
with open(vectorization_save_path, "wb") as f:
    pickle.dump({'config': vectorization.get_config(), 'vocab': vectorization.get_vocabulary()}, f)
print(f"TextVectorization layer config and vocabulary saved to {vectorization_save_path}")
print("Vectorization Vocabulary shape: ", len(vectorization.get_vocabulary()))
print("Vectorization Config", vectorization.get_config())
