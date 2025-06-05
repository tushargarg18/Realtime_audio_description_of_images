import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import re
import numpy as np
import matplotlib.pyplot as plt # Keep for potential visualization during training if needed

import tensorflow as tf
import keras
from keras import layers
import yaml
from keras.applications import efficientnet
from keras.layers import TextVectorization
import pickle # Added for saving/loading TextVectorization state

# Set random seed for reproducibility
keras.utils.set_random_seed(111)

# Load configuration from YAML file
with open("/mnt/d/DIT/First Sem/Computer Vision/EchoLens-Pretrained/EchoLens/src/config.yaml", "r") as f:
    config_yaml = yaml.safe_load(f)

# Global parameters from config
IMAGES_PATH = config_yaml["data"]["IMAGES_PATH"]
# IMAGE_SIZE = config_yaml["image_preprocessing"]["IMAGE_SIZE"]
IMAGE_SIZE = (299, 299)
VOCAB_SIZE = config_yaml["training"]["VOCAB_SIZE"]
SEQ_LENGTH = config_yaml["training"]["SEQ_LENGTH"]
EMBED_DIM = config_yaml["training"]["EMBED_DIM"]
FF_DIM = config_yaml["training"]["FF_DIM"]
BATCH_SIZE = config_yaml["training"]["BATCH_SIZE"]
EPOCHS = config_yaml["training"]["EPOCHS"]
AUTOTUNE = tf.data.AUTOTUNE

# --- Data Loading and Preprocessing Functions ---

def load_captions_data(filename):
    """Loads captions (text) data and maps them to corresponding images."""
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = {}
        text_data = []
        images_to_skip = set()

        for line in caption_data:
            line = line.rstrip("\n")
            img_name, caption = line.strip().lower().split(",", 1)
            img_name = '/mnt/d/DIT/First Sem/Computer Vision/EchoLens/DataSet/Images/' + img_name

            tokens = caption.strip().split()
            if len(tokens) < 5 or len(tokens) > SEQ_LENGTH:
                images_to_skip.add(img_name)
                continue

            if img_name.endswith("jpg") and img_name not in images_to_skip:
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
    """Split the captioning dataset into train and validation sets."""
    all_images = list(caption_data.keys())
    if shuffle:
        np.random.shuffle(all_images)
    train_size_count = int(len(caption_data) * train_size)
    training_data = {img_name: caption_data[img_name] for img_name in all_images[:train_size_count]}
    validation_data = {img_name: caption_data[img_name] for img_name in all_images[train_size_count:]}
    return training_data, validation_data

def custom_standardization(input_string):
    # strip_chars needs to be defined within the function or passed as an argument
    # or defined globally before this function.
    # Based on your original code, it was defined globally later.
    # For robustness, let's define it inside or ensure it's globally available.
    strip_chars = r"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
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

def process_input(img_path, captions):
    return decode_and_resize(img_path), vectorization(captions)

def make_dataset(images, captions):
    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.shuffle(BATCH_SIZE * 8)
    dataset = dataset.map(process_input, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset

# Load the dataset
captions_mapping, text_data = load_captions_data("/mnt/d/DIT/First Sem/Computer Vision/EchoLens/DataSet/captions.txt")

# Split the dataset into training and validation sets
train_data, valid_data = train_val_split(captions_mapping)
print("Number of training samples: ", len(train_data))
print("Number of validation samples: ", len(valid_data))

# --- TextVectorization Layer Initialization and Adaptation ---
# Define the TextVectorization layer
vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
)
# Adapt the layer to your text data to build the vocabulary
vectorization.adapt(text_data)

# # Save the TextVectorization layer
# vectorization_save_path = "vectorization_layer.keras"
# vectorization.save(vectorization_save_path)
# print(f"TextVectorization layer saved to {vectorization_save_path}")

# --- Saving the TextVectorization layer's config and vocabulary ---
# Get the config and vocabulary
vectorization_config = vectorization.get_config()
vectorization_vocab = vectorization.get_vocabulary()

# Save them using pickle
vectorization_save_path = "vectorization_layer_state.pkl"
with open(vectorization_save_path, "wb") as f:
    pickle.dump({'config': vectorization_config, 'vocab': vectorization_vocab}, f)
print(f"TextVectorization layer config and vocabulary saved to {vectorization_save_path}")

# Create TensorFlow datasets
train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))
valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()))


# --- Model Architecture Definitions ---
def get_cnn_model():
    base_model = efficientnet.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    base_model_out = base_model.output
    # Ensure the output shape is compatible with the encoder
    # EfficientNetB0 output is typically (None, H, W, Channels)
    # Reshape to (None, H*W, Channels) for sequence processing if needed,
    # or use GlobalAveragePooling2D to get (None, Channels)
    # Assuming encoder expects (batch_size, features_dim)
    base_model_out = layers.GlobalAveragePooling2D()(base_model_out) # Get (None, 1280) for EfficientNetB0
    # Add a dense layer to match EMBED_DIM if necessary, or ensure EMBED_DIM matches CNN output
    # For simplicity, assuming EMBED_DIM is 1280 or encoder can handle it.
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model


class PositionalEmbedding(layers.Layer):
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
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_tokens = embedded_tokens * self.embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

class TransformerEncoderBlock(layers.Layer):
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
        self.dense_1 = layers.Dense(embed_dim, activation="relu") # This might be the FFN layer, not just a dense layer for input

    def call(self, inputs, training, mask=None):
        # Assuming inputs are already in the correct shape for attention (batch, sequence_length, embed_dim)
        # If inputs from CNN are (batch, features_dim), they might need to be expanded to (batch, 1, features_dim)
        # for MultiHeadAttention if it expects a sequence.
        # Based on your original code, CNN output is directly passed to encoder.
        # If CNN output is (None, 2048), then the attention needs a sequence dimension.
        # Let's assume the encoder expects (batch, 1, embed_dim) where 1 is the sequence length for the image feature.
        # If CNN output is (None, 2048), and embed_dim is 2048, then inputs can be (batch, embed_dim)
        # and we need to add a sequence dim for attention.
        inputs_reshaped_for_attention = tf.expand_dims(inputs, 1) # (batch, 1, embed_dim)

        normalized_inputs = self.layernorm_1(inputs_reshaped_for_attention)
        dense_output = self.dense_1(normalized_inputs) # FFN part

        attention_output_1 = self.attention_1(
            query=dense_output, # Query from FFN output
            value=dense_output, # Value from FFN output
            key=dense_output,   # Key from FFN output
            attention_mask=None, # Self-attention, no padding mask needed here usually
            training=training,
        )
        # Residual connection and normalization
        out_1 = self.layernorm_2(dense_output + attention_output_1)
        return tf.squeeze(out_1, axis=1) # Squeeze back to (batch, embed_dim) if needed by decoder


class TransformerDecoderBlock(layers.Layer):
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
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)

        combined_mask = causal_mask
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32) # For cross-attention
            combined_mask = tf.minimum(tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32), causal_mask)

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
            attention_mask=padding_mask if mask is not None else None, # Use padding mask for cross-attention
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


class ImageCaptioningModel(keras.Model):
    def __init__(
        self,
        cnn_model,
        encoder,
        decoder,
        num_captions_per_image=5,
        image_aug=None,
        **kwargs # Added kwargs to pass to super().__init__
    ):
        super().__init__(**kwargs) # Pass kwargs to parent constructor
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.num_captions_per_image = num_captions_per_image
        self.image_aug = image_aug

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):
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
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]


# Define the loss function
cross_entropy = keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
    reduction=None,
)

# EarlyStopping criteria
early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)


# Learning Rate Scheduler for the optimizer
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


# Create a learning rate schedule
num_train_steps = len(train_dataset) * EPOCHS
num_warmup_steps = num_train_steps // 15
lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps)

image_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.3),
    ]
    )
#checkpoint_path = "/mnt/d/DIT/First Sem/Computer Vision/EchoLens-Pretrained/caption_model.weights.h5"
cnn_model = get_cnn_model()
encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)
caption_model = ImageCaptioningModel(
cnn_model=cnn_model,
encoder=encoder,
decoder=decoder,
image_aug=image_augmentation,
)

# Compile the model
caption_model.compile(optimizer=keras.optimizers.Adam(lr_schedule), loss=cross_entropy)

# Fit the model
print("Starting model training...")
caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=valid_dataset,
    callbacks=[early_stopping],
)
print("Model training complete.")

# --- Saving the trained model and TextVectorization layer ---
model_save_path = "image_captioning_model.keras"
caption_model.save(model_save_path)
print(f"Image Captioning Model saved to {model_save_path}")

# vectorization_save_path = "vectorization_layer.keras"
# vectorization.save(vectorization_save_path)
# print(f"TextVectorization layer saved to {vectorization_save_path}")
