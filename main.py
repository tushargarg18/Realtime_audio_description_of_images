import tensorflow as tf
import os
import json
import pandas as pd
import re
import numpy as np
import time
import matplotlib.pyplot as plt
import collections
import random
import requests
from math import sqrt
from PIL import Image
from tqdm.auto import tqdm
import pickle


caption_location = '/mnt/d/DIT/First Sem/Computer Vision/EchoLens/DataSet/captions.txt'

captions = pd.read_csv(caption_location)
captions['image'] = captions['image'].apply(
    lambda x: f'/mnt/d/DIT/First Sem/Computer Vision/EchoLens/DataSet/Images/{x}')
captions.head()

def preprocess(text):
    #conver all text into lower
    text = text.lower()
    #remove all character from text that are not words and whitespace
    text = re.sub(r'[^\w\s]', '', text) 
    #replace multiple whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    #remove any leading or trailing whitespace from the text
    text = text.strip()
    #Add start and end token to the text at begining and end of the text respectively
    text = '[start] ' + text + ' [end]'
    return text

captions['caption'] = captions['caption'].apply(preprocess)
captions.head()

#maximum length of tokenized captions
MAX_LENGTH = 40 
#maximum number of unique tokens in vocabularly 
VOCABULARY_SIZE = 10000
#number of samples that will propagated through the network at once. 
BATCH_SIZE = 32
#shuffling the dataset
BUFFER_SIZE = 1000
#dimensions of the word embedding vector
EMBEDDING_DIM = 512
# number of units in the recurrent layers
UNITS = 512

#Keras preprocessing layer that transforms text into sequences of integers.
tokenizer = tf.keras.layers.TextVectorization(
    #set maximum number of tokens (words) that the tokenizer will keep
    max_tokens=VOCABULARY_SIZE,
    
    standardize=None,
    #specifies the length of the output sequences
    output_sequence_length=MAX_LENGTH)
# Adapting the Tokenizer to all caption
tokenizer.adapt(captions['caption'])


#layer that maps strings to integer indices.
word2idx = tf.keras.layers.StringLookup(
    #specifies a token that will be treated as a mask
    mask_token="",
    vocabulary=tokenizer.get_vocabulary())
#The vocabulary is obtained from the tokenizer using the get_vocabulary() method, which returns a list of strings
#representing the vocabulary in order of frequency (most frequent first).

idx2word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),
    invert=True)


#initializes a defaultdict with lists as the default value type
img_to_cap_vector = collections.defaultdict(list)
# loop iterates over each image-caption pair in the captions DataFrame
for img, cap in zip(captions['image'], captions['caption']):
    img_to_cap_vector[img].append(cap)
#Shuffling and Splitting Keys:

img_keys = list(img_to_cap_vector.keys())
random.shuffle(img_keys)

slice_index = int(len(img_keys)*0.8)
img_name_train_keys, img_name_val_keys = (img_keys[:slice_index],
                                          img_keys[slice_index:])
#Creating Training and Validation Data:
train_imgs = []
train_captions = []
for imgt in img_name_train_keys:
    capt_len = len(img_to_cap_vector[imgt])
    train_imgs.extend([imgt] * capt_len)
    train_captions.extend(img_to_cap_vector[imgt])

val_imgs = []
val_captions = []
for imgv in img_name_val_keys:
    capv_len = len(img_to_cap_vector[imgv])
    val_imgs.extend([imgv] * capv_len)
    val_captions.extend(img_to_cap_vector[imgv])


def load_data(img_path, caption):
    #reads the image file
    img = tf.io.read_file(img_path)
    #decodes the JPEG-encoded image into a 3D tensor
    img = tf.io.decode_jpeg(img, channels=3)
    #resizes the image to the desired dimensions
    img = tf.keras.layers.Resizing(299, 299)(img)
    #normalize
    img = img / 255.
    #tokenizes the caption using the tokenizer created earlier
    caption = tokenizer(caption)
    return img, caption

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_imgs, train_captions))

train_dataset = train_dataset.map(
    load_data, num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_imgs, val_captions))

val_dataset = val_dataset.map(
    load_data, num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


image_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomContrast(0.3),
    ]
)

def CNN_Encoder():
    #load Inceptionv3 model
    inception_v3 = tf.keras.applications.InceptionV3(
#Excludes the top (final) fully connected layers of the model
#This means the model will output feature maps instead of classification probabilities
        include_top=False,
        #leverage pre-learned features
        weights='imagenet'
    )
     # the weights of the InceptionV3 model will not be updated during training
    inception_v3.trainable = False

    output = inception_v3.output
    #Reshapes the output tensor
    output = tf.keras.layers.Reshape(
        (-1, output.shape[-1]))(output)
    # cnn_model = tf.keras.Sequential([
    # tf.keras.Input(shape=(100,)),
    # tf.keras.layers.Embedding(input_dim=10000, output_dim=128)])
    cnn_model = tf.keras.models.Model(inception_v3.input, output)
    return cnn_model


class TransformerEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense = tf.keras.layers.Dense(embed_dim, activation="relu")

    #Forward Pass (call method):
    def call(self, x, training):
        x = self.layer_norm_1(x)
        x = self.dense(x)

        attn_output = self.attention(
            query=x,
            value=x,
            key=x,
            attention_mask=None,
            training=training
        )

        x = self.layer_norm_2(x + attn_output)

        return x
    

class Embeddings(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.token_embeddings = tf.keras.layers.Embedding(
            vocab_size, embed_dim)
        self.position_embeddings = tf.keras.layers.Embedding(
            max_len, embed_dim, input_shape=(None, max_len))


    def call(self, input_ids):
        #input_ids: A tensor of token IDs representing the input sequences.
        length = tf.shape(input_ids)[-1]
        #A range of position IDs from 0 to length - 1 is created
        position_ids = tf.range(start=0, limit=length, delta=1)
        #adds a new axis to make position_ids a batch-compatible tensor of shape
        position_ids = tf.expand_dims(position_ids, axis=0)

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        return token_embeddings + position_embeddings
    

Embeddings(tokenizer.vocabulary_size(), EMBEDDING_DIM, MAX_LENGTH)(next(iter(train_dataset))[1]).shape

class TransformerDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, embed_dim, units, num_heads):
        super().__init__()
# embedding layer to create token and positional embeddings.
        self.embedding = Embeddings(
            tokenizer.vocabulary_size(), embed_dim, MAX_LENGTH)
# for self attention
        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
#for attending to the encoder's output
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        #three layer normalization layers

        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()
        #Dense layers for FF network and output layer
        self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)

        self.out = tf.keras.layers.Dense(tokenizer.vocabulary_size(), activation="softmax")
        #two dropout layers
        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.dropout_2 = tf.keras.layers.Dropout(0.5)


    def call(self, input_ids, encoder_output, training, mask=None):
        embeddings = self.embedding(input_ids)

        combined_mask = None
        padding_mask = None
        #Prepares the masks for attention mechanisms
        if mask is not None:
            causal_mask = self.get_causal_attention_mask(embeddings)
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        #Applies self-attention on the embeddings
        attn_output_1 = self.attention_1(
            query=embeddings,
            value=embeddings,
            key=embeddings,
            attention_mask=combined_mask,
            training=training
        )
        #Adds the input embeddings to the attention output and normalizes
        out_1 = self.layernorm_1(embeddings + attn_output_1)
        #Applies attention on the encoder output (cross-attention).
        attn_output_2 = self.attention_2(
            query=out_1,
            value=encoder_output,
            key=encoder_output,
            attention_mask=padding_mask,
            training=training
        )
        #Adds the previous output to the cross-attention output and normalizes

        out_2 = self.layernorm_2(out_1 + attn_output_2)
        #Feedforward network and dropout
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        ffn_out = self.layernorm_3(ffn_out + out_2)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds

    #creates a causal mask to ensure that each position can only attend to earlier positions and itself, preventing information leakage from future tokens
    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),  tf.constant([1, 1], dtype=tf.int32)],
            axis=0
        )
        return tf.tile(mask, mult)
    

@tf.keras.utils.register_keras_serializable()
class ImageCaptioningModel(tf.keras.Model):

    def __init__(self, cnn_model, encoder, decoder, image_aug=None):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.image_aug = image_aug
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.Mean(name="accuracy")

    #Loss Calculation

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)
#This method calculates the masked loss by applying the mask to the loss values
#and then computing the average loss per non-padding token

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)
#This method calculates the masked accuracy by comparing predicted tokens to 
#the ground truth tokens and applying the mask


    def compute_loss_and_acc(self, img_embed, captions, training=True):
        encoder_output = self.encoder(img_embed, training=True)
        y_input = captions[:, :-1]
        y_true = captions[:, 1:]
        mask = (y_true != 0)
        y_pred = self.decoder(
            y_input, encoder_output, training=True, mask=mask
        )
        loss = self.calculate_loss(y_true, y_pred, mask)
        acc = self.calculate_accuracy(y_true, y_pred, mask)
        return loss, acc
#This method computes the loss and accuracy for a given batch by first encoding
#the image embeddings, preparing the input and target sequences for the decoder, and then calculating the loss and accuracy using the decoder's predictions.

    def train_step(self, batch):
        imgs, captions = batch

        if self.image_aug:
            imgs = self.image_aug(imgs)

        img_embed = self.cnn_model(imgs)

        with tf.GradientTape() as tape:
            loss, acc = self.compute_loss_and_acc(
                img_embed, captions
            )

        train_vars = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}
#This method performs a training step, including optional image augmentation, 
#forward pass, loss and accuracy computation, gradient computation, and model
 # weights update using the optimizer


    def test_step(self, batch):
        imgs, captions = batch

        img_embed = self.cnn_model(imgs)

        loss, acc = self.compute_loss_and_acc(
            img_embed, captions, training=False
        )

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}
#This method performs an evaluation step, similar to the training step but
#without gradient computation and weight updates.

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]
#This property returns the metrics used to track the loss and accuracy,
#allowing Keras to reset these metrics at the start of each epoch automatically

encoder = TransformerEncoderLayer(EMBEDDING_DIM, 1)
decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 8)

cnn_model = CNN_Encoder()
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=image_augmentation,
)

cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction="none"
)

early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

caption_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=cross_entropy
)


history = caption_model.fit(
    train_dataset,
    epochs=1,
    validation_data=val_dataset,
    callbacks=[early_stopping]
)

caption_model.save('image_captioning_model.keras')  # Will save in SavedModel format
# Save tokenizer as a pickle file
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
