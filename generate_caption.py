import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
from main import ImageCaptioningModel

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


caption_model = tf.keras.models.load_model(
    "/mnt/d/DIT/First Sem/Computer Vision/EchoLens-Pretrained/image_captioning_model.keras",
    custom_objects={"ImageCaptioningModel": ImageCaptioningModel},
    compile=False  # Prevent compilation during loading
)
#caption_model = tf.keras.models.load_model("/mnt/d/DIT/First Sem/Computer Vision/EchoLens-Pretrained/image_captioning_model.keras")


def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # Match input size to ResNet encoder
    img = np.array(img) / 255.0   # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dim
    return img
def generate_caption(model, image, tokenizer, max_length=20):
    input_seq = [tokenizer.word_index['<start>']]
    
    for _ in range(max_length):
        # Pad sequence
        seq_input = tf.keras.preprocessing.sequence.pad_sequences(
            [input_seq], maxlen=max_length, padding='post'
        )
        
        # Predict next word
        preds = model.predict([image, seq_input], verbose=0)
        predicted_id = np.argmax(preds[0, len(input_seq)-1])  # Assuming decoder returns a sequence
        
        word = tokenizer.index_word.get(predicted_id, '<unk>')
        input_seq.append(predicted_id)
        
        if word == '<end>':
            break
            
    caption = [tokenizer.index_word.get(i, '') for i in input_seq[1:-1]]
    return ' '.join(caption)

image_path = '/mnt/d/DIT/First Sem/Computer Vision/EchoLens-Pretrained/Teen-Boys-Playing-Basketball.jpg'
image = load_and_preprocess_image(image_path)
caption = generate_caption(caption_model, image, tokenizer)
print("Caption:", caption)
