import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.layers import Add

# Load captions
def load_captions(filename):
    captions = {}
    with open(filename, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) < 2:
                continue
            image_id, caption = tokens[0].split('#')[0], tokens[1]
            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append('startseq ' + caption + ' endseq')
    return captions

# Extract features from images using VGG16
def extract_features(directory):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = {}
    for name in tqdm(os.listdir(directory)):
        filename = os.path.join(directory, name)
        image = Image.open(filename).resize((224, 224))
        image = np.array(image)
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = preprocess_input(np.expand_dims(image, axis=0))
        feature = model.predict(image, verbose=0)
        features[name] = feature
    return features

# Create tokenizer
def create_tokenizer(descriptions):
    lines = [desc for caps in descriptions.values() for desc in caps]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# Prepare training sequences
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = [], [], []
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# Build captioning model
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder = Add()([fe2, se3])

    decoder = Dense(256, activation='relu')(decoder)
    outputs = Dense(vocab_size, activation='softmax')(decoder)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Generate caption from image
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([photo, seq], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text

# MAIN
if __name__ == "__main__":
    captions = load_captions("captions.txt")
    features = extract_features("images")
    tokenizer = create_tokenizer(captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(c.split()) for d in captions.values() for c in d)

    X1, X2, y = create_sequences(tokenizer, max_length, captions, features, vocab_size)
    model = define_model(vocab_size, max_length)
    model.fit([X1, X2], y, epochs=10, verbose=2)

    # Test on one image
    test_image = list(features.keys())[0]
    print("Caption for", test_image, ":", generate_caption(model, tokenizer, features[test_image], max_length))
