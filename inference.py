import numpy as np
import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertTokenizer
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# load huggingface 
distilbert = tf.keras.models.load_model('model\\transformer')
model_name = 'distilbert-base-uncased'
huggingface_tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# load keras
bilstm = tf.keras.models.load_model('model\\bilstm')
with open('tokenizer.pickle', 'rb') as handle:
    keras_tokenizer = pickle.load(handle)

def huggingface_classify(input_text, tokenizer, model, max_len=120):
    clean = re.sub(r"[-()\"#/@;:<>{}=~|.?,]", "", str(input_text))
    if 'user' in clean : clean.strip('user')
    tokens = [tokenizer.encode_plus(t, max_length=max_len, pad_to_max_length=True, add_special_tokens=True) for t in [clean]]
    tensor = np.array([a['input_ids'] for a in tokens])
    results = model.predict(tensor)   
    results = np.argmax(results,axis=1)
    labels = ['neutral', 'racist/sexist']
    return labels[results[0]]


def keras_classify(input_text, tokenizer, model, max_len=120):
    clean = re.sub(r"[-()\"#/@;:<>{}=~|.?,]", "", str(input_text))
    if 'user' in clean : clean.strip('user')
    testing_sequences = tokenizer.texts_to_sequences(clean)
    tensor = pad_sequences(testing_sequences,maxlen=max_len)
    results = model.predict(tensor)   
    results = np.argmax(results,axis=1)
    labels = ['neutral', 'racist/sexist']
    return labels[results[0]]


