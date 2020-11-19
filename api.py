"""
GET http://<server_address>:5000/huggingface?text=<input_texts>
"""
import flask
from flask import Flask, request 
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from inference import huggingface_classify, keras_classify

# load huggingface 
distilbert = tf.keras.models.load_model('model\\transformer')
model_name = 'distilbert-base-uncased'
huggingface_tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# load keras
bilstm = tf.keras.models.load_model('model\\bilstm')
with open('tokenizer.pickle', 'rb') as handle:
    keras_tokenizer = pickle.load(handle)

app = Flask(__name__)

# ===== request model prediction =======
@app.route('/huggingface', methods=['GET', 'POST']) 
def huggingface():
    if request.method == 'GET':
        text = request.args['text'] 
    elif request.method == 'POST':
        text = request.form['text']
    # predict 
    yhat = huggingface_classify(text, huggingface_tokenizer, distilbert)
    result = "Input: {} -> Prediction: {}".format(text,yhat)
    print(result)
    return flask.jsonify(result)

@app.route('/keras', methods=['GET', 'POST']) 
def keras():
    if request.method == 'GET':
        text = request.args['text'] 
    elif request.method == 'POST':
        text = request.form['text']
    # predict 
    yhat = keras_classify(text, keras_tokenizer, bilstm)
    result = "Input: {} -> Prediction: {}".format(text,yhat)
    print(result)
    return flask.jsonify(result)


# ===== Start Flask server ======
if __name__ == "__main__": 
    app.run() # localhost 
    # app.run(host='0.0.0.0', port=5000, debug=True) # web