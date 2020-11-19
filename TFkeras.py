import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle


# ============= load dataset ===============
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_sentences = train_df['tweet']
train_label = train_df['label']

# ============= Preprocess dataset ===============
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(train_sentences)
x = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)
y = tf.keras.utils.to_categorical(train_label)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)


# =========== Define and build model ==============
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
try:
    model = tf.keras.utils.multi_gpu_model(model, cpu_relocation=True)
    print("Training using multiple GPUs..")
except:
    print("Training using single GPU or CPU..")
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# ============== Train model ===============
epochs = 30
def warmup(epoch, lr):
    return max(lr +1e-6, 2e-5) # Used for increasing the learning rate 
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, min_delta=0.02, restore_best_weights=True),
    tf.keras.callbacks.LearningRateScheduler(warmup, verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=1e-6, patience=2, verbose=0, mode='auto', min_delta=0.001, cooldown=0, min_lr=1e-6)
]
hist = model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=16, validation_split=.15, callbacks=callbacks, verbose=1)
model.save('model\\bilstm', save_format='tf')

# ================ Test model =================
results = model.evaluate(X_test, y_test, verbose=0)
print('Loss: ', results[0])
print('Accuracy: ', results[1])

# ================== Submit ===================
test_sentences = test_df['tweet']
testing_sequences = tokenizer.texts_to_sequences(test_sentences)
x_test = pad_sequences(testing_sequences,maxlen=max_length)
yhat = model.predict(x_test)
yhat_test = np.argmax(yhat,axis=1)
test_df["Prediction"] = yhat_test
print(test_df.head())
test_df.to_csv("submission.csv", index=False)