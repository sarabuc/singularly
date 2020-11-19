import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import re
from transformers import TFDistilBertModel, DistilBertTokenizer

model_name = 'distilbert-base-uncased'
pretrained_model = TFDistilBertModel.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

train_path = 'train.csv'
test_path = 'test.csv'
input_column = 'tweet'
label_column = 'label'

# ============= load dataset ===============
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# --- get classes and convert to categorical ---
label_cols = train_df[label_column].unique()

if type(label_cols) == np.ndarray:
    train_df["target"] = train_df[label_column]
    # train_df.pop(label_column)
    num_classes = label_cols.shape[0]
else :
    num_classes = len(label_column)
    train_df[label_column] = train_df[label_column].astype('category')
    train_df["target"] = train_df[label_column].cat.codes
    if label_column in test_df:
        test_df[label_column] = test_df[label_column].astype('category')
        test_df["target"] = test_df[label_column].cat.codes

print(train_df.head())
print(test_df.head())

# ============= Preprocess dataset ===============
def clean_text(text):
    """ Clean text """
    clean = re.sub(r"[-()\"#/@;:<>{}=~|.?,]", "", str(text))
    if 'user' in clean : clean.strip('user')
    return clean

def get_inputs(input_text, tokenizer, max_len=120):
    """ Gets tensors from text using the tokenizer"""
    tokens = [tokenizer.encode_plus(t, max_length=max_len, pad_to_max_length=True, add_special_tokens=True) for t in input_text]
    tensor = np.array([a['input_ids'] for a in tokens])
    return tensor

train_df[input_column] = train_df[input_column].map(clean_text)
test_df[input_column] = test_df[input_column].map(clean_text)
print(train_df.head())
print(test_df.head())
train_texts = train_df[input_column]
train_labels = tf.keras.utils.to_categorical(train_df['target'].values,num_classes)

if 'target' in test_df:
    test_texts = test_df[input_column]
    test_labels = tf.keras.utils.to_categorical(test_df['target'].values,num_classes)
    X_train, X_test, y_train, y_test = train_texts, train_labels, test_texts, test_labels
else:
    X_train, X_test, y_train, y_test = train_test_split(train_texts, train_labels, test_size=0.2, random_state=50)

X_train = get_inputs(X_train, tokenizer)
X_test = get_inputs(X_test, tokenizer)
print('Done data processing')


# =========== Define and build model ==============
# Define token ids as inputs
inputs = tf.keras.Input(shape=(120,), name='inputs', dtype='int32')
# Call model
transformer = pretrained_model
transformer_outputs = transformer(inputs)[0]
# add classification layers
transformer_outputs = tf.squeeze(transformer_outputs[:, -1:, :], axis=1) # Collect last step from last hidden state (CLS)
transformer_outputs = tf.keras.layers.Dropout(.3)(transformer_outputs) # Apply dropout for regularization
outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='outputs')(transformer_outputs) # Final output 

# Compile model
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
try:
    model = tf.keras.utils.multi_gpu_model(model, cpu_relocation=True)
    print("Training using multiple GPUs..")
except:
    print("Training using single GPU or CPU..")
model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# ============ Train model ============
epochs = 30
checkpoint_filepath = 'model\\checkpoint' # latest = tf.train.latest_checkpoint(checkpoint_filepath); model.load_weights(latest) ==> Load the previously saved weights
def warmup(epoch, lr):
    return max(lr +1e-6, 2e-5) # Used for increasing the learning rate 
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, min_delta=0.02, restore_best_weights=True),
    tf.keras.callbacks.LearningRateScheduler(warmup, verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=1e-6, patience=2, verbose=0, mode='auto', min_delta=0.001, cooldown=0, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,save_weights_only=False, verbose=0, monitor='val_accuracy',mode='max',save_best_only=True)
]
hist = model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=16, validation_split=.15, callbacks=callbacks, verbose=1)
model.save('model\\transformer', save_format='tf')
print('Done training')

# ============ Load model ============
model = tf.keras.models.load_model('model\\transformer')

# ============ Test model ============
results = model.evaluate(X_test, y_test, verbose=0)
print('Loss: ', results[0])
print('Accuracy: ', results[1])

# ================== Prediction ===================
test_texts = test_df[input_column]
x_test = get_inputs(test_texts, tokenizer)
yhat = model.predict(x_test)
yhat_test = np.argmax(yhat,axis=1)
test_df["Prediction"] = yhat_test
print(test_df.head())
test_df.to_csv("submission.csv", index=False)

