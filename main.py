#%% imports
import os
import re
import json
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping


URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
MODEL_PATH = os.path.join(os.getcwd(), 'model','natasha_model.h5')
OHE_PATH = os.path.join(os.getcwd(),'model','ohe.pkl')
TOKEN_PATH = os.path.join(os.getcwd(),'model','tokenizer.json')

#%% 1. Data Loading
df = pd.read_csv(URL)
text = df['text']
cat = df['category']
labels = df['category'].unique()

#%% 2. EDA
df.head()
cat.unique()
df.info()
print(text[100])

#%% 3. Data Cleaning

for index, data in enumerate(text):
    # clean url 
    temp = re.sub('www\.\w+\.\w+.\w+',' ',data)
    temp = re.sub('\w+\.\w+',' ', temp)
    # clean contractions
    temp = re.sub(r'\b[a-zA-Z]\b',' ',temp)
    # clean numbers and punctuations
    temp = re.sub('[^a-zA-Z]',' ',temp)
    # convert text to lowercase
    text[index] = temp.lower()

#%% 5. Data Preprocessing

# Data Tokenization
vocab_num = 5000
oov_token = '<OOV>'
tokenizer = Tokenizer(num_words=vocab_num, oov_token=oov_token)

# Training the tokenizer to learn words
tokenizer.fit_on_texts(text)

# Check the trained words
word_index = tokenizer.word_index
print(list(word_index.items())[0:10])

# Transform the sentences to int
text = tokenizer.texts_to_sequences(text)


#%% Find maxlen value
maxlen = []
for i in text:
    maxlen.append(len(i))

maxlen = int(np.ceil(np.median(maxlen)))

#%% Padding and Truncating
text = pad_sequences(text, maxlen=maxlen, padding='post',truncating='post')

# Target - Category
ohe = OneHotEncoder(sparse=False)
cat = ohe.fit_transform(np.expand_dims(cat, axis=-1))

#%% Train-Test split
X_train,X_test,y_train,y_test = train_test_split(text, cat, train_size=0.7, shuffle=True, random_state=123)

#%% 6. Model Development
input_shape = np.shape(X_train)[1:]
embedding_dim = 256
np_classes = len(np.unique(y_train, axis=0))

model = Sequential()
model.add(Embedding(vocab_num,embedding_dim))
model.add(LSTM(embedding_dim,input_shape=input_shape,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(embedding_dim))
model.add(Dropout(0.3))
model.add(Dense(np_classes,activation='softmax'))
model.summary()
plot_model(model, show_shapes=True)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')

# Callbacks
log_dir = os.path.join(os.getcwd(),'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = TensorBoard(log_dir=log_dir)

# early stopping callback
es_callback = EarlyStopping(monitor='loss',patience=30)
hist = model.fit(X_train,y_train, 
                    validation_data=(X_test, y_test), 
                    epochs=50, 
                    callbacks=[tb_callback, es_callback])

#%% 7. Model Analysis
y_pred = model.predict(X_test)
y_true = y_test

y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_true, axis=1)
labels = df['category'].unique()

# Classification Report
print(classification_report(y_true,y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(cmap=plt.cm.Purples, ax=ax)
plt.show()

#%% 8. Model Deployment
# Save Model
model.save(MODEL_PATH)

# Save One-Hot-Encoder
with open('ohe.pkl','wb') as f:
    pickle.dump(ohe,f)

# Save Tokenizer
token_json = tokenizer.to_json()
with open('tokenizer.json','w') as json_file:
    json.dump(token_json,json_file)