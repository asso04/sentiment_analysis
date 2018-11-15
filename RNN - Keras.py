import csv
import pandas as pd
import numpy as np
# fix random seed for reproducibility
np.random.seed(42)


from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import keras

print('RNN_v07 adjusted dropout and validation')

#load file
df = pd.read_csv('Trainded Dataset - EnglishFiltered - ngrams.csv', error_bad_lines=False)

X = df['Normalized Text NGRAMS']
Y = df['Sentiment']
tokenizer = Tokenizer(split=",")

tokenizer.fit_on_texts(X)
#Split a sentence into a list of words.
X_tokenized = tokenizer.texts_to_sequences(X.values)


Y = pd.get_dummies(df['Sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X_tokenized,Y, test_size = 0.2, random_state = 42)
# print('Train Shape', X_train.shape,Y_train.shape)
# print('Text Shape', X_test.shape,Y_test.shape)

from keras.preprocessing import sequence

# Set the maximum number of words per document (for both training and testing)
max_words = 100

#Pad sequences 
X_train = sequence.pad_sequences(X_train, maxlen=max_words)

X_test = sequence.pad_sequences(X_test, maxlen=max_words)

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

vocabulary_size = len(tokenizer.word_counts)


#Model design
model =  Sequential()
model.add(Embedding(vocabulary_size + 1, 128))
model.add(LSTM(256, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(2, activation='sigmoid'))

#Optimizer
optimizer_adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer=optimizer_adam,
              metrics=['accuracy'])

print(model.summary())

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
#batch size and number of epochs
batch_size = 100
num_epochs = 8

checkpointer = ModelCheckpoint(filepath='rnn_model_best.h5', monitor='val_loss',
                               save_best_only=True, verbose=1)
earlystopper = EarlyStopping(monitor='val_loss', patience=10)

#save log
csv_logger = CSVLogger('log_ngrams_v09.csv', append=True, separator=';')

# TODO: Train your model
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=num_epochs,
          callbacks=[checkpointer, earlystopper, csv_logger],
          validation_split=0.3,
          validation_data=(X_test, Y_test),
          verbose=2)


score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

model.save('rnn_model_google_v09.h5')