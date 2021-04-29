import keras
import sys
import pandas as pd
from keras import layers
from keras import regularizers
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from os import path

args = sys.argv
print(args)
if len(args) > 1:
    if args[1] == "train":
        dfWiki = pd.read_json(args[2])

        train, test = train_test_split(dfWiki.drop(['word', 'sentence'], axis=1), test_size=0.1)
        train = train.to_numpy()
        test = test.to_numpy()
        encoding_dim = 32
        # This is our input image
        # This is our input image
        input = keras.Input(shape=(train.shape[1],))
        encoded = layers.Dense(500, activation='tanh')(input)
        encoded = layers.Dense(2000, activation='tanh')(encoded)
        encoded = layers.Dense(250, activation='tanh')(encoded)
        encoded = layers.Dense(encoding_dim, activation='tanh')(encoded)

        decoded = layers.Dense(250, activation='tanh')(encoded)
        decoded = layers.Dense(2000, activation='tanh')(decoded)
        decoded = layers.Dense(500, activation='tanh')(decoded)
        decoded = layers.Dense(train.shape[1], activation='tanh')(decoded)

        autoencoder = keras.Model(input, decoded)

        encoder = keras.Model(input, encoded)

        autoencoder.compile(optimizer='rmsprop', loss='mse')

        autoencoder.fit(train, train,
                        epochs=200,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(test, test))

        encoder.save('../models/compressor')
    elif args[1] == "convert":
        print("Read file", flush=True)
        dfWiki = pd.read_json(args[2])
        words = dfWiki['word']
        sentences = dfWiki['sentence']
        print("Drop words", flush=True)
        data = dfWiki.drop(['word', 'sentence'], axis=1)
        data = data.to_numpy()
        print("Loading compressor", flush=True)
        compressor = keras.models.load_model('../models/compressor')
        print("Compression...", flush=True)
        data_compressed = compressor.predict(data)
        dfWikiCompressed = pd.DataFrame(data_compressed)
        dfWikiCompressed['word'] = words
        dfWikiCompressed['sentence'] = sentences
        print("Save compressed file", flush=True)
        dfWikiCompressed.to_json(args[2].replace('.json', '-compressed.json'))
        print("Finish", flush=True)

else:
    compressor = keras.models.load_model('../models/compressor')