import keras
import sys
import pandas as pd
from keras import layers
from keras import regularizers
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

args = sys.argv
print(args)
if len(args) > 0:
    if args[1] == "train":
        dfWiki = pd.read_json(args[2])

        train, test = train_test_split(dfWiki.drop(['word'], axis=1), test_size=0.1)
        train = train.to_numpy()
        test = test.to_numpy()
        transformer = MinMaxScaler(feature_range=(32, 768)).fit(train)
        train = transformer.transform(train)
        test = transformer.transform(test)
        encoding_dim = 32
        # This is our input image
        input = keras.Input(shape=(train.shape[1],))
        encoded = layers.Dense(500, activation='relu')(input)
        encoded = layers.Dense(2000, activation='relu')(encoded)
        encoded = layers.Dense(250, activation='relu')(encoded)
        encoded = layers.Dense(1000, activation='relu')(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

        decoded = layers.Dense(1000, activation='relu')(encoded)
        decoded = layers.Dense(250, activation='relu')(decoded)
        decoded = layers.Dense(2000, activation='relu')(decoded)
        decoded = layers.Dense(500, activation='relu')(decoded)
        decoded = layers.Dense(train.shape[1], activation='sigmoid')(decoded)

        autoencoder = keras.Model(input, decoded)

        encoder = keras.Model(input, encoded)

        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        autoencoder.fit(train, train,
                        epochs=60,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(test, test))

        encoder.save('../models/compressor')
        joblib.dump(transformer, '../models/normalizer.pkl')
    elif args[1] == "convert":
        print("Read file", flush=True)
        dfWiki = pd.read_json(args[2])
        words = dfWiki['word']
        print("Drop words", flush=True)
        data = dfWiki.drop(['word'], axis=1)
        data = data.to_numpy()
        print("Loading compressor", flush=True)
        compressor = keras.models.load_model('../models/compressor')
        print("Loading normalizer", flush=True)
        normalizer = joblib.load('../models/normalizer.pkl')
        data_normalized = normalizer.transform(data)
        print("Compression...", flush=True)
        data_compressed = compressor.predict(data_normalized)
        dfWikiCompressed = pd.DataFrame(normalizer.inverse_transform(data_compressed))
        print("Save compressed file", flush=True)
        dfWikiCompressed.to_json(args[2].replace('.json', '-compressed.json'))
        print("Finish", flush=True)

else:
    compressor = keras.models.load_model('../models/compressor')
    normalizer = joblib.load('../models/normalizer.pkl')

def compressVector(v):
    vector_normalized = normalizer.transform([v])
    vector_compressed = compressor.predict(vector_normalized)

    return normalizer.inverse_transform([vector_compressed[0]])