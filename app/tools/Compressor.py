import keras
import sys
import pandas as pd
from keras import layers
from keras import regularizers
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

args = sys.argv

if len(args) > 0:
    if args[0] == "train":
        dfWiki = pd.read_json(args[1])

        train, test = train_test_split(dfWiki.drop(['word'], axis=1), test_size=0.1)
        train = train.to_numpy()
        test = test.to_numpy()
        transformer = MinMaxScaler().fit(train)
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
                        epochs=200,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(test, test))

        encoder.save('../models/compressor')
        joblib.dump(transformer, '../models/normalizer.pkl')


else:
    compressor = keras.models.load_model('../models/compressor')
    normalizer = joblib.load('../models/normalizer.pkl')

def compressVector(v):
    vector_normalized = normalizer.transform([v])
    vector_compressed = compressor.predict(vector_normalized)

    return normalizer.inverse_transform([vector_compressed[0]])