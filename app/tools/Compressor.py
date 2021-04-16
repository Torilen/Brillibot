import keras
import pandas as pd

compressor = keras.models.load_model('../models/compressor')

def compressVectorDfdim1Todim2(df):
    return pd.DataFrame(compressor.predict(df.to_numpy()))