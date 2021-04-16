import keras
import pandas as pd

def compressVectorDfdim1Todim2(df, compressor=None):
    return pd.DataFrame(compressor.predict(df.to_numpy()))