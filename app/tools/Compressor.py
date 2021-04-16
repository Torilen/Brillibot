import keras
import pandas as pd

def compressVectorDfdim1Todim2(df, compressor_model=None):
    return pd.DataFrame(compressor_model.predict(df.to_numpy()))