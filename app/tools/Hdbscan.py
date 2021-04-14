import joblib
import hdbscan
import nltk
import sys
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

nltk.download('stopwords')

model = hdbscan.HDBSCAN(min_cluster_size=2)

args = sys.argv
print(args)
if len(args) > 1:
    if args[1] == "init_train":
        dfWiki = pd.read_json(args[2])
        dfWiki = dfWiki[~dfWiki.word.isin(stopwords.words('english'))]
        data_formatted = []
        for col in dfWiki.columns:
            if col != "word" and col != "sentence":
                data_formatted.append(dfWiki[col].tolist())
        data = np.array(data_formatted[0:768]).T
        labels = model.fit_predict(data)
        dfWiki['clusterid'] = labels
        dfWiki.to_json(args[2].replace('.json', '-clustered.json'))
        joblib.dump(model, '../models/hdbscan_trained.pkl')
else:
    model = joblib.load('../models/hdbscan_trained.pkl')


