import pandas as pd
from Embedder import getContextualEmbedding, concatEmbeddingEn
from transformers import BertModel
from transformers import FlaubertModel
from transformers import BertTokenizer
from transformers import FlaubertTokenizer


LANG = "EN"
if LANG == "FR":
  tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased', do_lowercase=False)
  model, log = FlaubertModel.from_pretrained('flaubert/flaubert_base_cased', output_loading_info=True)
else:
  tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lowercase=False)
  model = BertModel.from_pretrained('bert-base-cased')


my_file = open("wiki.dump", "r")
content = my_file.read()
my_file.close()
dfWiki = pd.DataFrame()
number = len(content.split())
i = 0

print("Start !")
p = content.split('\n')
print("{} articles to processed".format(len(p)))
for sentence in p:
  for sent in range(len(sentence.split())-500):
    embed = getContextualEmbedding(tokenizer, model, sentence[sent*500:(sent+1)*500])
    embed = concatEmbeddingEn(embed)
    df2 = pd.DataFrame(embed[0])
    df2['word'] = [s.replace("</w>", "") for s in embed[1]]
    dfWiki = pd.concat([dfWiki, df2])
  i+=1
  if (i % 1 == 0):
    ldfwiki = len(dfWiki)
    print('Processed ' + str(i) + ' articles -- dfWiki size = ' + str(ldfwiki) + ' -- '+str(ldfwiki/number*100)+'% of words processed')
  if (i % 10 == 0):
    dfWiki.to_json('{}-{}-dfWiki.json'.format(0, i))

import keras
from keras import layers
from keras import regularizers
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
train, test = train_test_split(dfWiki.drop(['word'], axis=1), test_size=0.2)
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
