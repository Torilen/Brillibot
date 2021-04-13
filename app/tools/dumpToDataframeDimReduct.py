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

print("Start !", flush=True)
p = content.split('\n')
print("{} articles to processed".format(len(p)), flush=True)
for sentence in p:
  for sent in range(len(sentence.split())-500):
    embed = getContextualEmbedding(tokenizer, model, sentence[sent*500:(sent+1)*500])
    embed = concatEmbeddingEn(embed)
    df2 = pd.DataFrame(embed[0])
    df2['word'] = [s.replace("</w>", "") for s in embed[1]]
    sentences = []
    doc = embed[1]
    h = 0
    windows_size = 10
    for word in embed[1]:
      sentences.append(' '.join(doc[min(0, h-windows_size):min(len(embed[1]), h+windows_size)]))
      h+=1
    df2['sentence'] = sentences
    dfWiki = pd.concat([dfWiki, df2])
  i+=1
  if (i % 1 == 0):
    ldfwiki = len(dfWiki)
    print('Processed ' + str(i) + ' articles -- dfWiki size = ' + str(ldfwiki) + ' -- '+str(ldfwiki/number*100)+'% of words processed', flush=True)
  if (i % 10 == 0):
    dfWiki.reset_index(drop=True, inplace=True)
    dfWiki.to_json('{}-{}-dfWiki.json'.format(0, i))

