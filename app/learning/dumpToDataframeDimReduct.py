import pandas as pd
from ..tools.Embedder import getContextualEmbedding, concatEmbeddingEn
from transformers import BertTokenizer, FlaubertTokenizer, BertModel, FlaubertModel

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
for sentence in content.split('\n'):
  for sent in range(len(sentence.split())-500):
    embed = getContextualEmbedding(tokenizer, model, sentence[sent*500:(sent+1)*500])
    embed = concatEmbeddingEn(embed)
    df2 = pd.DataFrame(embed[0])
    df2['word'] = [s.replace("</w>", "") for s in embed[1]]
    dfWiki = pd.concat([dfWiki, df2])
  i+=1
  if (i % 10 == 0):
    ldfwiki = len(dfWiki)
    print('Processed ' + str(i) + ' articles -- dfWiki size = ' + str(ldfwiki) + ' -- '+str(ldfwiki/number*100)+'% of words processed')