import numpy as np
import pandas as pd
import re
from transformers import BertModel
from transformers import FlaubertModel
from transformers import BertTokenizer
from transformers import FlaubertTokenizer
import torch

LANG = "EN"
if LANG == "FR":
    tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased', do_lowercase=False)
    model, log = FlaubertModel.from_pretrained('flaubert/flaubert_base_cased', output_loading_info=True)
else:
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lowercase=False)
    model = BertModel.from_pretrained('bert-base-cased')


def processString(s):
    s = re.sub(r'[^\w\s]', ' ', s)
    s = s.lower()
    print('SENTENCE PROCESSED : {}'.format(s))
    return s

def getContextualEmbedding(sentence, verbose=False, tokenizer=tokenizer, model=model):
  sentence = processString(sentence)
  tokens = tokenizer.tokenize(sentence)
  tokens = ['[CLS]'] + tokens + ['[SEP]']
  T = 30
  padded_tokens = tokens +['[PAD]' for _ in range(T-len(tokens))]
  if verbose:
    print("Padded tokens are \n {} ".format(padded_tokens))
  attn_mask_l = [ 1 if token != '[PAD]' else 0 for token in padded_tokens  ]
  if verbose:
    print("Attention Mask are \n {} ".format(attn_mask_l))
  seg_ids = [0 for _ in range(len(padded_tokens))]
  if verbose:
    print("Segment Tokens are \n {}".format(seg_ids))
  sent_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
  if verbose:
    print("senetence idexes \n {} ".format(sent_ids))
  token_ids = torch.tensor(sent_ids).unsqueeze(0)
  attn_mask = torch.tensor(attn_mask_l).unsqueeze(0)
  seg_ids   = torch.tensor(seg_ids).unsqueeze(0)
  output = model(token_ids, attention_mask = attn_mask,token_type_ids = seg_ids)
  return [output.last_hidden_state.cpu().detach().numpy()[0][1:sum(attn_mask_l)-1], tokens[1:len(tokens)-1]]

def concatEmbeddingFr(embeddings):
  i = 0
  result_V = []
  result = []
  temp_V = []
  temp = []
  for elem in embeddings[1]:
    if not "</w>" in elem:
      temp_V.append(embeddings[0][i])
      temp.append(elem)
    else:
      if len(temp) > 0:
        temp_V.append(embeddings[0][i])
        temp.append(elem)
        result_V.append(np.mean(np.array(temp_V), axis=0).tolist())
        result.append(''.join(temp))
        temp_V = []
        temp = []
      else:
        result_V.append(embeddings[0][i])
        result.append(elem)
    i+=1
  result_V = np.array(result_V)
  return [result_V, result]

def concatEmbeddingEn(embeddings):
  print("CONCAT", flush=True)
  result_V = []
  result = []
  temp_V = []
  temp = []
  print(embeddings[1], flush=True)
  for i in range(len(embeddings[1])-1):
    if not "##" in embeddings[1][i+1]:
      if len(temp) == 0:
        result_V.append(embeddings[0][i])
        result.append(embeddings[1][i])
      else:
        temp_V.append(embeddings[0][i])
        temp.append(embeddings[1][i].replace('##', ''))
        result_V.append(np.mean(np.array(temp_V), axis=0).tolist())
        result.append(''.join(temp))
        temp_V = []
        temp = []
    else:
      temp_V.append(embeddings[0][i])
      temp.append(embeddings[1][i].replace('##', ''))

  if len(temp) > 0:
    temp.append(embeddings[1][len(embeddings[1])-1].replace('##', ''))
    temp_V.append(embeddings[0][len(embeddings[1])-1])
    result_V.append(np.mean(np.array(temp_V), axis=0).tolist())
    result.append(''.join(temp))
  else:
    result_V.append(embeddings[0][len(embeddings[1])])
    result.append(embeddings[1][len(embeddings[1])])
  result_V = np.array(result_V)
  print(result, flush=True)
  print(result_V.shape, flush=True)
  return [result_V, result]