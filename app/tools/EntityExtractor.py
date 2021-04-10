import nltk
import os

from nltk.tag.stanford import StanfordPOSTagger
from nltk.stem import PorterStemmer
root_path="./app/lib/stanford-postagger-full-2018-10-16/"
pos_tagger = StanfordPOSTagger(root_path + "models/english-bidirectional-distsim.tagger", root_path + "stanford-postagger.jar", encoding='utf8')
#entity_tag = ['ADJ', 'ET', 'NC', 'NP', 'PRO', 'N']

entity_tag = ['NN', 'JJ', 'NNS', 'NNP', 'NNPS', 'JJR', 'JJS']
nltk.download('punkt')
porter = PorterStemmer()

def pos_tag(sentence):
    tokens = nltk.word_tokenize(sentence)
    tags = pos_tagger.tag(tokens)
    return tags

def get_good_words(sentence_tagged):
    i = list()
    index_word = 0
    print(sentence_tagged)
    for entity in sentence_tagged:
        if(entity[1] in entity_tag):
            i.append([porter.stem(entity[0]), index_word])
        index_word+=1
    return i

def get_entities(sentence):
    return(get_good_words(pos_tag(sentence)))
