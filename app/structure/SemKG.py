import json
import numpy as np
import nltk
import joblib
import pandas as pd
import hdbscan
from tools.Embedder import concatEmbeddingFr, concatEmbeddingEn, getContextualEmbedding
import tools.Compressor
from nltk.corpus import stopwords
import keras
from scipy import spatial
import subprocess



class SemKG:
    pd.set_option('display.max_columns', 5)
    nltk.download('stopwords')
    graph = dict()
    graphNodeId = dict()
    graphNeighbour = dict()

    hdbscan_model = joblib.load('./app/models/hdbscan_trained.pkl')
    #dfWiki = pd.read_json('./app/tools/0-270-dfWiki-compressed-clustered.json')
    dfWiki = pd.DataFrame()
    print("DFWIKI", flush=True)
    print(dfWiki, flush=True)
    compressor = keras.models.load_model('./app/models/compressor')

    def get_occur_relation(self, s, o):
        return self.graph[(s, o)]

    def get_graph(self):
        return {"graph": self.graph, "nodesId": self.graphNodeId, "successor": self.graphSuccessor}

    def to_json(self):
        return json.dumps(self.get_graph())

    def load_from_json(self, json):
        data = json.loads(json)
        self.graph = data["graph"]
        self.graphNodeId = data["nodesId"]

    def save(self, path, name="semkg.json"):
        with open(path+"/"+name, 'w') as json_file:
            json.dump(self.to_json(), json_file)

    def add_node(self, node):
        if node not in list(self.graphNodeId.keys()):
            self.graphNodeId[node] = len(list(self.graphNodeId.keys()))+1

    def get_node_id(self, node):
        if node in list(self.graphNodeId.keys()):
            return self.graphNodeId[node]
        else:
            return -1

    def add_relation(self, s, o):
        self.add_node(s)
        self.add_node(o)

        if (s, o) not in list(self.graph.keys()):
            if (o, s) not in list(self.graph.keys()):
                self.graph[(s, o)] = 1
                self.graph[(o, s)] = 1
            else:
                self.graph[(o, s)] += 1
                self.graph[(s, o)] = self.graph[(o, s)]
        else:
            self.graph[(s, o)] += 1
            self.graph[(o, s)] = self.graph[(s, o)]

        if s in list(self.graphNeighbour.keys()):
            if o not in self.graphNeighbour[s]:
                self.graphNeighbour[s].append(o)
        else:
            self.graphNeighbour[s] = list()
            self.graphNeighbour[s].append(o)

        if o in list(self.graphNeighbour.keys()):
            if s not in self.graphNeighbour[o]:
                self.graphNeighbour[o].append(s)
        else:
            self.graphNeighbour[o] = list()
            self.graphNeighbour[o].append(s)


    def add_relations(self, rels, epikg, input):
        for rel in rels:
            s = rel[0]
            o = rel[1]
            print("Tuples")
            print(rel)
            self.add_relation(s[0], o[0])
            epikg.add_relation(self.graphNodeId[s[0]], self.graphNodeId[o[0]], input, s[1], o[1])

    def get_all_nodes_in_neighbour(self, entity):
        #print(list(self.graphNeighbour.keys()))
        if(entity in list(self.graphNeighbour.keys())):
            neighbour = self.graphNeighbour[entity]
            weights = [self.graph[(entity, n)] for n in neighbour]
            res = []

            for j in range(min(3, len(weights))):
                index_max = np.argmax(weights)
                res.append([neighbour[index_max], weights[index_max]])
                weights[index_max] = 0
        else:
            res = []

        return res

    def semantic_propagation(self, entity, steps, i):
        childs = self.get_all_nodes_in_neighbour(entity)
        l = list()
        l.append(entity)
        for child in childs:
            # child[0] => entity | #child[1] => weight
            l.append(child[0])
        for child in childs:
            if(i < steps):
                res_t = self.semantic_propagation(child[0], steps, i+1)
                #print(res_t)
                l = [*l, *self.semantic_propagation(child[0], steps, i+1)]
            else:
                return l
        return l

    def get_nearest_member_of_cluster(self, source, cluster_target, top_n):
        print(cluster_target, flush=True)
        print(cluster_target.values, flush=True)
        print(cluster_target.values.T, flush=True)
        word = cluster_target['word']
        clusterid = cluster_target['clusterid']
        sentence = cluster_target['sentence']
        keywordsId = cluster_target['keywordsId']
        answers = cluster_target['answer']
        clusterDf = cluster_target.drop(['word', 'clusterid', 'sentence', 'keywordsId', 'answer'], axis=1)
        clusterDf['distance'] = clusterDf.apply(lambda x: 1 - spatial.distance.cosine(list(x), list(source)), axis=1)
        clusterDf['word'] = word
        clusterDf['clusterid'] = clusterid
        clusterDf['sentence'] = sentence
        clusterDf['keywordsId'] = keywordsId
        clusterDf['answer'] = answers
        clusterDfsorted = clusterDf.sort_values(by=['distance'], inplace=False, ascending=False)
        result = clusterDfsorted.head(top_n)
        print("NEAREST NEIGHBOUR", flush=True)
        print(result[['word', 'sentence','distance', 'clusterid', 'keywordsId', 'answer']], flush=True)
        return [list(result.index), list(result.distance)]

    def learn(self, personas, keywordsId, answers):
        i = 0
        for persona in personas:
            embed = concatEmbeddingEn(getContextualEmbedding(persona, verbose=True))
            df2 = pd.DataFrame(embed[0])
            df2 = tools.Compressor.compressVectorDfdim1Todim2(df2, self.compressor)
            df2 = df2.rename(columns=str)
            df2['word'] = [s.replace("</w>", "") for s in embed[1]]
            sentences = []
            doc = ' '.join(embed[1])
            h = 0
            windows_size = 20
            print(doc, flush=True)
            print(embed[1], flush=True)
            for word in embed[1]:
                sentences.append(' '.join(embed[1][max(0, h - windows_size):min(len(embed[1]), h + windows_size)]))
                h += 1
            print("SENTENCES", flush=True)
            print(sentences, flush=True)
            df2['sentence'] = sentences
            df2 = df2[~df2.word.isin(stopwords.words('english'))]
            data_formatted = []
            for col in df2.columns:
                if col != "word" and col != "sentence":
                    data_formatted.append(df2[col].tolist())
            data = np.array(data_formatted[0:32]).T
            print("TO LEARN", flush=True)
            print(data, flush=True)
            print(data.shape, flush=True)
            #self.hdbscan_model.fit(data)
            labels, _ = hdbscan.approximate_predict(self.hdbscan_model, data)
            df2['clusterid'] = labels
            df2['keywordsId'] = keywordsId[i]
            df2['answer'] = answers[i]
            print(df2.head(), flush=True)
            print(df2.columns, flush=True)
            print(self.dfWiki.columns, flush=True)
            print(self.dfWiki, flush=True)
            self.dfWiki = pd.concat([self.dfWiki, df2]).reset_index(drop=True)
            print(self.dfWiki)
            print("TAIL", flush=True)
            print(self.dfWiki.tail(10), flush=True)
            i += 1

    def get_stories(self, epikg, entities_word, entities_vector, top_n=5, steps=5):
        if len(entities_word) > 0:
            dfVector = tools.Compressor.compressVectorDfdim1Todim2(pd.DataFrame(entities_vector), self.compressor)
            data_formatted = []
            for col in dfVector.columns:
                if col != "word" and col != "sentence" and col != "keywordsId" and col != "answer":
                    data_formatted.append(dfVector[col].tolist())
            data = np.array(data_formatted[0:32]).T
            print("TO SPEAK", flush=True)
            print(data, flush=True)
            print(data.shape, flush=True)

            stories = []
            similarities_score = []

            if data.shape[0] > 0:
                labels, _ = hdbscan.approximate_predict(self.hdbscan_model, data)
                dfVector['word'] = entities_word
                dfVector['clusterid'] = labels
                dfVector['keywordsId'] = [list() for i in range(len(entities_word))]
                dfVector['answer'] = ['' for i in range(len(entities_word))]
                result = pd.DataFrame()
                for index, row in dfVector.iterrows():
                    v = row.values.T
                    print("GET STORIES", flush=True)
                    print(v, flush=True)
                    cluster = self.dfWiki[self.dfWiki.clusterid == row.clusterid]
                    result_index, similarities_score = self.get_nearest_member_of_cluster(v[:-2], cluster, top_n)
                    df2 = cluster.loc[result_index]
                    df2['distance'] = similarities_score
                    result = pd.concat([result, df2]).reset_index(drop=True).sort_values(by=['distance'], inplace=False, ascending=False)
                    result.drop_duplicates(subset="sentence", keep='first', inplace=True)

            return result
        else:
            return pd.DataFrame()