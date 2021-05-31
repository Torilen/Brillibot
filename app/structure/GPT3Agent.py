import os
import openai

from flask import jsonify
from tools.Translator import translate_base, detect, translate_by_url, translate_by_api
from tools.Utils import process_output_chatbot
from tools.EntityExtractor import get_entities, get_entities_index
from tools.Embedder import concatEmbeddingFr, concatEmbeddingEn, getContextualEmbedding
from tools.Converter import Entities2Tuples
from structure.SemKG import SemKG
from structure.EpiKG import EpiKG
from parlai.core.agents import create_agent

class GPT3Agent:
    api_key = ''
    semkg = SemKG()
    epikg = EpiKG()
    polyencoder = None
    ip = ""
    history = []
    persona_history = []

    def __init__(self, personality, keywordsId, answers, ip, keywordsCond, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key
        self.addStoriesLive(personality)
        self.learn(personality, keywordsId, answers, keywordsCond)
        self.ip = ip
        args, self.polyencoderagent = self.initPolyEncoder(ip, personality)

    def addStoriesLive(self, personality):
        self.history += ["\nAI: "+p for p in personality]
        self.persona_history += ["\nAI: "+p for p in personality]

    def learn(self, sentences, keywordsId, answers, keywordsCond):
        self.semkg.learn(sentences, keywordsId, answers, keywordsCond)

    def initPolyEncoder(self, ip, personality):
        f = open('candidates{}.txt'.format(ip), "w")
        f.write(' \n'.join(personality))
        f.close()
        args = {'optimizer': 'adamax', 'learningrate': 5e-05, 'batchsize': 256, 'embedding_size': 768,
                'num_epochs': 8.0, 'model': 'transformer/polyencoder', 'n_layers': 12, 'n_heads': 12, 'ffn_size': 3072,
                'gradient_clip': 0.1}
        args['eval_candidates'] = 'fixed'
        args['encode_candidate_vecs'] = 'true'
        args['fixed_candidates_path'] = 'candidates.txt'
        args['model_file'] = 'zoo:pretrained_transformers/model_poly/model'
        args['candidates'] = 'batch'
        args['override'] = {'model': 'transformer/polyencoder',
                            'model_file': '/data1/home/mrim/bentebia/anaconda3/envs/grafbot/lib/python3.7/site-packages/data/models/pretrained_transformers/model_poly/model',
                            'encode_candidate_vecs': True, 'eval_candidates': 'fixed',
                            'fixed_candidates_path': 'candidates{}.txt'.format(ip)}

        return args, create_agent(args)

    def speak(self, reply_text, keywordsUnlocked):
        user_language = detect(reply_text)
        # user_language = "en"
        english_version_of_user_input = translate_base(reply_text, src=user_language)
        # english_version_of_user_input = reply_text
        embedded = concatEmbeddingEn(getContextualEmbedding(english_version_of_user_input, verbose=False))
        entities = get_entities(english_version_of_user_input)
        stories = self.semkg.get_stories(self.epikg, [x[0] for x in entities], [embedded[0][x[1]] for x in entities],
                                         keywordsUnlocked)
        print("STORIES: ")
        print(stories)
        if len(stories) > 0:
            if not stories.iloc[0].answer == '':
                self.history.append("\nHuman: "+english_version_of_user_input)
                json_return = dict()

                if (user_language != "en"):
                    json_return['text'] = process_output_chatbot(stories.iloc[0].answer)
                    json_return['text'] = translate_base(stories.iloc[0].answer, dest=user_language)
                else:
                    json_return['text'] = process_output_chatbot(stories.iloc[0].answer)

                json_return['user_lang'] = user_language
                json_return['stories'] = [stories.iloc[0].sentence]
                json_return['score'] = [stories.iloc[0].distance]
                json_return['keywordsId'] = [stories.iloc[0].keywordsId]
                return jsonify(json_return)
            else:
                if len(stories) > 1:
                    m = min(3, len(stories))
                    good_stories = []
                    print("CREATE CANDIDATES", flush=True)
                    for p in range(m):
                        os.remove('candidates{}.txt'.format(self.ip))
                        args, self.polyencoderagent = self.initPolyEncoder(self.ip,
                                                                           [e for e in list(stories.sentence.values) if
                                                                            not e in good_stories])
                        print("OBSERVE", flush=True)
                        self.polyencoderagent.observe({'episode_done': False,
                                                       'text': ' \n'.join(
                                                           self.persona_history) + '\n' + english_version_of_user_input})
                        print("ACT", flush=True)
                        res = self.polyencoderagent.act()
                        print("PRINT ACT", flush=True)
                        print(res, flush=True)
                        good_stories.append(res['text'])
                    self.addStoriesLive(good_stories)
                else:
                    print("I don't remember anything", flush=True)
                self.history.append("\nHuman: "+english_version_of_user_input)
                response = openai.Completion.create(
                    engine="davinci",
                    prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly."
                           "\n"+''.join(self.history)+"\nAI:",
                    temperature=0.9,
                    max_tokens=150,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.6,
                    stop=["\n", " Human:", " AI:"]
                )

                print(response)

                json_return = dict()

                if (user_language != "en"):
                    json_return['text'] = process_output_chatbot(response)
                    json_return['text'] = translate_base(json_return['text'], dest=user_language)
                else:
                    json_return['text'] = process_output_chatbot(response)

                json_return['user_lang'] = user_language
                json_return['stories'] = good_stories
                json_return['score'] = list(stories[stories.sentence.isin(good_stories)].distance.values)
                json_return['keywordsId'] = list()
                return jsonify(json_return)
        else:
            self.history.append("\nHuman: " + english_version_of_user_input)
            response = openai.Completion.create(
                engine="davinci",
                prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly."
                       "\n" + ''.join(self.history) + "\nAI:",
                temperature=0.9,
                max_tokens=150,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.6,
                stop=["\n", " Human:", " AI:"]
            )

            json_return = dict()

            if (user_language != "en"):
                json_return['text'] = process_output_chatbot(response)
                json_return['text'] = translate_base(json_return['text'], dest=user_language)
            else:
                json_return['text'] = process_output_chatbot(response)

            json_return['user_lang'] = user_language
            json_return['stories'] = list()
            json_return['score'] = list()
            json_return['keywordsId'] = list()
            return jsonify(json_return)

    def reset(self):
        self.history = []

    def get(self, val):
        if val == 'agent':
            return self
        else:
            return None