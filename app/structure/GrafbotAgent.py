from flask import jsonify
from parlai.scripts.interactive import setup_args
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from tools.Translator import translate_base, detect, translate_by_url, translate_by_api
from tools.Utils import process_output_chatbot
from tools.EntityExtractor import get_entities, get_entities_index
from tools.Embedder import concatEmbeddingFr, concatEmbeddingEn, getContextualEmbedding
from tools.Converter import Entities2Tuples
from structure.SemKG import SemKG
from structure.EpiKG import EpiKG
import os

class GrafbotAgent:
    parser = setup_args()
    opt = None
    agent = None
    world = None
    semkg = SemKG()
    epikg = EpiKG()
    polyencoder = None
    ip = ""
    history = []
    persona_history = []

    def __init__(self, personality, ip):
        self.opt = self.parser.parse_args(print_args=False)
        self.opt['task'] = 'parlai.agents.local_human.local_human:LocalHumanAgent'
        self.agent = create_agent(self.opt, requireModelExists=True)
        self.addStoriesLive(personality)
        self.learn(personality)
        #self.addStoriesLive(personality)
        self.world = create_task(self.opt, self.agent)
        self.ip = ip
        args, self.polyencoderagent = self.initPolyEncoder(ip, personality)

    def addStoriesLive(self, personality):
        self.history += personality
        self.persona_history += personality
        personalityText = ' \n'.join(["your persona: " + personaField for personaField in personality])
        print(personalityText)
        if(len(personality) > 0):
            self.agent.observe({'episode_done': False, 'text': personalityText})

    def learn(self, sentences):
        self.semkg.learn(sentences)

    def initPolyEncoder(self, ip, personality):
        f = open('candidates{}.txt'.format(ip), "w")
        f.write(' \n'.join(["your persona: " + personaField for personaField in personality]))
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

    def speak(self, reply_text):
        print("Reply : "+reply_text)
        user_language = detect(reply_text)
        #user_language = "en"
        english_version_of_user_input = translate_base(reply_text, src=user_language)
        #english_version_of_user_input = reply_text
        embedded = concatEmbeddingEn(getContextualEmbedding(english_version_of_user_input, verbose=False))
        entities = get_entities(english_version_of_user_input)
        stories = self.semkg.get_stories(self.epikg, [x[0] for x in entities], [embedded[0][x[1]] for x in entities])
        print("STORIES: ")
        print(stories)
        if len(stories) > 1:
            m = min(3, len(stories))
            good_stories = []
            for p in range(m):
                os.remove('candidates{}.txt'.format(self.ip))
                f = open('candidates{}.txt'.format(self.ip), "a")
                for story in [e for e in stories if not e in good_stories]:
                    f.write(story+"\n")
                f.close()
                self.polyencoderagent.observe({'episode_done': False,
                               'text': ' \n'.join(["your persona: " + personaField for personaField in self.persona_history])+'\n'+english_version_of_user_input})
                res = self.polyencoderagent.act()
                print(res, flush=True)
                good_stories.append(self.polyencoderagent.act().text)
            self.addStoriesLive(good_stories)
        else:
            print("I don't remember anything", flush=True)
        self.history.append(english_version_of_user_input)
        print(self.history, flush=True)
        self.agent.observe({'episode_done': False, 'text': english_version_of_user_input})
        model_res = self.agent.act()

        json_return = dict()

        if (user_language != "en"):
            json_return['text'] = process_output_chatbot(model_res['text'])
            json_return['text'] = translate_base(json_return['text'], dest=user_language)
        else:
            json_return['text'] = process_output_chatbot(model_res['text'])

        json_return['user_lang'] = user_language
        json_return['stories'] = stories
        return jsonify(json_return)

    def get(self, val):
        if val == 'opt':
            return self.opt
        elif val == 'agent':
            return self.agent
        elif val == 'world':
            return self.world
        else:
            return None
