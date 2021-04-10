from flask import jsonify
from parlai.scripts.interactive import setup_args
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from tools.Translator import translate_base, detect, translate_by_url, translate_by_api
from tools.Utils import process_output_chatbot
from tools.EntityExtractor import get_entities
from tools.Embedder import concatEmbeddingFr, concatEmbeddingEn, getContextualEmbedding
from tools.Converter import Entities2Tuples
from structure.SemKG import SemKG
from structure.EpiKG import EpiKG

class GrafbotAgent:
    parser = setup_args()
    opt = None
    agent = None
    world = None
    semkg = SemKG()
    epikg = EpiKG()

    def __init__(self, personality):
        self.opt = self.parser.parse_args(print_args=False)
        self.opt['task'] = 'parlai.agents.local_human.local_human:LocalHumanAgent'
        self.agent = create_agent(self.opt, requireModelExists=True)
        self.addStoriesLive(personality[:3])
        self.learn(personality[3:])
        #self.addStoriesLive(personality)
        self.world = create_task(self.opt, self.agent)

    def addStoriesLive(self, personality):
        personalityText = ' \n'.join(["your persona: " + personaField for personaField in personality])
        print(personalityText)
        if(len(personality) > 0):
            self.agent.observe({'episode_done': False, 'text': personalityText})

    def learn(self, sentences):
        for sentence in sentences:
            entities = get_entities(sentence)
            tuples = Entities2Tuples(entities, "linear")
            self.semkg.add_relations(tuples, self.epikg, sentence)

    def speak(self, reply_text):
        print("Reply : "+reply_text)
        user_language = detect(reply_text)
        #user_language = "en"
        english_version_of_user_input = translate_base(reply_text, src=user_language)
        #english_version_of_user_input = reply_text
        #embedded = concatEmbedding(getContextualEmbedding(english_version_of_user_input, False))
        entities = get_entities(english_version_of_user_input)
        stories = self.semkg.get_stories(self.epikg, [x[0] for x in entities])
        print("STORIES: ")
        print(stories)
        self.addStoriesLive(stories)
        reply = {'episode_done': False, 'text': english_version_of_user_input}
        self.get('agent').observe(reply)
        model_res = self.get('agent').act()

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
