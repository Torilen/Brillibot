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

class BlenderBaseAgent:
    api_key = ''
    ip = ""
    history = []
    agent = None
    persona_history = []

    def __init__(self, personality, ip):
        self.addStoriesLive(personality)
        self.ip = ip

    def addStoriesLive(self, personality):
        self.history += personality
        self.persona_history += personality
        personalityText = ' \n'.join(["your persona: " + personaField for personaField in personality])
        print(personalityText)
        if (len(personality) > 0):
            self.agent.observe({'episode_done': False, 'text': personalityText})

    def speak(self, reply_text, keywordsUnlocked):
        user_language = detect(reply_text)
        # user_language = "en"
        english_version_of_user_input = translate_base(reply_text, src=user_language)
        self.history.append(english_version_of_user_input)
        print(self.history, flush=True)
        self.agent.observe({'episode_done': False, 'text': english_version_of_user_input})
        model_res = self.agent.act()
        print(model_res, flush=True)

        json_return = dict()

        if (user_language != "en"):
            json_return['text'] = process_output_chatbot(model_res['text'])
            json_return['text'] = translate_base(json_return['text'], dest=user_language)
        else:
            json_return['text'] = process_output_chatbot(model_res['text'])

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