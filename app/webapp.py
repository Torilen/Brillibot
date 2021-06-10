from flask import Flask, request, jsonify
from flask_restx import Resource, Api, fields
from flask_cors import CORS
from typing import Dict, Any
from threading import Thread
from OpenSSL import SSL
from werkzeug.middleware.proxy_fix import ProxyFix
import os
import sys

from tools.Utils import process_output_chatbot
from structure.GrafbotAgent import GrafbotAgent
from structure.GPT3Agent import GPT3Agent
from structure.BlenderBaseAgent import BlenderBaseAgent
import json

context = SSL.Context(SSL.SSLv23_METHOD)
cer = 'grafbot.com.crt'
key = 'grafbot.com.key'
api_key_openai = ''
model_to_use = 'grafbot'
env = "ubuntu"
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app)
CORS(app)

confReset = api.model('reset', {})
confInteract = api.model('interact', {
    'data': fields.String(description="Message de l'utilisateur pour l'agent", required=True, example='Hi, how are you ?'),
    'keywordsUnlocked': fields.List(fields.Integer, description="Liste des identifiants de mots-clés déjà débloqué par l'utilisateur. Peut être une liste vide []", required=True, example=[0,1,5,4,8]),
})
confCreateAgent = api.model('createAgent', {
    'data': fields.List(fields.String, description="Données de chargement d'un agent donné au format suivant : ligne de personnalité;ids de mots-clés débloqués avec cette query séparés par des |; réponse scriptée à retourner si les mots-clés sont détectés; ids de mots-clés conditionnant la détection d'autres mots-clés séparés par des |\\n etc", required=True, example=["My name is Aniss;;;","I have 23 years old;;;","My job is Data Scientist;;;","My cay has already eaten a dog;0|1|5;Réponse associée à my cat has already eaten a dog;2|3","Souvenir 2;;;"]),
    'model': fields.String(description="Choix du modèle génératif à utiliser (grafbot/gpt3)", required=True, example='grafbot'),
})


#api = Api(app)

SHARED: Dict[Any, Any] = {}
    
os.environ['JAVAHOME'] = "/usr/lib/jvm/java-1.11.0-openjdk-amd64"

@api.route('/interact', endpoint='interact', doc={"description": r"""@returns : {text:chatbot_answer, user_lang:country_code, good_stories:list[text_stories], score:list[float_stories], keywordsId:list[keywordsUnlocked]}
    > Ce endpoint renvoie une réponse associé au code langue utilisateur détecté, une liste de souvenirs, une liste de scores de pertinence associée à la liste de souvenirs et une liste d'indentifiants de mots clés débloqués par la query"""})
class Interact(Resource):
    @api.expect(confInteract)
    def _interactive_running(self, opt, reply_text):
        reply = {'episode_done': False, 'text': reply_text}
        SHARED[request.remote_addr].get('agent').observe(reply)
        model_res = SHARED[request.remote_addr].get('agent').act()
        return model_res

    def post(self):
        if(not request.remote_addr in list(SHARED.keys())):
            res = dict()
            res['text'] = "You have to define me before"
            res['user_lang'] = "en"
            return jsonify(res)
        else:
            return SHARED[request.remote_addr].speak(request.form['data'], request.form['keywordsUnlocked'])

@api.route('/createAgent', endpoint='createAgent', doc={"description": r"""@returns : {creation: int}
    > Ce endpoint créé un agent avec les données d'input. Il renvoie un code qui permet de savoir si l'agent a correctement été créé.
    > 0 -> Erreur, l'agent n'est pas créé
    > 1 -> OK, l'agent est correctement créé
    > 2 -> OK, l'agent est correctement créé mais il existait déjà un agent associé à cette IP et il a été écrasé"""})
class CreateAgent(Resource):
    @api.expect(confCreateAgent)
    def post(self):
        personaData = json.loads(request.form['data'])
        model = request.form['model']
        print(personaData)
        persona = list()
        keywordsId = list()
        keywordsCond = list()
        answers = list()
        for e in personaData:
            eSplit = e.split(";")
            persona.append(eSplit[0])
            keywordsId.append(eSplit[1])
            answers.append(eSplit[2])
            keywordsCond.append(eSplit[3])
        print(persona, keywordsId, answers)
        shared_temp = SHARED.copy()
        if model == 'grafbot':
            SHARED[request.remote_addr] = GrafbotAgent(personality=persona, ip=request.remote_addr, keywordsId=keywordsId, answers=answers, keywordsCond=keywordsCond)
        elif model == 'gpt3':
            SHARED[request.remote_addr] = GPT3Agent(personality=persona, ip=request.remote_addr,
                                                       keywordsId=keywordsId, answers=answers,
                                                       keywordsCond=keywordsCond, api_key=os.getenv("openai_api_key"))
        elif model == 'base':
            SHARED[request.remote_addr] = BlenderBaseAgent(personality=persona, ip=request.remote_addr,
                                                    keywordsId=keywordsId, answers=answers,
                                                    keywordsCond=keywordsCond, api_key=os.getenv("openai_api_key"))
        if (request.remote_addr not in list(shared_temp.keys())):
            res = dict()
            res['creation'] = 1
            res['debug'] = "Model : {}\n" \
                           "IP : {}\n" \
                           "It's a new agent.".format(model, request.remote_addr)
            return jsonify(res)
        else:
            res = dict()
            res['creation'] = 2
            res['debug'] = "Model : {}\n" \
                           "IP : {}\n" \
                           "You already created an agent. It was erased.".format(model, request.remote_addr)
            return jsonify(res)
        res = dict()
        res['creation'] = 0
        res['debug'] = "Model : {}\n" \
                       "IP : {}\n" \
                       "Error, your agent wasn't created.".format(model, request.remote_addr)
        return jsonify(res)

@api.route('/reset', endpoint='reset', doc={"description": r"""@returns : {reset: int}
    > Ce endpoint permet de supprimer l'agent d'un utilisateur.
    > 1 -> OK, l'agent est correctement supprimé"""})
class Reset(Resource):
    @api.expect(confReset)
    def post(self):
        if (request.remote_addr in list(SHARED.keys())):
            SHARED[request.remote_addr].get('agent').reset()
            res = dict()
            res['reset'] = 1
            return jsonify(res)

if __name__ == '__main__':
    context = (cer, key)
    print(sys.argv)
    app.run(host="0.0.0.0", port=int("5000"), use_reloader=False, debug=True, threaded=True)

