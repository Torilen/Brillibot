from flask import Flask, request, jsonify
from flask_restplus import Resource, Api
from flask_cors import CORS
from typing import Dict, Any
from threading import Thread
from OpenSSL import SSL
from werkzeug.middleware.proxy_fix import ProxyFix
import os

from tools.Utils import process_output_chatbot
from structure.GrafbotAgent import GrafbotAgent
import json

context = SSL.Context(SSL.SSLv23_METHOD)
cer = 'grafbot.com.crt'
key = 'grafbot.com.key'

env = "ubuntu"
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app)
CORS(app)



#api = Api(app)

SHARED: Dict[Any, Any] = {}
    
os.environ['JAVAHOME'] = "/usr/lib/jvm/java-1.11.0-openjdk-amd64"

@api.route('/interact')
class Interact(Resource):
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
            return SHARED[request.remote_addr].speak(request.form['data'])

@api.route('/createAgent')
class CreateAgent(Resource):
    def post(self):
        persona = json.loads(request.form['data'])
        print(persona)
        shared_temp = SHARED.copy()
        SHARED[request.remote_addr] = GrafbotAgent(personality=persona, ip=request.remote_addr)
        if (request.remote_addr not in list(shared_temp.keys())):
            res = dict()
            res['creation'] = 1
            return jsonify(res)
        else:
            res = dict()
            res['creation'] = 2
            return jsonify(res)
        res = dict()
        res['creation'] = 0
        return jsonify(res)

@api.route('/reset')
class Reset(Resource):
    def post(self):
        if (request.remote_addr in list(SHARED.keys())):
            SHARED[request.remote_addr].get('agent').reset()
            res = dict()
            res['reset'] = 1
            return jsonify(res)

if __name__ == '__main__':
    context = (cer, key)
    app.run(host="0.0.0.0", port=int("5000"), use_reloader=False, debug=True, threaded=True)

