from googletrans import Translator
from bs4 import BeautifulSoup
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
import urllib
import six

def translate_by_url(text, src='en', dest='en'):
    link = "https://translate.google.fr/?hl=fr#view=home&client=gtx&op=translate&sl="+src+"&tl="+dest+"&text="+text
    print(link)
    myfile = urllib.request.urlopen(link).read()
    soup = BeautifulSoup(myfile, "html")
    div = soup.find("span", {"class": "tlid-translation translation"})

    content = str(div)

    print("Le contenu de la traduction est : {}".format(content))

def translate_base(text, src='en', dest='en'):
    translator = Translator()
    print("SRC: "+src+" DEST: "+dest+" TEXT: "+text)
    return translator.translate(text, src=src, dest=dest).text

def translate_by_api(text, env, src='en', dest='en'):
    urls = dict()
    urls["windows"] = "C:/Users/scorp/Grafbot-c753fa94ac04.json"
    urls["ubuntu"] = "/root/Grafbot/Grafbot-c753fa94ac04.json"

    path = dict()
    path["windows"] = "web/"
    path["ubuntu"] = "/root/Grafbot/app/web/"
    credentials = service_account.Credentials.from_service_account_file(urls[env])
    translate_client = translate.Client(credentials=credentials)

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(
        text, target_language=dest, model="nmt")

    print(u'Text: {}'.format(result['input']))
    print(u'Translation: {}'.format(result['translatedText']))
    print(u'Detected source language: {}'.format(
        result['detectedSourceLanguage']))
    return result['translatedText']

def detect(text):
    translator = Translator()
    print(text)
    print(translator.detect(text))
    print(translator.detect(text).lang)
    return translator.detect(text).lang

if __name__ == '__main__':
    translate_by_url("Hi how are you ?")