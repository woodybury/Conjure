import os
import listening
import speech_recognition as sr
from nltk.corpus import wordnet
import simpleaudio as sa
import connect
import cv2, time
import numpy as np

channel = connect.join()

searchsound = sa.WaveObject.from_wave_file('sound/search.wav')

r = sr.Recognizer()

if os.uname()[1] == 'raspberrypi':
    mic = sr.Microphone(device_index=1)
else:
    mic = sr.Microphone()

def transform( image ):
    img = cv2.imread(image, 0)

    # resize img for transform
    img = cv2.resize(img, (16,24), interpolation = cv2.INTER_NEAREST)

    # add img together x3 for total transform
    img = np.concatenate((img, img, img), axis=1)

    # flatten array
    img = img.flatten()

    # stringify for server
    transformSend = ""
    for ele in img:
        transformSend+=(" "+str(ele))

    # if you want to look at the numbers :)
    print (transformSend)

    # uncomment this to send to server
    channel.push("input",{"body": transformSend})


def website():
    print ('listening for website')
    listening.recognition(recognize_url, 'website', False)

def recognize_url():
    playsound = searchsound.play()
    playsound.wait_done()
    print ('listening for url')
    with mic as source:
        audio = r.listen(source)
    response = {
        "success": True,
        "error": None,
        "text": None
    }
    try:
        response["text"] = r.recognize_google(audio)
    except sr.RequestError:
        response["success"] = False
        response["error"] = "unavailable"
    except sr.UnknownValueError:
        response["error"] = "unknown"

    if response['text']:

        term = response['text'].replace(" ", "_")
        print ('search term: ' + term)
        cmd="say " + term
        os.system(cmd)

        transform('./img/dom1.png')


def recognize_search():
    playsound = searchsound.play()
    playsound.wait_done()
    print ('listening for search terms')
    with mic as source:
        audio = r.listen(source)
    response = {
        "success": True,
        "error": None,
        "text": None
    }
    try:
        response["text"] = r.recognize_google(audio)
    except sr.RequestError:
        response["success"] = False
        response["error"] = "unavailable"
    except sr.UnknownValueError:
        response["error"] = "unknown"

    if response['text']:

        term = response['text'].replace(" ", "_")
        print ('search term: ' + term)
        cmd="say " + term
        os.system(cmd)

        transform('./img/dom.png')

        synsets = []
        lemmas = []
        hyponyms = []
        hypernyms = []
        holonyms = []
        meronyms = []
        entailments = []

        sep = '.'

        for syn in wordnet.synsets(term):
            name = syn.name().split(sep, 1)[0]
            synsets.append(name)

            for lem in syn.lemmas():
                lemmas.append(lem.name())

            for hypo in syn.hyponyms():
                name = hypo.name().split(sep, 1)[0]
                hyponyms.append(name)

            for hyper in syn.hypernyms():
                name = hyper.name().split(sep, 1)[0]
                hypernyms.append(name)

            for holo in syn.part_holonyms():
                name = holo.name().split(sep, 1)[0]
                holonyms.append(name)

            for mero in syn.part_meronyms():
                name = mero.name().split(sep, 1)[0]
                meronyms.append(name)

            for ent in syn.entailments():
                name = ent.name().split(sep, 1)[0]
                entailments.append(name)

        synset = str(set(synsets))
        print('synsets: ' + synset)
        cmd="say " + 'synsets: ' + synset
        # os.system(cmd)

        lemma = str(set(lemmas))
        print('lemmas: ' + lemma)
        cmd="say " + 'lemmas: ' + lemma
        # os.system(cmd)

        hyponym = str(set(hyponyms))
        print('hyponyms: ' + hyponym)
        cmd="say " + 'hyponyms: ' + hyponym
        # os.system(cmd)

        hypernym = str(set(hypernyms))
        print('hypernyms: ' + hypernym)
        cmd="say " + 'hypernyms: ' + hypernym
        # os.system(cmd)

        holonym = str(set(holonyms))
        print('holonyms: ' + holonym)
        cmd="say " + 'holonyms: ' + holonym
        # os.system(cmd)

        meronym = str(set(meronyms))
        print('meronyms: ' + meronym)
        cmd="say " + 'meronyms: ' + meronym
        # os.system(cmd)

        entailment = str(set(entailments))
        print('entailments: ' + entailment)
        cmd="say " + 'entailments: ' + entailment
        # os.system(cmd)

if __name__ == "__main__":
    #tutuorial
    transform('./img/Typology.png')
    #website
    website()
    #search
    while (1):
        print ('listening for keyword')
        listening.recognition(recognize_search, 'search', False)