import os
import listening
import speech_recognition as sr
from nltk.corpus import wordnet
import simpleaudio as sa

searchsound = sa.WaveObject.from_wave_file('sound/search.wav')

r = sr.Recognizer()

if os.uname()[1] == 'raspberrypi':
    mic = sr.Microphone(device_index=1)
else:
    mic = sr.Microphone()

def recognize():
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
        # os.system(cmd)

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
        cmd="say " + synset
        # os.system(cmd)

        lemma = str(set(lemmas))
        print('lemmas: ' + lemma)
        cmd="say " + lemma
        # os.system(cmd)

        hyponym = str(set(hyponyms))
        print('hyponyms: ' + hyponym)
        cmd="say " + hyponym
        # os.system(cmd)

        hypernym = str(set(hypernyms))
        print('hypernyms: ' + hypernym)
        cmd="say " + hypernym
        # os.system(cmd)

        holonym = str(set(holonyms))
        print('holonyms: ' + holonym)
        cmd="say " + holonym
        # os.system(cmd)

        meronym = str(set(meronyms))
        print('meronyms: ' + meronym)
        cmd="say " + meronym
        # os.system(cmd)

        entailment = str(set(entailments))
        print('entailments: ' + entailment)
        cmd="say " + entailment
        # os.system(cmd)


while (1):
    print ('listening for keyword')
    listening.recognition(recognize, 'search', False)