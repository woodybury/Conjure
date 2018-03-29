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

        print ('search term: ' + response['text'])

        synonyms = []

        for syn in wordnet.synsets(response['text']):
            for l in syn.lemmas():
                synonyms.append(l.name())

        print('synsets: ' + str(set(synonyms)))


while (1):
    print ('listening for keyword')
    listening.recognition(recognize, 'search', False)