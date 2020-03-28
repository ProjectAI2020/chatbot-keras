import speech_recognition as sr

r = sr.Recognizer()
r.energy_threshold = 2000


with sr.Microphone() as source:
    print('Say Something')
    audio = r.listen(source)

try:
   print("Speech was:" + r.recognize_google(audio))
except LookupError:
   print('Speech not understood')
