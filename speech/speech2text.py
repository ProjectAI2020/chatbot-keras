import speech_recognition as sr

class SpeechText:
	def __init__(self):
		self.recognizer = sr.Recognizer()

	def convert_speech_text(self):
		with sr.Microphone() as source:
			audio = self.recognizer.listen(source)
		
		try:
			response = self.recognizer.recognize_google(audio)
		except:
			response = ""

		return response
