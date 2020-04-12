import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matix
    bag = [0]*len(words)
    for sentence_word in sentence_words:
        for index, word in enumerate(words):
            if word == sentence_word:
                # assign 1 if current word is in the vocab position
                bag[index] = 1
                if show_details:
                    print(f"found in baf: {word}")
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    prediction = bow(sentence, words, show_details=False)
    response = model.predict(np.array([prediction]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(response) if r>ERROR_THRESHOLD]

    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intents": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intents']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']==tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(message):
    ints = predict_class(message, model)
    result = get_response(ints, intents)
    return result

# create a GUI using tkinter
import tkinter as tk
import speech.speech2text as speech

class GUI:
    def __init__(self):
        self.base = tk.Tk()
        self.base.title("Placement Cell")
        self.base.geometry("600x440")
        self.base.resizable(0,0)

    def send(self, event=None):
        msg = self.EntryBox.get().strip()
        self.EntryBox.delete("0", tk.END)

        if msg != '':
            self.ChatLog.config(state= tk.NORMAL)
            self.ChatLog.insert(tk.END, "You: " + msg + '\n\n')
            self.ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

            res = chatbot_response(msg)
            self.ChatLog.insert(tk.END, "Bot: " + res + '\n\n')

            self.ChatLog.config(state=tk.DISABLED)
            self.ChatLog.yview(tk.END)
    
    def get_voice(self):
        #Give Feedback to user
        self.EntryBox.insert(0, "Listening...")
        self.EntryBox.config(state=tk.DISABLED)

        #Convert Speech to Text
        voice = speech.SpeechText()
        response = voice.convert_speech_text()

        self.EntryBox.config(state=tk.NORMAL)
        #Output voice response
        self.EntryBox.delete(0, tk.END)
        if not response=="":
            self.EntryBox.insert(0, response)

    def create_gui(self):
        #Create Chat window
        self.ChatLog = tk.Text(self.base, bg="white", height="8", width="50")
        self.ChatLog.config(state=tk.DISABLED)

        #Bind scrollbar to Chat window
        self.scrollbar = tk.Scrollbar(self.base, command=self.ChatLog.yview, cursor="mouse")
        self.ChatLog['yscrollcommand'] = self.scrollbar.set

        #Create the box to enter message
        self.EntryBox = tk.Entry(self.base, bg="white",width="50", font="Arial")
        self.EntryBox.bind("<Return>", self.send)
        self.EntryBox.focus_set()

        #Create Button to send message
        self.SendButton = tk.Button(self.base, text="Send", 
                        width="12", height=5, command=self.send)

        #Create Button to use voice
        self.VoiceButton = tk.Button(self.base, text="Voice", 
                        width="12", height=5, command=self.get_voice)

        #Place all components on the screen
        self.scrollbar.place(x=582,y=7, height=384)
        self.ChatLog.place(x=6,y=6, height=386, width=578)
        self.EntryBox.place(x=6, y=405, width=420)
        self.SendButton.place(x=430, y=405, height=25,width=80)
        self.VoiceButton.place(x=515, y=405, height=25,width=80)

        self.base.mainloop()


if __name__ == "__main__":
    gui_object = GUI()
    gui_object.create_gui()