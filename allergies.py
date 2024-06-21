import pyttsx3
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import nltk 
from sklearn.model_selection import train_test_split
import random
import warnings
warnings.simplefilter('ignore')
# nltk.download("punkt")

def speak(text): 
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    print("")
    print(f"==> Doctor AI : {text}")
    print("")
    Id = r'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0'
    engine.setProperty('voices',Id)
    engine.say(text=text)
    engine.runAndWait()
    
def speechrecognition():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening.....")
        r.pause_threshold = 1
        audio = r.listen(source,0,8)
        
    try:
        print("Recognizing....")
        query = r.recognize_google(audio,language="en")
        return query.lower()
    
    except:
        return ""


allergies_medicines = {
    "peanuts": {
       "cure": ["Antihistamine A", "Antihistamine B"],
        "symptons": ["Itching", "Swelling", "Vomiting", "Diarrhea", "Anaphylaxis"]
    },
    "pollen":{
        "cure": ["Antihistamine C", "Antihistamine D"],
        "symptons": ["nasal congestion","runny nose"],
    },
    "dust":{ 
        "cure" :["Antihistamine E", "Antihistamine F"],
        "symptons":[ "Sneezing", "Watery eyes", ]
    },
    "Hay Fever":{
        "cure": ["cetirizine"],
        "symptons": ["Frequent headaches", "Continuous sneezing, especially after waking up in the morning"],
    },
    "Conjunctivitis": {
        "cure": ["Antihistamine eye drops"],
        "symptons": ["Red eye", "Irritation, itching, and a sensation of the presence of a foreign particle in the eye"],
    },
    "Hives": {
        "cure": ["Corticosteroid creams or ointments"],
        "symptons": ["Painful swelling of lips and throat", "Batches of red or skin coloured welts"],
    },
    "Food": {
        "cure": ["Epinephrine auto-injectors"],
        "symptons": ["Bloating", "Nausea and vomiting", "Metallic taste in mouth"],
    },
    "Drug": {
        "cure": ["Corticosteroids"],
        "symptons": ["Skin rash", "Hives", "Fever"],
    },
    "Insect Sting": {
        "cure": ["Epinephrine auto-injectors"],
        "symptons": ["itching and swelling in areas other than the sting site", "Feeling faint, dizzy"],
    },
    "Latex ": {
        "cure": ["Epinephrine "],
        "symptons": ["Hives", "Itching", "Stuffy or runny nose"],
    },
}

   
training_data = []
labels = [] 

for  intent , data in allergies_medicines.items():
    for pattern in data['symptons']:
        training_data.append(pattern.lower())
        labels.append(intent)

# print(training_data)
# print(labels)
 

vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize,stop_words="english",max_df=0.8,min_df=1)  
X_train = vectorizer.fit_transform(training_data)
X_train,X_test,Y_train,Y_test = train_test_split(X_train,labels,test_size=0.5,random_state=42,stratify=labels)

model =SVC(kernel='linear', probability=True, C=1.0)
model.fit(X_train , Y_train)

predictions = model.predict(X_test)

def predict_intent(user_input):
    user_input = user_input.lower()
    input_vector = vectorizer.transform([user_input])
    allergies_medicines = model.predict(input_vector)[0]
    return allergies_medicines

print("Hi I'm your AI Doctor, How may i help you?")
while True:
    user_input = speechrecognition()
    if user_input.lower() == "exit":
        print("AI Doctor Bye!")
        break
    
    intent = predict_intent(user_input)
    if intent in allergies_medicines:
        responses = allergies_medicines[intent]['cure']
        response = random.choice(responses)
        speak(response)
    else:
        speak("I'm not sure how to answer to that")
