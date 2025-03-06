import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from flask import Flask, request, jsonify, render_template
from transformers import MarianMTModel, MarianTokenizer
import speech_recognition as sr
import pyttsx3

app = Flask(__name__)

def load_translation_model(src_lang="en", tgt_lang="fr"):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**inputs)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Speak now...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return "Error: Unable to fetch results."

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    src_text = data.get("text", "")
    src_lang = "en"
    tgt_lang = "fr"
    tokenizer, model = load_translation_model(src_lang, tgt_lang)
    translated_text = translate_text(src_text, tokenizer, model)
    speak_text(translated_text)
    return jsonify({"translated_text": translated_text})

if __name__ == '__main__':
    app.run(debug=True)
