import json
import random
import nltk
from flask import Flask, render_template, request

# --- ESSENTIAL DOWNLOADS FOR CLOUD ---
# These lines ensure it works on Codespaces/Render
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- PART 1: TRAIN THE MODEL ---
# Load Data
with open('intents.json') as file:
    data = json.load(file)

tags = []
inputs = []
responses = {}

for intent in data['intents']:
    responses[intent['tag']] = intent['responses']
    for pattern in intent['patterns']:
        inputs.append(pattern)
        tags.append(intent['tag'])

# Vectorization (Convert Text -> Numbers)
# This matches your "Preprocessing" and "Model Selection" methodology
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(inputs)

# Train Model (Logistic Regression)
clf = LogisticRegression(random_state=0, max_iter=1000)
clf.fit(X, tags)

def chat_response(user_text):
    # Predict the tag
    input_vector = vectorizer.transform([user_text])
    predicted_tag = clf.predict(input_vector)[0]
    
    # Check Confidence
    probs = clf.predict_proba(input_vector)
    max_prob = max(probs[0])
    
    # DEBUG PRINT: This will show in your terminal (black box)
    print(f"User asked: '{user_text}'")
    print(f"Predicted: '{predicted_tag}' with confidence: {max_prob}")

    # If the bot is confused (less than 10% sure), say sorry
    if max_prob < 0.1:
        return "I am not sure about that. Please contact the GUB Admission office."
    
    return random.choice(responses[predicted_tag])

# --- PART 2: FLASK WEB APP ---
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chat_response(userText)

if __name__ == "__main__":
    app.run(debug=True)