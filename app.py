import json
import random
import nltk
from flask import Flask, render_template, request

# Download necessary NLTK data (required for cloud hosting)
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

# Convert text to numbers (Vectorization)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(inputs)

# Train Model
clf = LogisticRegression(random_state=0, max_iter=1000)
clf.fit(X, tags)

def chat_response(user_text):
    input_vector = vectorizer.transform([user_text])
    predicted_tag = clf.predict(input_vector)[0]

    # Simple confidence check
    probs = clf.predict_proba(input_vector)
    max_prob = max(probs[0])

    if max_prob < 0.2:
        return "I am sorry, I do not understand."

    return random.choice(responses[predicted_tag])

# --- PART 2: WEB APP ---
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