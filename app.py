import os
from flask import Flask, render_template, request
import google.generativeai as genai

# --- CONFIGURATION ---
# PASTE YOUR API KEY HERE INSIDE THE QUOTES
API_KEY = "AIzaSyA-LcU6yKCVe6sNoXU8ovXyq6iBZrBXgac" 

genai.configure(api_key=API_KEY)

# --- THE KNOWLEDGE BASE ---
# This is what the AI "knows". You can paste the entire university brochure content here.
gub_knowledge = """
You are a helpful AI assistant for Green University of Bangladesh (GUB).
Answer questions based ONLY on the following information. If you don't know, say "Please contact the admission office."

FACT SHEET:
- University Name: Green University of Bangladesh (GUB)
- Location: The Permanent Campus is at Purbachal American City, Kanchan, Rupganj.
- City Office: Begum Rokeya Sarani, Dhaka.
- Departments: CSE, EEE, Textile, BBA, LLB, English, Sociology, Journalism.
- CSE Tuition: Total cost is approx 5,26,000 BDT to 7,14,500 BDT (depends on waiver).
- Admission Fee: Tk. 20,000 (Non-refundable).
- Waivers: Up to 100% based on SSC/HSC GPA. Special waiver for freedom fighters and siblings.
- Vice Chancellor: Prof. Dr. Md. Golam Samdani Fakir.
- Website: www.green.edu.bd
- Contact: 01757074301, 01757074302
- Clubs: Robotics Club, Debating Society (GUDC), Cultural Club.
- Grading: A+ (80%+), D (40% is pass).
"""

# --- FLASK APP ---
app = Flask(__name__)
model = genai.GenerativeModel('gemini-1.5-flash')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_question = request.args.get('msg')
    
    # We construct a "Prompt" for the AI
    # We tell it: "Here is the knowledge. Here is the user question. Answer it."
    full_prompt = f"{gub_knowledge}\n\nStudent Question: {user_question}\nAI Answer:"
    
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return "Sorry, my brain is taking a nap. (API Error)"

if __name__ == "__main__":
    app.run(debug=True)