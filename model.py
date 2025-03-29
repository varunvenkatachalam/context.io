import os
import requests
import json
import io
from flask import Flask, request, jsonify, render_template
from PIL import Image
import google.generativeai as genai

app = Flask(__name__)

# Configure API Keys
HF_API_KEY = ""
GOOGLE_API_KEY = ""
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
chat_model = genai.GenerativeModel("gemini-1.5-flash")
chat = chat_model.start_chat(history=[])

# Hugging Face API URLs
API_URL1 = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
API_URL2 = "https://api-inference.huggingface.co/models/Falconsai/text_summarization"

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/chat', methods=['GET', 'POST'])
def chat_with_ai():
    if request.method == 'POST':
        message = request.form.get('message')
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        response = chat.send_message(message)
        return jsonify({'response': response.text})
    return render_template('index.html')

@app.route('/chat/history', methods=['GET'])
def get_chat_history():
    return jsonify({'history': chat.history})

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        image = request.files['image']
        prompt = request.form.get('prompt')
        if not prompt:
            return jsonify({"error": "Prompt is missing"}), 400
        
        image_path = f"/tmp/{image.filename}"
        image.save(image_path)
        uploaded_file = genai.upload_file(image_path, mime_type="image/jpeg")
        chat_session = model.start_chat(history=[{"role": "user", "parts": [uploaded_file, prompt]}])
        response = chat_session.send_message(prompt)
        result = response.text if response else "Error analyzing the image"
        return jsonify({"result": result})
    return render_template('index1.html')

@app.route("/generate_image", methods=["GET", "POST"])
def generate_image():
    if request.method == "POST":
        prompt = request.form.get("prompt")
        if not prompt:
            return jsonify({"error": "Prompt is missing"}), 400
        
        response = requests.post(API_URL1, headers=headers, json={"inputs": prompt})
        if response.status_code == 200:
            try:
                image = Image.open(io.BytesIO(response.content))
                output_path = "static/output_image.png"
                image.save(output_path)
                return jsonify({"image_url": f"/{output_path}"})
            except IOError as e:
                return jsonify({"error": f"Image processing failed: {e}"}), 500
        else:
            return jsonify({"error": f"API request failed ({response.status_code})"}), 500
    return render_template("generateimage.html")

@app.route('/summary', methods=['GET', 'POST'])
def summarize():
    if request.method == 'POST':
        text = request.form.get('text')
        if not text:
            return jsonify({"error": "Text is required"}), 400
        output = requests.post(API_URL2, headers=headers, json={"inputs": text}).json()
        return jsonify(output)
    return render_template('summary.html')

@app.route('/about')
def about():
    return render_template('aboutme.html')

if __name__ == "__main__":
    app.run(debug=True, port=5656)