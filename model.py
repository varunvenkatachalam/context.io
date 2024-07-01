import os
import requests
import json
import base64
from flask import Flask, request, jsonify, render_template
import io
from PIL import Image


import google.generativeai as genai

app = Flask(__name__)

model1 = "llama3"
model2 = "llava"
API_URL1 = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
API_URL2 = "https://api-inference.huggingface.co/models/Falconsai/text_summarization"
headers = {"Authorization": "Bearer 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"}

api_key = "AIzaSyDnhDLrV74ffbfHx7fEto9Mf_SEquohqao"

# Configure Google Generative AI
genai.configure(api_key=api_key)

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

os.environ['GOOGLE_API_KEY'] = "AIzaSyDnhDLrV74ffbfHx7fEto9Mf_SEquohqao"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model1 = genai.GenerativeModel('gemini-pro')
chat_model = genai.GenerativeModel('gemini-pro')

# Initialize chat history
chat = chat_model.start_chat(history=[])

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/chat')
def chat_page():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat_with_ai():
    message = request.form.get('message')
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    # Send message to the Generative AI model
    response = chat.send_message(message)
    
    # Return the response
    return jsonify({'response': response.text})

@app.route('/chat/history', methods=['GET'])
def get_chat_history():
    # Return the chat history
    return jsonify({'history': chat.history})

@app.route('/analyze')
def index():
    return render_template('index1.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image = request.files['image']
    prompt = request.form.get('prompt')

    if not prompt:
        return jsonify({"error": "Prompt is missing"}), 400
    
    if image.filename == '':
        return jsonify({"error": "No selected image"}), 400
    
    # Save the uploaded image to a temporary file
    image_path = "/tmp"
    image.save(image_path)

    # Upload the image to Gemini
    uploaded_file = upload_to_gemini(image_path, mime_type="image/jpeg")
    
    # Start a chat session with the model and send the prompt along with the uploaded image
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [uploaded_file, prompt],
            }
        ]
    )

    response = chat_session.send_message(prompt)

    if response:
        result = response.text
    else:
        result = "Error occurred while analyzing the image"

    return jsonify({"result": result})

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

@app.route("/generate_image", methods=["GET", "POST"])
def generate_image():
    if request.method == "POST":
        prompt = request.form.get("prompt")
        if not prompt:
            return jsonify({"error": "Prompt is missing"}), 400

        payload = {"inputs": prompt}
        response = requests.post(API_URL1, headers=headers, json=payload)

        if response.status_code == 200:
            try:
                image_bytes = response.content
                image = Image.open(io.BytesIO(image_bytes))
                image.save("static/output_image.png")
                return jsonify({"image_url": "/static/output_image.png"})
            except IOError as e:
                return jsonify({"error": f"Unable to process the image. Details: {e}"}), 500
        else:
            return jsonify({"error": f"API request failed with status code {response.status_code}"}), 500
    else:
        return render_template("generateimage.html")
    

def query(payload):
    response = requests.post(API_URL2, headers=headers, json=payload)
    return response.json()


@app.route('/summery')
def summarize():
    return render_template('summery.html')


@app.route('/summery', methods=['POST'])
def summarize1():
    text = request.form['text']
    output = query({"inputs": text})
    return jsonify(output)


@app.route('/about')
def about():
    return render_template('aboutme.html')
    
if __name__ == "__main__":
    app.run(debug=True,port=5656)
