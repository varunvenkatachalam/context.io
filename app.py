from flask import Flask, request, jsonify, render_template
import os
import google.generativeai as genai

app = Flask(__name__)

# Configure Google Generative AI with API key
os.environ['GOOGLE_API_KEY'] = "AIzaSyDnhDLrV74ffbfHx7fEto9Mf_SEquohqao"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-pro')
chat_model = genai.GenerativeModel('gemini-pro')

# Initialize chat history
chat = chat_model.start_chat(history=[])

@app.route('/')
def index():
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

if __name__ == '__main__':
    app.run(debug=True)
