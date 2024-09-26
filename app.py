# app.py
from flask import Flask, render_template, request, jsonify
from assistant.core import VirtualAssistant

app = Flask(__name__)
assistant = VirtualAssistant()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_command', methods=['POST'])
def process_command():
    command = request.json['command']
    response = assistant.process_command(command)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)