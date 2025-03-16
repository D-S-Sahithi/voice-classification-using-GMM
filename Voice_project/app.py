from flask import Flask, request, render_template
import os
app = Flask(__name__)
AUDIO_FOLDER = 'training_audios'
os.makedirs(AUDIO_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = AUDIO_FOLDER
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio_data' not in request.files:
        return 'No audio file uploaded.', 400

    audio_file = request.files['audio_data']
    if audio_file.filename == '':
        return 'No selected file.', 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
    
    audio_file.save(file_path)
    
    return 'File uploaded successfully.', 200

if __name__ == '__main__':
    app.run(debug=True)
