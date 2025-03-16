from flask import Flask, request, render_template,jsonify
import os
import gaussian_model as gm
import testmodel as final
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    name = request.form.get('name') 

    if not name:
        return jsonify({"error": "Name is required"}), 400

    response = gm.train_model(name)  

    if response:
        return jsonify({"message": "Training successful"}), 200
    else:
        return jsonify({"error": "Training failed","text": response}), 500

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio_data' not in request.files:
        return 'No audio file uploaded.', 400

    audio_file = request.files['audio_data']
    if audio_file.filename == '':
        return 'No selected file.', 400
    file_path = os.path.join('training_audios', audio_file.filename)
    
    audio_file.save(file_path)
    
    return 'File uploaded successfully.', 200

@app.route('/test', methods=['POST'])
def uploadpro():
    if 'audio_data' not in request.files:
        return 'No audio file uploaded.', 400

    audio_file = request.files['audio_data']
    if audio_file.filename == '':
        return 'No selected file.', 400
    file_path = os.path.join('test', audio_file.filename)
    audio_file.save(file_path)
    print(file_path)
    rey=final.test(file_path)

    return 'File uploaded successfully.'+rey, 200

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)

    except Exception as e:
        print(f"Server error: {e}")  
        while True:
            pass  