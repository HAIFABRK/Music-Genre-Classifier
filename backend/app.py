from flask import Flask, request, jsonify
import librosa
import numpy as np
import joblib
import os
from flask_cors import CORS
app = Flask(__name__)


CORS(app)
try:
    svm_model = joblib.load("svm_model.pkl")
    print("svm_model chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement de svm_model.pkl : {e}")

try:
    label_encoder = joblib.load("encoder.pkl")
    print("encoder chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement de encoder.pkl : {e}")

# Function to extract MFCC features from a WAV file
def extract_mfcc(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        features = np.mean(mfccs, axis=1)
        return features.tolist()
    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques : {e}")
        return None


@app.route('/svm_service', methods=['POST'])
def svm_service():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save the uploaded file locally
        file_path = f"temp_audio.wav"
        file.save(file_path)
        # Extract features
        features = extract_mfcc(file_path)
        if features is None:
            return jsonify({"error": "Feature extraction failed"}), 400
        
        # Predict the genre
        prediction = svm_model.predict([features])
        genre = label_encoder.inverse_transform(prediction)
        return jsonify({"genre": genre[0]})
    except Exception as e:
        print(f"Erreur lors de la prédiction : {e}")
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)