from flask import Flask, request, jsonify, send_from_directory
import os
import tempfile
import torch
import numpy as np
from utils import get_model, preprocess_image

app = Flask(__name__)

LABELS = ["prediabetes", "insulin_resistance", "hypertension"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pth')


@app.route('/')
def index():
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'static'), 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'file missing'}), 400
    f = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.exists(MODEL_PATH):
        model = get_model(num_outputs=len(LABELS), weights_path=MODEL_PATH, device=device)
        x = preprocess_image(tmp_path).to(device)
        with torch.no_grad():
            logits = model(x)[0]
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
        note = 'Model predictions (trained on synthetic labels)'
    else:
        probs = np.random.rand(len(LABELS)).tolist()
        note = 'Placeholder probabilities (model.pth not found)'

    os.unlink(tmp_path)
    return jsonify({'labels': LABELS, 'probabilities': probs, 'note': note})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
