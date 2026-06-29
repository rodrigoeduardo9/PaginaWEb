import sys
import os
import json
import torch
from utils import get_model, preprocess_image

LABELS = ["prediabetes", "insulin_resistance", "hypertension"]


def main(image_path):
    model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(model_path):
        print(json.dumps({'error': 'model.pth not found'}))
        return
    model = get_model(num_outputs=len(LABELS), weights_path=model_path, device=device)
    x = preprocess_image(image_path).to(device)
    with torch.no_grad():
        logits = model(x)[0]
        probs = torch.sigmoid(logits).cpu().numpy().tolist()
    out = {'image': image_path, 'labels': LABELS, 'probabilities': probs}
    print(json.dumps(out))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python run_predict.py PATH_TO_IMAGE')
    else:
        main(sys.argv[1])
