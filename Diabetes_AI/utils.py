from PIL import Image
import torchvision.transforms as T
import torch
import torchvision.models as models
import torch.nn as nn


def get_model(num_outputs=3, weights_path=None, device='cpu'):
    m = models.mobilenet_v2(pretrained=True)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, num_outputs))
    if weights_path:
        state = torch.load(weights_path, map_location=device)
        m.load_state_dict(state)
    m.to(device)
    m.eval()
    return m


def preprocess_image(image_path):
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return tf(img).unsqueeze(0)
