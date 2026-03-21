import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_image_features(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
    return features.cpu().numpy()

def get_text_features(text):
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
    return features.cpu().numpy()
