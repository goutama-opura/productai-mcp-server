import requests
from io import BytesIO
from PIL import Image
import torch
import clip
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def fetch_image(url: str) -> Image.Image:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"Error fetching image from {url}: {e}")
        return None

def generate_image_embedding(img: Image.Image) -> list:
    if img is None:
        return []
    try:
        img_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(img_input)
        embedding = embedding.cpu().numpy().flatten()
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating image embedding: {e}")
        return []
