import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

_model = None
_processor = None

def _load():
    global _model, _processor
    if _model: return
    _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def rank_images(images: list, text: str) -> list:
    if len(images) <= 1:
        return list(images)
    _load()
    inputs = _processor(
        text=[text], images=images, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        out = _model(**inputs)
    scores = out.logits_per_image.squeeze().tolist()
    if isinstance(scores, float):
        scores = [scores]
    ranked = sorted(zip(scores, images), reverse=True)
    return [img for _, img in ranked]