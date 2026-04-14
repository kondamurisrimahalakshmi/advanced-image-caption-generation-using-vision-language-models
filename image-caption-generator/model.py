import os

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


_MODEL_NAME = "Salesforce/blip-image-captioning-base"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_processor = None
_model = None


def _load_model():
    global _processor, _model
    if _processor is None or _model is None:
        _processor = BlipProcessor.from_pretrained(_MODEL_NAME)
        _model = BlipForConditionalGeneration.from_pretrained(_MODEL_NAME)
        _model.to(_DEVICE)
        _model.eval()


def _open_image(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    max_side = 1024
    w, h = img.size
    if max(w, h) > max_side:
        if w >= h:
            new_w = max_side
            new_h = int(h * (max_side / w))
        else:
            new_h = max_side
            new_w = int(w * (max_side / h))
        img = img.resize((new_w, new_h))

    return img


@torch.inference_mode()
def generate_caption(image_path: str) -> str:
    """
    Generate a natural language caption for an image on disk.
    """
    _load_model()
    image = _open_image(image_path)

    inputs = _processor(images=image, return_tensors="pt").to(_DEVICE)
    output_ids = _model.generate(
        **inputs,
        max_new_tokens=30,
        num_beams=5,
    )
    caption = _processor.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption

