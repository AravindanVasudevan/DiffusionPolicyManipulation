from __future__ import annotations
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import torch



_MODEL_NAME = "openai/clip-vit-base-patch32"
_tokenizer = None
_text_model = None

def _load(device):

    global _tokenizer, _text_model
    if _text_model is None:
        _tokenizer = CLIPTokenizer.from_pretrained(_MODEL_NAME)
        _text_model = CLIPTextModelWithProjection.from_pretrained(_MODEL_NAME).to(device).eval()
        for p in _text_model.parameters():
            p.requires_grad_(False)

    return _tokenizer, _text_model

@torch.no_grad()
def embed_text(strings: list[str], device) -> torch.Tensor:

    tokenizer, model = _load(device)
    tokens = tokenizer(strings, padding=True, truncation=True, return_tensors="pt").to(device)

    return model(**tokens).text_embeds