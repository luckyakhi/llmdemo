from transformers import AutoModel

AutoModel.from_pretrained("bert-base-uncased", force_download=True)
