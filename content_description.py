import torch
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
from datasets import load_dataset

model = InstructBlipForConditionalGeneration.from_pretrained(
    "Salesforce/instructblip-vicuna-7b",
    load_in_4b = False,
    torch_dtype = torch.bfloat16,
)

processor = InstructBlipProcessor(
    "Salesforce/instructblip-vicuna-7b",
)

datasets = [
    ("detection-datasets/fashionpedia", None, "val"),
    ("keremberke/nfl-object-detection", "mini", "test"),
    ("keremberke/plane-detection", "mini", "train"),
    ("Matthijis/snacks", None, "validation"),
    ("romkmr/mini_pets", None, "test"),
    ("keremberke/pokemon-classification", "mini", "train")
]

counter = 