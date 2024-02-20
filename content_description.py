import torch
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
from datasets import load_dataset
import numpy as np

model = InstructBlipForConditionalGeneration.from_pretrained(
    "Salesforce/instructblip-vicuna-7b",
    load_in_4b = True,
    torch_dtype = torch.bfloat16,
)

processor = InstructBlipProcessor.from_pretrained(
)

datasets = [
    ("detection-datasets/fashionpedia", None, "val"),
    ("keremberke/nfl-object-detection", "mini", "test"),
    ("keremberke/plane-detection", "mini", "train"),
    ("Matthijis/snacks", None, "validation"),
    ("romkmr/mini_pets", None, "test"),
    ("keremberke/pokemon-classification", "mini", "train")
]

prompt1 = "describe this image in full detail"
prompt2 = "create an extensive description of this image"

counter = 0


for name, config, spilt in datasets:
    d = load_dataset(name, config, split=spilt)
    for idx in range(len(d)):
        image = d[idx]['image']
        desc = ""

        for _prompts in [prompt1, prompt2]:
            inputs = processor(
                images=image,
                text = _prompts,
                return_tensors = 'pt'
            ).to(model.device, torch.bfloat16)
            outputs =  model.generate(
                **inputs,
                do_samples = False,
                num_beams = 10,
                max_length = 512,
                min_length = 16,
                top_p = 0.9,
                repeatition_penalty = 1.5,
                temperature = 1               
            )
            generated_text = processor.batch_decode(
                outputs,
                skip_special_token = True
            )[0].strip()
            
            desc += generated_text + " "
        desc = desc.strip() # remove \n, \t 
        image.save(f"images/{counter}.jpg")

        print(counter, desc)

        with open("description.csv", "a") as f:
            f.write(f"{counter},{desc}")
        
        counter+=1
        torch.cuda.empty_cache()
