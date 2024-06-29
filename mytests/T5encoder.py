from transformers import MarianMTModel, MarianTokenizer 
import torch

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus")

input_ids = tokenizer("I want to buy a car", return_tensors='pt').input_ids
output_ids = model.generate(input_ids)[0]
print(tokenizer.decode(output_ids))

