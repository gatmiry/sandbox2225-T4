from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from mymodelnew2 import MyModel 
mymodel = MyModel.from_pretrained('posmodel3_weights_2024-07-29--17:41:44alaki')
tokenizer = AutoTokenizer.from_pretrained('gpt2')


class EmbedCalculator():
    def calculate_embedding(input_text):
        with torch.no_grad():
            outputs = tokenizer(input_text, return_tensors='pt')
            outputs['input_ids'] = outputs['input_ids'] + tokenizer('</s>')['input_ids']
            hidden_embedding = mymodel.encoder(outputs['input_ids']).last_hidden_state.detach()[0, outputs['input_ids'].shape[1]-1, :]
        return hidden_embedding
