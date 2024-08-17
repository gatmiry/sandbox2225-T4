from transformers import AutoTokenizer
import torch
from mymodelsummary import MyModel
from transformers import DataCollatorForLanguageModeling
import numpy as np
#checkpoint = 'nodotmodel3_weights_sirui'
checkpoint = 'nodotmodel3_weights_2024-08-09--02:19:44alaki' ## second summary model, 4 epochs
#checkpoint = 'nodotmodel3_weights_2024-08-05--20:10:40alaki' ## final variable length model, 4 epochs
#checkpoint = './testmodel3weights_2024-07-07--05:59:32'
#checkpoint = './model3weights_2024-07-04--16:34:15'
device = 'cuda'
model = MyModel.from_pretrained(checkpoint)
model.eval()
model.to(device)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
context_length = 1024 

class EmbedCalculator:
    def __init__(self):
        self.hidden_size = model.config.hidden_size
    print('im here!')
    #def __init__(self, tokenizer):
    #    self.tokenizer = tokenizer
    def calculate_embedding(self, inputs):
        with torch.no_grad():
            #print(inputs[0])
            #print('im right before tokenizer')
            outputs = tokenizer([input_text + tokenizer.eos_token for input_text in inputs], return_length=True)
            #print('im right after tokenizer')
            outputs['input_ids'] = [input_ids for input_ids in outputs['input_ids'] if len(input_ids) <= context_length]
            outputs['length'] =  [length for length in outputs['length'] if length <= context_length]
            input_ids = collator(outputs['input_ids'])['input_ids'].to(device)
            if len(input_ids) > 512:
                print('len input is ', len(input_ids))
            hidden_embedding = model.encoder(input_ids).last_hidden_state.detach()[np.arange(input_ids.shape[0]), [i-1 for i in outputs['length']], :]
        return hidden_embedding

    def calculate_embedding_fromids(self, input_ids):
        with torch.no_grad():
            #print(inputs[0])
            input_ids = collator(input_ids)
            hidden_embedding = model.encoder(input_ids).last_hidden_state.detach()[np.arange(input_ids.shape[0]), [i-1 for i in outputs['length']], :]
        return hidden_embedding